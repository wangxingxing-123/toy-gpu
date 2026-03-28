#include "toy/AST.h"
#include "toy/Dialect.h"
#include "toy/Lexer.h"
#include "toy/MLIRGen.h"
#include "toy/Parser.h"
#include "toy/Passes.h"

#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Affine/Transforms/Passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"

#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"

#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"   // 新增
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"      // 新增
#include "mlir/Target/LLVMIR/Export.h"

#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/IR/Module.h"                  // 新增，用于 DumpLLVMIR
#include "llvm/IR/LLVMContext.h"             // 新增，用于 LLVMContext
#include <memory>

namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional, cl::desc("<input toy file>"), cl::init("-"));
enum Action { None, DumpAST, DumpMLIR, DumpMLIRAffine, DumpMLIRLLVM, DumpLLVMIR, RunJIT };
static cl::opt<enum Action> emitAction("emit", cl::init(None), cl::values(
    clEnumValN(DumpAST,       "ast",        "AST"),
    clEnumValN(DumpMLIR,      "mlir",       "MLIR"),
    clEnumValN(DumpMLIRAffine,"mlir-affine","Affine"),
    clEnumValN(DumpMLIRLLVM,  "mlir-llvm",  "LLVM Dialect"),
    clEnumValN(DumpLLVMIR,    "llvm",       "LLVM IR"),
    clEnumValN(RunJIT,        "jit",        "JIT")));
static cl::opt<bool> enableOpt("opt", cl::desc("Enable optimizations"));

// 给 GPU 模块贴上 NVVM 编译目标的标签
namespace {
class AttachTargetPass : public mlir::PassWrapper<AttachTargetPass, mlir::OperationPass<mlir::gpu::GPUModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AttachTargetPass)
  void runOnOperation() override {
    auto module = getOperation();
    auto target = mlir::NVVM::NVVMTargetAttr::get(module.getContext());
    module->setAttr("targets", mlir::ArrayAttr::get(module.getContext(), {target}));
  }
};
} // namespace

static std::unique_ptr<toy::ModuleAST> parseInputFile(llvm::StringRef filename) {
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(filename);
    if (std::error_code ec = fileOrErr.getError()) return nullptr;
    auto buffer = fileOrErr.get()->getBuffer();
    toy::LexerBuffer lexer(buffer.begin(), buffer.end(), std::string(filename));
    toy::Parser parser(lexer);
    return parser.parseModule();
}

static int loadMLIR(mlir::MLIRContext &context, mlir::OwningOpRef<mlir::ModuleOp> &module) {
    auto moduleAST = parseInputFile(inputFilename);
    if (!moduleAST) return 6;
    module = toy::mlirGen(context, *moduleAST);
    return !module ? 1 : 0;
}

static int loadAndProcessMLIR(mlir::MLIRContext &context, mlir::OwningOpRef<mlir::ModuleOp> &module) {
    if (loadMLIR(context, module)) return -1;
    mlir::PassManager pm(module.get()->getName());
    if (mlir::failed(mlir::applyPassManagerCLOptions(pm))) return 4;

    // ── 优化 & Shape Inference ──────────────────────────────────────────────
    if (enableOpt || emitAction >= DumpMLIRAffine) {
        pm.addPass(mlir::createInlinerPass());
        auto &optPM = pm.nest<mlir::toy::FuncOp>();
        optPM.addPass(mlir::toy::createShapeInferencePass());
        optPM.addPass(mlir::createCanonicalizerPass());
        optPM.addPass(mlir::createCSEPass());
    }

    // ── Toy → Affine ────────────────────────────────────────────────────────
    if (emitAction >= DumpMLIRAffine) {
        pm.addPass(mlir::toy::createLowerToAffinePass());
    }

    // ── Affine → GPU → NVVM binary → LLVM Dialect ───────────────────────────
    if (emitAction >= DumpMLIRLLVM) {
        // 1. Affine for 循环 → gpu.launch
        auto &funcPM = pm.nest<mlir::func::FuncOp>();
        funcPM.addPass(mlir::createConvertAffineForToGPUPass());
        funcPM.addPass(mlir::createLowerAffinePass());

        // 2. gpu.launch → gpu.launch_func + gpu.module（kernel outline）
        pm.addPass(mlir::createGpuKernelOutliningPass());

        // 3. GPU module 内部：SCF → CF → NVVM ops → binary blob
        auto &gpuPM = pm.nest<mlir::gpu::GPUModuleOp>();
        gpuPM.addPass(std::make_unique<AttachTargetPass>());
        gpuPM.addPass(mlir::createSCFToControlFlowPass());
        gpuPM.addPass(mlir::createConvertGpuOpsToNVVMOps());
        gpuPM.addPass(mlir::createCanonicalizerPass());   // 新增：清理残留 NVVM ops
        gpuPM.addPass(mlir::createGpuModuleToBinaryPass());   // → PTX/cubin blob

        // 4. 外层：gpu.launch_func + 其余所有方言 → LLVM Dialect
        //    （LowerToLLVM.cpp 内部已调用 populateGpuToLLVMConversionPatterns）
        pm.addPass(mlir::toy::createLowerToLLVMPass());
    }

    return mlir::failed(pm.run(*module)) ? 4 : 0;
}

int main(int argc, char **argv) {
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmParsers();
    llvm::InitializeAllAsmPrinters();

    mlir::registerPassManagerCLOptions();
    cl::ParseCommandLineOptions(argc, argv, "toy compiler\n");

    mlir::DialectRegistry registry;
    mlir::func::registerAllExtensions(registry);

    registry.insert<mlir::toy::ToyDialect,
                    mlir::scf::SCFDialect,
                    mlir::gpu::GPUDialect,
                    mlir::LLVM::LLVMDialect,
                    mlir::NVVM::NVVMDialect,
                    mlir::affine::AffineDialect,
                    mlir::memref::MemRefDialect,
                    mlir::arith::ArithDialect,
                    mlir::func::FuncDialect>();

    // 注册所有需要的翻译接口
    mlir::registerGPUDialectTranslation(registry);
    mlir::registerNVVMDialectTranslation(registry);      // 新增
    mlir::registerBuiltinDialectTranslation(registry);   // 移到这里，统一注册
    mlir::registerLLVMDialectTranslation(registry);      // 移到这里，统一注册

    mlir::MLIRContext context(registry);
    context.loadAllAvailableDialects();

    mlir::OwningOpRef<mlir::ModuleOp> module;
    if (loadAndProcessMLIR(context, module)) return -1;

    // ── 输出 MLIR Dialect ───────────────────────────────────────────────────
    if (emitAction <= DumpMLIRLLVM) {
        module->print(llvm::outs());
        return 0;
    }

    // ── 输出 LLVM IR（文本格式）────────────────────────────────────────────
    if (emitAction == DumpLLVMIR) {
        llvm::LLVMContext llvmContext;
        auto llvmModule = mlir::translateModuleToLLVMIR(*module, llvmContext);
        if (!llvmModule) {
            llvm::errs() << "Failed to translate to LLVM IR\n";
            return -1;
        }
        llvmModule->print(llvm::outs(), nullptr);
        return 0;
    }

    // ── JIT 执行 ────────────────────────────────────────────────────────────
    if (emitAction == RunJIT) {
        mlir::ExecutionEngineOptions engineOptions;
        llvm::SmallVector<llvm::StringRef, 1> sharedLibs;
        sharedLibs.push_back("/home/wangxinxing/桌面/llvm-project/build/lib/libmlir_cuda_runtime.so");
        engineOptions.sharedLibPaths = sharedLibs;

        auto engine = mlir::ExecutionEngine::create(*module, engineOptions);
        if (!engine) {
            llvm::errs() << "Failed to create execution engine\n";
            return -1;
        }
        if ((*engine)->invokePacked("main")) {
            llvm::errs() << "JIT execution failed\n";
            return -1;
        }
        return 0;
    }

    return 0;
}