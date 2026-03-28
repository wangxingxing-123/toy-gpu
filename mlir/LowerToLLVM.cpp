#include "toy/Dialect.h"
#include "toy/Passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/PassManager.h"
#include <memory>

namespace mlir {
namespace toy {

#define GEN_PASS_DEF_LOWERTOLLVM
#include "toy/Passes.h.inc"

namespace {
class PrintOpLowering : public ConversionPattern {
public:
  explicit PrintOpLowering(MLIRContext *context)
      : ConversionPattern(toy::PrintOp::getOperationName(), 1, context) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

class ToyToLLVMLoweringPass : public impl::LowerToLLVMBase<ToyToLLVMLoweringPass> {
public:
  using impl::LowerToLLVMBase<ToyToLLVMLoweringPass>::LowerToLLVMBase;
  void runOnOperation() override {
    ModuleOp module = getOperation();

    mlir::registerGPUDialectTranslation(*module->getContext());
    mlir::registerNVVMDialectTranslation(*module->getContext());

    LLVMTypeConverter typeConverter(&getContext());
    RewritePatternSet patterns(&getContext());

    populateAffineToStdConversionPatterns(patterns);
    populateSCFToControlFlowConversionPatterns(patterns);
    arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
    cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
    populateFuncToLLVMConversionPatterns(typeConverter, patterns);
    populateGpuToLLVMConversionPatterns(typeConverter, patterns);
    patterns.add<PrintOpLowering>(&getContext());

    // ── 第一阶段：转换所有可转换的 op ──────────────────────────────────────
    // UnrealizedConversionCast / gpu / nvvm 暂时放行，第二阶段再清理
    LLVMConversionTarget target(getContext());
    target.addLegalOp<ModuleOp>();
    target.addLegalOp<UnrealizedConversionCastOp>();
    target.addLegalDialect<mlir::gpu::GPUDialect>();
    target.addLegalDialect<mlir::NVVM::NVVMDialect>();

    if (failed(applyFullConversion(module, target, std::move(patterns))))
      return signalPassFailure();

    // ── 第二阶段：消除残留的 unrealized_conversion_cast ─────────────────────
    // cast 消除后，gpu.launch_func 的参数变成纯 llvm 类型，
    // GpuToLLVM 的 pattern 才能真正匹配并把它变成 cuLaunchKernel 调用
    PassManager pm(module->getContext(), module.getOperationName());
    pm.addPass(createReconcileUnrealizedCastsPass());
    if (failed(pm.run(module)))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> createLowerToLLVMPass() {
  return std::make_unique<ToyToLLVMLoweringPass>();
}

} // namespace toy
} // namespace mlir