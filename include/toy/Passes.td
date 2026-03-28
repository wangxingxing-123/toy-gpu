#ifndef TOY_PASSES
#define TOY_PASSES

include "mlir/Pass/PassBase.td"

def ShapeInference : Pass<"toy-shape-inference", "mlir::toy::FuncOp"> {
  let summary = "Infer shapes for Toy dialect operations";
  let constructor = "mlir::toy::createShapeInferencePass()";
  let dependentDialects = ["mlir::toy::ToyDialect"];
}

// 关键修复：将这里提升为 ModuleOp，以便在其中替换整个 func
def LowerToAffine : Pass<"toy-lower-affine", "mlir::ModuleOp"> {
  let summary = "Lower Toy dialect to Affine dialect";
  let constructor = "mlir::toy::createLowerToAffinePass()";
  let dependentDialects = ["mlir::affine::AffineDialect", "mlir::memref::MemRefDialect", "mlir::func::FuncDialect"];
}

def LowerToLLVM : Pass<"toy-lower-llvm", "mlir::ModuleOp"> {
  let summary = "Lower Toy dialect to LLVM dialect";
  let constructor = "mlir::toy::createLowerToLLVMPass()";
  let dependentDialects = ["mlir::LLVM::LLVMDialect", "mlir::scf::SCFDialect"];
}

#endif // TOY_PASSES