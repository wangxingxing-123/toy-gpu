#ifndef TOY_PASSES_H
#define TOY_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include <memory>

namespace mlir {
    namespace toy {

        std::unique_ptr<Pass> createShapeInferencePass();
        std::unique_ptr<Pass> createLowerToAffinePass();
        std::unique_ptr<Pass> createLowerToLLVMPass();

#define GEN_PASS_DECL
#include "toy/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "toy/Passes.h.inc"

    } // namespace toy
} // namespace mlir

#endif // TOY_PASSES_H