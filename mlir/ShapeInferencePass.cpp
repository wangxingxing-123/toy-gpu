#include "toy/Dialect.h"
#include "toy/Passes.h"
#include "toy/ShapeInferenceInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <memory>

namespace mlir {
  namespace toy {

    // 注意：GEN_PASS_DEF 必须在这个位置！
#define GEN_PASS_DEF_SHAPEINFERENCE
#include "toy/Passes.h.inc"

    namespace {
      static bool returnsDynamicShape(Operation *op) {
        return llvm::any_of(op->getResultTypes(), [](Type type) {
          return !llvm::cast<TensorType>(type).hasStaticShape();
        });
      }

      static bool allResultShapesKnown(Operation *op) {
        return llvm::all_of(op->getResultTypes(), [](Type type) {
          return llvm::cast<TensorType>(type).hasStaticShape();
        });
      }

      class ShapeInferencePass : public impl::ShapeInferenceBase<ShapeInferencePass> {
      public:
        using impl::ShapeInferenceBase<ShapeInferencePass>::ShapeInferenceBase;
        void runOnOperation() override {
          auto f = getOperation();
          llvm::SmallPtrSet<Operation *, 16> opWorklist;
          f.walk([&](Operation *op) {
            if (returnsDynamicShape(op)) opWorklist.insert(op);
          });

          while (!opWorklist.empty()) {
            auto *nextOp = *opWorklist.begin();
            opWorklist.erase(nextOp);
            if (!allResultShapesKnown(nextOp)) {
              if (auto shapeOp = llvm::dyn_cast<ShapeInference>(nextOp))
                shapeOp.inferShapes();
              else {
                nextOp->emitError("unable to infer shape");
                return signalPassFailure();
              }
            }
          }
        }
      };
    } // namespace

    std::unique_ptr<Pass> createShapeInferencePass() {
      return std::make_unique<ShapeInferencePass>();
    }

  } // namespace toy
} // namespace mlir