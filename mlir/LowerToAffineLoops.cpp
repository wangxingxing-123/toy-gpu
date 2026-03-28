#include "toy/Dialect.h"
#include "toy/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include <memory>

namespace mlir {
namespace toy {

#define GEN_PASS_DEF_LOWERTOAFFINE
#include "toy/Passes.h.inc"

namespace {

class FuncOpLowering : public OpConversionPattern<toy::FuncOp> {
public:
  using OpConversionPattern<toy::FuncOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(toy::FuncOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto nameAttr = op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());
    auto newFuncOp = rewriter.create<func::FuncOp>(
        op->getLoc(), nameAttr.getValue(), op.getFunctionType());

    rewriter.inlineRegionBefore(op.getBody(), newFuncOp.getBody(), newFuncOp.end());
    rewriter.eraseOp(op);
    return success();
  }
};

struct ConstantOpLowering : public OpConversionPattern<toy::ConstantOp> {
  using OpConversionPattern<toy::ConstantOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(toy::ConstantOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    DenseElementsAttr constantValue = op.getValue();
    Location loc = op.getLoc();

    auto tensorType = llvm::cast<RankedTensorType>(op.getType());
    auto memRefType = MemRefType::get(tensorType.getShape(), tensorType.getElementType());
    auto alloc = rewriter.create<memref::AllocOp>(loc, memRefType);

    auto valueShape = memRefType.getShape();
    SmallVector<Value, 8> constantIndices;

    if (!valueShape.empty()) {
      for (auto i : llvm::seq<int64_t>(
               0, *std::max_element(valueShape.begin(), valueShape.end())))
        constantIndices.push_back(rewriter.create<arith::ConstantIndexOp>(loc, i));
    } else {
      constantIndices.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
    }

    SmallVector<Value, 2> indices;
    auto valueIt = constantValue.value_begin<FloatAttr>();
    std::function<void(uint64_t)> storeElements = [&](uint64_t dimension) {
      if (dimension == valueShape.size()) {
        rewriter.create<affine::AffineStoreOp>(
            loc, rewriter.create<arith::ConstantOp>(loc, *valueIt++), alloc,
            llvm::ArrayRef<Value>(indices));
        return;
      }
      for (uint64_t i = 0, e = valueShape[dimension]; i != e; ++i) {
        indices.push_back(constantIndices[i]);
        storeElements(dimension + 1);
        indices.pop_back();
      }
    };

    storeElements(0);
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

template <typename LoweredBinaryOp>
struct BinaryOpLowering : public ConversionPattern {
  BinaryOpLowering(TypeConverter &typeConverter, StringRef rootOpName, MLIRContext *context)
      : ConversionPattern(typeConverter, rootOpName, 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto tensorType = llvm::cast<RankedTensorType>(op->getResultTypes()[0]);
    auto shape = tensorType.getShape();
    auto memRefType = MemRefType::get(shape, tensorType.getElementType());
    auto alloc = rewriter.create<memref::AllocOp>(loc, memRefType);

    SmallVector<Value, 4> loopIvs;

    // 神仙级：递归生成任意维度的 affine.for 循环！
    std::function<void(unsigned, ConversionPatternRewriter&)> buildLoops = [&](unsigned dim, ConversionPatternRewriter &builder) {
      if (dim == shape.size()) {
        auto lhs = builder.create<affine::AffineLoadOp>(loc, operands[0], loopIvs);
        auto rhs = builder.create<affine::AffineLoadOp>(loc, operands[1], loopIvs);
        auto result = builder.create<LoweredBinaryOp>(loc, lhs, rhs);
        builder.create<affine::AffineStoreOp>(loc, result, alloc, loopIvs);
        return;
      }
      auto loop = builder.create<affine::AffineForOp>(loc, 0, shape[dim]);
      loopIvs.push_back(loop.getInductionVar());

      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(loop.getBody());
      buildLoops(dim + 1, builder);

      loopIvs.pop_back();
    };

    buildLoops(0, rewriter);
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

struct TransposeOpLowering : public OpConversionPattern<toy::TransposeOp> {
  using OpConversionPattern<toy::TransposeOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(toy::TransposeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto tensorType = llvm::cast<RankedTensorType>(op.getType());
    auto shape = tensorType.getShape();
    auto memRefType = MemRefType::get(shape, tensorType.getElementType());
    auto alloc = rewriter.create<memref::AllocOp>(loc, memRefType);

    SmallVector<Value, 4> loopIvs;

    std::function<void(unsigned, ConversionPatternRewriter&)> buildLoops = [&](unsigned dim, ConversionPatternRewriter &builder) {
      if (dim == shape.size()) {
        SmallVector<Value, 4> reversedIvs(llvm::reverse(loopIvs));
        auto val = builder.create<affine::AffineLoadOp>(loc, adaptor.getOperands()[0], reversedIvs);
        builder.create<affine::AffineStoreOp>(loc, val, alloc, loopIvs);
        return;
      }
      auto loop = builder.create<affine::AffineForOp>(loc, 0, shape[dim]);
      loopIvs.push_back(loop.getInductionVar());

      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(loop.getBody());
      buildLoops(dim + 1, builder);

      loopIvs.pop_back();
    };

    buildLoops(0, rewriter);
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

struct ReturnOpLowering : public OpConversionPattern<toy::ReturnOp> {
  using OpConversionPattern<toy::ReturnOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(toy::ReturnOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getOperands());
    return success();
  }
};

struct CastOpLowering : public OpConversionPattern<toy::CastOp> {
  using OpConversionPattern<toy::CastOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(toy::CastOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getOperands()[0]);
    return success();
  }
};

struct PrintOpLowering : public OpConversionPattern<toy::PrintOp> {
  using OpConversionPattern<toy::PrintOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(toy::PrintOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto memrefType = cast<MemRefType>(adaptor.getOperands()[0].getType());
    auto module = op->getParentOfType<ModuleOp>();

    // 把 print 变成标准的 C 函数调用，完美避开材料化错误！
    auto printSym = "printMemrefF64";
    auto printFunc = module.lookupSymbol<func::FuncOp>(printSym);
    if (!printFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      auto funcType = rewriter.getFunctionType(memrefType, {});
      printFunc = rewriter.create<func::FuncOp>(loc, printSym, funcType);
      printFunc.setPrivate();
    }

    rewriter.create<func::CallOp>(loc, printFunc, adaptor.getOperands()[0]);
    rewriter.eraseOp(op);
    return success();
  }
};

class ToyToAffineLoweringPass : public impl::LowerToAffineBase<ToyToAffineLoweringPass> {
public:
  using impl::LowerToAffineBase<ToyToAffineLoweringPass>::LowerToAffineBase;
  void runOnOperation() override {
    auto module = getOperation();

    // 核心转换器：正式向 MLIR 宣布，所有 Tensor 都在这一步变成了 MemRef！
    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    typeConverter.addConversion([](RankedTensorType type) -> Type {
      return MemRefType::get(type.getShape(), type.getElementType());
    });

    ConversionTarget target(getContext());
    target.addLegalDialect<affine::AffineDialect, BuiltinDialect, arith::ArithDialect,
                           func::FuncDialect, memref::MemRefDialect>();
    target.addIllegalDialect<toy::ToyDialect>();

    RewritePatternSet patterns(&getContext());
    patterns.add<FuncOpLowering, ConstantOpLowering, CastOpLowering,
                 ReturnOpLowering, PrintOpLowering, TransposeOpLowering>(typeConverter, &getContext());

    patterns.add<BinaryOpLowering<arith::AddFOp>>(typeConverter, toy::AddOp::getOperationName(), &getContext());
    patterns.add<BinaryOpLowering<arith::MulFOp>>(typeConverter, toy::MulOp::getOperationName(), &getContext());

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> createLowerToAffinePass() {
  return std::make_unique<ToyToAffineLoweringPass>();
}

} // namespace toy
} // namespace mlir