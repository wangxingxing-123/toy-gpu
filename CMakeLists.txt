cmake_minimum_required(VERSION 3.15)
project(toy-gpu CXX C)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 寻找 LLVM 和 MLIR 核心库
if(NOT DEFINED MLIR_DIR)
    set(MLIR_DIR "/home/wangxinxing/桌面/llvm-project/build/lib/cmake/mlir")
endif()
find_package(MLIR REQUIRED CONFIG)

# 引入 MLIR 和 LLVM 的 CMake 工具宏
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)

# =====================================================================
# 关键修复 1：提前在 build 目录下建好文件夹，防止 tblgen 找不到输出路径
# =====================================================================
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/toy)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/mlir)

# 设置头文件搜索路径
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/mlir)

# =========================================================================
# TableGen 自动生成逻辑
# =========================================================================
set(MLIR_TABLEGEN_FLAGS -I ${MLIR_INCLUDE_DIRS} -I ${CMAKE_CURRENT_SOURCE_DIR}/include)

# 1. ShapeInference
set(LLVM_TARGET_DEFINITIONS ${CMAKE_CURRENT_SOURCE_DIR}/include/toy/ShapeInferenceInterface.td)
mlir_tablegen(include/toy/ShapeInferenceOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(include/toy/ShapeInferenceOpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(ToyShapeInferenceGen)

# 2. Ops & Dialect
set(LLVM_TARGET_DEFINITIONS ${CMAKE_CURRENT_SOURCE_DIR}/include/toy/Ops.td)
mlir_tablegen(include/toy/Dialect.h.inc -gen-dialect-decls -dialect=toy)
mlir_tablegen(include/toy/Dialect.cpp.inc -gen-dialect-defs -dialect=toy)
mlir_tablegen(include/toy/Ops.h.inc -gen-op-decls)
mlir_tablegen(include/toy/Ops.cpp.inc -gen-op-defs)
add_public_tablegen_target(ToyOpsGen)

# 3. Passes (使用我们刚手写的 Passes.td)
set(LLVM_TARGET_DEFINITIONS ${CMAKE_CURRENT_SOURCE_DIR}/include/toy/Passes.td)
mlir_tablegen(include/toy/Passes.h.inc -gen-pass-decls -name Toy)
add_public_tablegen_target(ToyPassesGen)

# 4. ToyCombine
set(LLVM_TARGET_DEFINITIONS ${CMAKE_CURRENT_SOURCE_DIR}/mlir/ToyCombine.td)
mlir_tablegen(mlir/ToyCombine.inc -gen-rewriters)
add_public_tablegen_target(ToyCombineGen)

# =====================================================================
# 关键修复 2：将所有 TableGen 目标打包成一个超级目标，强制绑定依赖
# =====================================================================
add_custom_target(ToyAllTableGenTargets
    DEPENDS ToyShapeInferenceGen ToyOpsGen ToyPassesGen ToyCombineGen
)

# 收集源文件
file(GLOB TOY_SRC "mlir/*.cpp")

# MLIR 链接库 (修复了 registerAllExtensions 和 GPU 降级库的名字)
set(MLIR_LIBS
    MLIRAnalysis MLIRCastInterfaces MLIRCallInterfaces MLIRControlFlowDialect
    MLIRFuncDialect MLIRIR MLIRParser MLIRPass MLIRSideEffectInterfaces
    MLIRTransforms MLIRTransformUtils MLIRSupport MLIRTargetLLVMIRExport
    MLIRExecutionEngine MLIRToLLVMIRTranslationRegistration
    MLIRAffineToStandard MLIRSCFToControlFlow MLIRFuncAllExtensions
    MLIRSCFToGPU MLIRGPUTransforms
    MLIRGPUToNVVMTransforms MLIRNVVMDialect MLIRGPUToLLVMIRTranslation
    MLIRGPUToGPURuntimeTransforms MLIRReconcileUnrealizedCasts
)

# 生成可执行文件
add_executable(toyc toyc.cpp ${TOY_SRC})
llvm_update_compile_flags(toyc)

# =========================================================================
# 终极修复：精确点名我们需要 X86 和 NVPTX 底层硬件库，抛弃有 bug 的 all-targets
# =========================================================================
llvm_map_components_to_libnames(LLVM_TARGET_LIBS X86 NVPTX AsmParser AsmPrinter)

# 将找齐的 LLVM_TARGET_LIBS 追加到我们的链接列表中
target_link_libraries(toyc PRIVATE ${MLIR_LIBS} ${LLVM_TARGET_LIBS})

# 强制 toyc 在编译任何 C++ 代码前，必须先完成超级目标（生成所有 .inc）
add_dependencies(toyc ToyAllTableGenTargets)