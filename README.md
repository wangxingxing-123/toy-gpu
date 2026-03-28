# toy-gpu
MLIR Toy Compiler Extended with NVIDIA GPU Acceleration | 支持英伟达GPU的MLIR Toy编译器
MLIR-Toy-GPU
An extended implementation of the MLIR official Toy tutorial with full NVIDIA GPU compilation support.
This project ports the classic MLIR Toy language compiler to NVIDIA GPUs by integrating the MLIR GPU, NVVM, LLVM, and all core standard dialects. All dialects are statically linked, requiring no dynamic plugin loading. It provides a complete compilation pipeline: Toy language → MLIR IR → GPU NVVM → Executable binary.
Key Features

    Based on MLIR Official Toy Tutorial
    Full NVIDIA GPU acceleration support (GPU + NVVM Dialects)
    Statically linked all core dialects: toy, func, arith, memref, scf, affine, gpu, nvvm, llvm
    No plugin loading required (zero command-line parameters)
    Complete compilation pipeline for CPU/GPU heterogeneous computing
    Ideal for learning MLIR, GPU compiler development, and dialect extension

Tech Stack

    MLIR (LLVM)
    NVIDIA GPU / NVVM
    C++
    CMake

MLIR-Toy-GPU
基于 MLIR 官方 Toy 教程 扩展的 英伟达 GPU 加速编译器。
本项目在原版 Toy 编译器基础上，完成了英伟达 GPU 编译适配，集成 MLIR 全量核心方言（GPU/NVVM/LLVM/func/arith 等），所有方言静态链接，无需动态加载插件，实现了完整的编译流水线：
Toy 语言 → MLIR IR → GPU 底层指令 → 可执行程序。
核心特性

    基于 MLIR 官方 Toy 教程开发，学习友好
    完整支持 NVIDIA GPU 硬件加速 编译
    静态集成 toy/func/arith/memref/scf/affine/gpu/nvvm/llvm 全方言
    零插件加载，无命令行参数依赖
    支持 CPU/GPU 异构编译
    适用于 MLIR 入门、GPU 编译器开发、方言扩展学习

技术栈

    MLIR (LLVM 编译器框架)
    NVIDIA GPU / NVVM 底层方言
    C++ / CMake
