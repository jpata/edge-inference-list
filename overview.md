# Machine Learning Quantization and Efficient Edge Inference Tools

## The ML-to-Hardware Landscape: A Framework for Analysis

### Introduction

The deployment of machine learning (ML) models onto edge devices presents a significant challenge. A "semantic gap" exists between high-level models created in frameworks like PyTorch and TensorFlow and the low-level, resource-constrained hardware environments. This requires a complex process of optimization, quantization, and compilation to reduce latency, memory footprint, and power consumption.

## The Deployment Pipeline

A typical four-stage workflow for deploying an ML model to hardware:
1.  **Stage 1: Frontend Training & Quantization:** Occurs in the native training framework (e.g., PyTorch, Keras). Involves techniques like Quantization-Aware Training (QAT) and Post-Training Quantization (PTQ).
2.  **Stage 2: Intermediate Representation (IR) & Optimization:** The model is exported to a framework-agnostic format, with ONNX being the *lingua franca*.
3.  **Stage 3: Backend Compilation & Hardware Synthesis:** A specialized tool ingests the IR and performs hardware-specific optimizations, generating a deployable artifact (e.g., C++ code, HLS for FPGAs).
4.  **Stage 4: Hardware-in-the-Loop (HIL) Deployment & Runtime:** A library on the edge device loads the compiled model, manages data, and controls the hardware accelerator.

## ML Inference & Synthesis Toolchains

Here's how the major tools to the stages:

| Toolchain / Tool | Primary Function (Stage) | Supported Frontends | Quantization Strategy | Primary Hardware Target(s) |
| :--- | :--- | :--- | :--- | :--- |
| QKeras / HGQ | Frontend QAT Library (Stage 1) | Keras (TensorFlow, JAX, PyTorch) | Arbitrary-Precision QAT | FPGAs (via hls4ml) |
| Brevitas | Frontend QAT Library (Stage 1) | PyTorch | Arbitrary-Precision QAT | FPGAs (via FINN) |
| TFMOT | Frontend QAT/PTQ Library (Stage 1) | TensorFlow / Keras | 8-bit Integer QAT/PTQ | Mobile CPUs, GPUs, EdgeTPU (via TFLite) |
| PyTorch torch.ao | Frontend QAT/PTQ Library (Stage 1) | PyTorch | 8-bit Integer QAT/PTQ | Mobile CPUs, GPUs (via ExecuTorch) |
| QONNX | Intermediate Representation (Stage 2) | ONNX-based (from Brevitas, QKeras) | Arbitrary-Precision | FPGAs (Hand-off to FINN, hls4ml) |
| hls4ml | NN-to-HLS Compiler (Stage 3) | Keras, PyTorch, ONNX, QONNX | Arbitrary-Precision (from Frontend) | FPGAs (Custom Hardware) |
| FINN | Dataflow NN-to-HLS/RTL Compiler (Stage 3) | QONNX (from Brevitas) | Arbitrary-Precision (from Frontend) | FPGAs (Custom Hardware) |
| Vitis AI | Model Compiler (Stage 3) & Runtime (Stage 4) | TF, PyTorch, ONNX | 8-bit Integer PTQ/QAT | FPGAs / ACAPs (DPU/AIE Overlay) |
| Apache TVM | Heterogeneous Compiler Stack (Stage 3) | All (via ONNX) | 8-bit, Mixed-Precision | CPUs, GPUs, Accelerators (e.g., DPU) |
| Intel OpenVINO | Heterogeneous Compiler (Stage 3) & Runtime (Stage 4) | TF, PyTorch, ONNX | 8-bit Integer | Intel CPUs, iGPUs, VPUs |
| ExecuTorch | Edge Runtime (Stage 4) | PyTorch | 8-bit (from torch.ao) | Mobile CPUs, GPUs, DSPs |
| TensorFlow Lite | Edge Runtime (Stage 4) | TensorFlow | 8-bit (from TFMOT) | Mobile CPUs, GPUs, EdgeTPU |
| PYNQ | Hardware Runtime Environment (Stage 4) | Python | N/A (Controls accelerator) | AMD-Xilinx SoCs (Zynq, Kria) |

## Frontend Quantization-Aware Training (QAT) Frameworks

### Introduction to Quantization-Aware Training (QAT)

While 8-bit Post-Training Quantization (PTQ) is standard for mobile deployment, FPGAs and embedded systems benefit from more aggressive quantization (e.g., 4-bit, 2-bit). QAT simulates quantization effects during training, allowing the model to learn robustness and maintain accuracy at very low bit-widths.

### TensorFlow / Keras Ecosystem

#### QKeras

A Keras extension for "drop-in replacement" layers that define complex, arbitrary-precision quantization schemes. It's the de facto QAT frontend for the `hls4ml` compiler.
*   GitHub: [https://github.com/google/qkeras](https://github.com/google/qkeras)

#### HGQ (High Granularity Quantization)

An advanced QAT algorithm that automatically finds the optimal bit-widths for each parameter, balancing accuracy against a hardware resource budget. It integrates with `hls4ml`.
*   GitHub (Original): [https://github.com/calad0i/HGQ](https://github.com/calad0i/HGQ)
*   GitHub (HGQ2): [https://github.com/calad0i/HGQ2](https://github.com/calad0i/HGQ2)

#### TensorFlow Model Optimization Toolkit (TFMOT)

Google's official toolkit for optimizing models, focusing on 8-bit integer quantization for deployment to mobile CPUs, GPUs, and EdgeTPUs via TensorFlow Lite (TFLite).
*   GitHub: [https://github.com/tensorflow/model-optimization](https://github.com/tensorflow/model-optimization)

### PyTorch Ecosystem

#### Brevitas

A PyTorch-based QAT library from AMD-Xilinx, serving as the preferred frontend for the FINN compiler. It models the reduced-precision hardware data-path at training time.
*   GitHub: [https://github.com/Xilinx/brevitas](https://github.com/Xilinx/brevitas)

#### HAWQ (Hessian Aware Quantization)

A mixed-precision quantization library that uses second-order information (the Hessian) to automatically determine bit-widths for different layers, allocating more bits to more "sensitive" layers.
*   GitHub: [https://github.com/Zhen-Dong/HAWQ](https://github.com/Zhen-Dong/HAWQ)

#### PyTorch Native Quantization (torch.ao) & ExecuTorch

PyTorch's native library (`torch.ao`) focuses on 8-bit PTQ and QAT. For edge deployment, **ExecuTorch** is the modern runtime for running quantized PyTorch models on mobile and embedded devices.
*   GitHub (Quantization): [https://github.com/pytorch/ao](https://github.com/pytorch/ao)
*   GitHub (Edge Runtime): [https://github.com/pytorch/executorch](https://github.com/pytorch/executorch)

### JAX Ecosystem

#### AQT (Accurate Quantized Training)

A Google library for JAX focused on "What you train is what you serve" (WYTIWYS), ensuring a bit-exact match between simulated training quantization and real serving quantization.
*   GitHub: [https://github.com/google/aqt](https://github.com/google/aqt)

#### Qwix

A modern JAX quantization library from Google supporting QAT, PTQ, and modern numerics like int4 and fp8, with a focus on LLMs and accelerators.
*   GitHub: [https://github.com/google/qwix](https://github.com/google/qwix)

## The "Glue": Intermediate Representations

### ONNX (Open Neural Network Exchange)

Allows models to be moved between frameworks. However, its standard quantization support is limited to 8-bit integers, which is insufficient for arbitrary-precision FPGA models.
*   GitHub: [https://github.com/onnx/onnx](https://github.com/onnx/onnx)

### QONNX (Quantized ONNX)

An extension to ONNX created by the FINN and hls4ml communities. It adds custom operators to represent arbitrary-precision quantization, preserving this crucial information for FPGA backend compilers.
*   GitHub: [https://github.com/fastmachinelearning/qonnx](https://github.com/fastmachinelearning/qonnx)

### ONNX Runtime

A cross-platform compiler and runtime from Microsoft that can ingest a floating-point ONNX model and perform its own 8-bit quantization, competing with TVM and OpenVINO.
*   GitHub: [https://github.com/microsoft/onnxruntime](https://github.com/microsoft/onnxruntime)

## End-to-End Compilers for FPGA Synthesis

### Custom FPGA layout vs. runtime instructions

1.  **"Custom layout:** Used by `hls4ml` and `FINN`, this approach generates new, specialized hardware logic for the neural network, offering the lowest latency at the cost of slow hardware synthesis times.
2.  **Runtime overlay:** Used by `Vitis AI`, this approach uses a pre-built, fixed AI accelerator (DPU or AIE) on the FPGA. The NN is compiled into instructions for this runtime, offering a faster, more "software-like" workflow.

| Philosophy | Key Tools | Core Technology | Workflow | Pros | Cons |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Custom layout | hls4ml, FINN | NN-to-HLS/RTL Compilation | (Slow) Generate HLS/RTL -> Synthesize | Ultra-low latency, arbitrary-precision | Potentially slow build times, complex |
| Runtime instructions | Vitis AI | DPU / AIE Accelerators | (Fast) Compile NN -> Load instructions | "Software-like" workflow, fast iteration | Higher latency, less specialized |

### "Custom layout" Philosophy: hls4ml

A Python package with roots in high-energy physics, built for ultra-low latency. It translates models into fully unrolled and pipelined HLS C++ code, creating a massive, parallel hardware circuit.
*   GitHub: [https://github.com/fastmachinelearning/hls4ml](https://github.com/fastmachinelearning/hls4ml)

### "Custom layout" Philosophy: FINN

A dataflow compiler from AMD-Xilinx that builds accelerators by stitching together pre-optimized, composable hardware blocks, leading to more resource-efficient designs. It is the backend for the Brevitas/QONNX pipeline.
*   GitHub: [https://github.com/Xilinx/finn](https://github.com/Xilinx/finn)
*   HLS Library: [https://github.com/Xilinx/finn-hlslib](https://github.com/Xilinx/finn-hlslib)

### "Runtime overlay" Philosophy: Vitis AI & AI Engines

AMD-Xilinx's official development stack. It uses a "Deep-Learning Processor Unit (DPU)" or "AI Engines (AIE)" as a pre-existing accelerator. The user compiles their NN to run on this hardware, providing a mainstream, user-friendly path for FPGA acceleration.
*   GitHub: [https://github.com/Xilinx/Vitis-AI](https://github.com/Xilinx/Vitis-AI)

## General-Purpose Compilers for Heterogeneous Edge Inference

These compilers target a wide range of hardware, not just FPGAs.

### Apache TVM

An open-source ML compilation framework that can ingest models from any framework (via ONNX) and generate high-performance machine code for any target, from ARM CPUs to GPUs and specialized accelerators.
*   GitHub: [https://github.com/apache/tvm](https://github.com/apache/tvm)

### Intel OpenVINO

Intel's open-source toolkit for optimizing and deploying AI inference on the Intel ecosystem (CPUs, iGPUs, VPUs). It provides a vertically-integrated, all-in-one solution.
*   GitHub: [https://github.com/openvinotoolkit/openvino](https://github.com/openvinotoolkit/openvino)

### Glow

A machine learning compiler from Meta/PyTorch that uses Ahead-of-Time (AOT) compilation to generate a self-contained, dependency-free executable bundle, ideal for embedded systems.
*   GitHub: [https://github.com/pytorch/glow](https://github.com/pytorch/glow)

## Advanced Optimization & Specialized Tooling

### Hardware-Aware Pruning & Quantization: pQuant

A tool for "End-to-End Hardware-Aware Model Compression" that streamlines both pruning and quantization before feeding a model to a backend like `hls4ml`.
*   GitHub: [https://github.com/nroope/PQuant](https://github.com/nroope/PQuant)

### Low-Level Arithmetic Optimization: da4ml

A library for "Distributed Arithmetic (DA)," a "multiplier-less" technique for performing matrix-vector multiplication using only efficient adders and Look-Up Tables (LUTs), trading scarce DSP blocks for more abundant LUTs on an FPGA.
*   GitHub: [https://github.com/calad0i/da4ml](https://github.com/calad0i/da4ml)

### Pre-Synthesis Resource & Latency Estimation: rule4ml

Solves the slow hardware synthesis problem by providing near-instant, pre-synthesis estimates of FPGA resource utilization and latency using its own ML models.
*   GitHub: [https://github.com/IMPETUS-UdeS/rule4ml](https://github.com/IMPETUS-UdeS/rule4ml)

## Alternative Model Paradigms

### Boosted Decision Trees: conifer

A "sister project" to `hls4ml` that translates trained Boosted Decision Trees (from scikit-learn, xgboost, etc.) into FPGA firmware for extreme low-latency inference.
*   GitHub: [https://github.com/thesps/conifer](https://github.com/thesps/conifer)

### Symbolic Regression: The "Ultimate" Compression

Instead of compressing a large NN, Symbolic Regression (SR) tools discover a simple, interpretable symbolic formula that fits the data. This formula is trivial to implement in HLS, resulting in a tiny and efficient hardware model.
*   **PySR:** A high-performance SR library using genetic programming. [https://github.com/MilesCranmer/PySR](https://github.com/MilesCranmer/PySR)
*   **SymbolNet:** A neural symbolic regression tool that uses an NN to help discover the symbolic formula. [https://github.com/hftsoi/SymbolNet](https://github.com/hftsoi/SymbolNet)

## Hardware Abstraction, Runtimes, and Deployment

### PYNQ (Python Productivity for Zynq)

An open-source project from AMD-Xilinx providing a bootable Linux environment for its SoC platforms. It allows users to control FPGA accelerators (like a Vitis AI DPU or an `hls4ml` design) from Python code running on the ARM CPU.
*   GitHub: [https://github.com/Xilinx/PYNQ](https://github.com/Xilinx/PYNQ)

### DPU-PYNQ

The specific PYNQ package that acts as the driver for the Vitis AI DPU, connecting the Python software world to the DPU hardware.
*   GitHub: [https://github.com/Xilinx/DPU-PYNQ](https://github.com/Xilinx/DPU-PYNQ)

### Vivado / Vitis

The "ground truth" AMD-Xilinx hardware design suites. Compilers like `hls4ml` and `FINN` generate HLS code that is fed into Vitis/Vivado for the final, time-consuming synthesis into a hardware bitstream.

## Workflow

*   **Goal: Mobile phones (ARM CPUs, mobile GPUs).**
    *   **TensorFlow:** Use TFMOT and deploy with TensorFlow Lite.
    *   **PyTorch:** Use `torch.ao` and deploy with ExecuTorch.
*   **Goal: Intel edge devices (CPUs, iGPUs, VPUs).**
    *   Use the Intel OpenVINO toolchain.
*   **Goal: Many different heterogeneous edge devices.**
    *   Use a general-purpose compiler like Apache TVM or ONNX Runtime.
*   **Goal: Xilinx/AMD FPGA with an easy, "software-like" workflow.**
    *   Use the "Overlay" philosophy with Vitis AI and the DPU.
*   **Goal: Xilinx/AMD FPGA with the absolute lowest latency.**
    *   Use the "Custom Hardware" philosophy.
    *   **Keras:** Use QKeras/HGQ with `hls4ml`.
    *   **PyTorch:** Use Brevitas -> QONNX -> FINN.
*   **Goal: "My `hls4ml` synthesis is too slow."**
    *   Use `rule4ml` for pre-synthesis estimation.
*   **Goal: "My `hls4ml` design is out of DSPs."**
    *   Use `da4ml` to trade DSPs for LUTs.
*   **Goal: "My model is a Boosted Decision Tree."**
    *   Use `conifer`.
*   **Goal: "I need a simpler, interpretable model."**
    *   Explore Symbolic Regression with PySR or SymbolNet.
