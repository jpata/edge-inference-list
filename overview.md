# Summary of ML quantization and edge inference tools

### Introduction

Deploying machine learning (ML) models to edge devices requires bridging the gap between high-level models from frameworks like PyTorch and TensorFlow and low-level, resource-constrained hardware. This is done through optimization, quantization, and compilation to reduce latency, memory usage, and power consumption.

## Deployment

A typical four-stage workflow for deploying an ML model to hardware:
1.  **Stage 1: Frontend Training & Quantization:** Done in the original training framework (e.g., PyTorch, Keras), using techniques like Quantization-Aware Training (QAT) and Post-Training Quantization (PTQ).
2.  **Stage 2: Intermediate Representation (IR) & Optimization:** The model is exported to a framework-agnostic format, like ONNX.
3.  **Stage 3: Backend Compilation & Hardware Synthesis:** A tool compiles the IR and performs hardware-specific optimizations, generating a deployable artifact (e.g., C++ code, HLS for FPGAs).
4.  **Stage 4: Hardware-in-the-Loop (HIL) Deployment & Runtime:** A library on the edge device loads the compiled model, manages data, and runs the hardware accelerator.

## Toolchains

Here's how the major tools map to the stages:

| Toolchain / Tool | Supported Frontends | Quantization Strategy | Primary Hardware Target(s) |
| :--- | :--- | :--- | :--- |
| **Stage 1: Frontend Training & Quantization** |
| QKeras / HGQ | Keras (TensorFlow, JAX, PyTorch) | Arbitrary-Precision QAT | FPGAs (via hls4ml) |
| Brevitas | PyTorch | Arbitrary-Precision QAT | FPGAs (via FINN) |
| TFMOT | TensorFlow / Keras | 8-bit Integer QAT/PTQ | Mobile CPUs, GPUs, NPUs (via LiteRT) |
| torchao | PyTorch | Advanced QAT/PTQ (INT4/8, FP8, Sparsity) | Mobile CPUs, GPUs, NPUs (via ExecuTorch) |
| **Stage 2: Intermediate Representation (IR) & Optimization** |
| QONNX | ONNX-based (from Brevitas, QKeras) | Arbitrary-Precision, Minifloats | FPGAs (FINN, hls4ml) |
| **Stage 3: Backend Compilation & Hardware Synthesis** |
| hls4ml | Keras, PyTorch, ONNX, QONNX | Arbitrary-Precision (from Frontend) | FPGAs (Custom Hardware) |
| FINN | QONNX (from Brevitas) | Arbitrary-Precision (from Frontend) | FPGAs (Custom Hardware) |
| Vitis AI | TF, PyTorch, ONNX | 8-bit Integer PTQ/QAT | FPGAs / ACAPs (DPU/AIE Overlay) |
| Apache TVM | All (via ONNX) | 8-bit, Mixed-Precision | CPUs, GPUs, Accelerators |
| Intel OpenVINO | TF, PyTorch, ONNX | Advanced PTQ (INT4/8, NF4, etc.) via NNCF | Intel CPUs, iGPUs, NPUs |
| **Stage 4: Hardware-in-the-Loop (HIL) Deployment & Runtime** |
| ExecuTorch | PyTorch | Low-Precision (from torchao) | Mobile CPUs, GPUs, DSPs |
| LiteRT | TF, PyTorch, JAX | Low-Precision (from TFMOT, etc.) | Mobile CPUs, GPUs, NPUs |
| PYNQ | Python | N/A (Controls accelerator) | AMD-Xilinx SoCs (Zynq, Kria) |

## Frontend Quantization-Aware Training (QAT) Frameworks

### Introduction to Quantization-Aware Training (QAT)

While 8-bit Post-Training Quantization (PTQ) is common for mobile deployment, FPGAs and embedded systems can use more aggressive quantization (e.g., 4-bit, 2-bit). QAT simulates quantization effects during training, which helps the model maintain accuracy at lower bit-widths.

### TensorFlow / Keras Ecosystem

#### QKeras

A Keras extension with layers for arbitrary-precision quantization. It is a QAT frontend for the `hls4ml` compiler. It is now considered stable but has been superseded by HGQ2, which is the recommended QAT tool for `hls4ml`.
*   GitHub: [https://github.com/google/qkeras](https://github.com/google/qkeras)
*   State: Stable / Superseded

#### HGQ (High Granularity Quantization)

A QAT algorithm that automatically finds the optimal bit-width for each parameter based on a hardware resource budget. It integrates with `hls4ml`. This tool has been superseded by HGQ2, which is built on Keras 3 and is the recommended QAT frontend for `hls4ml`.
*   GitHub (Original): [https://github.com/calad0i/HGQ](https://github.com/calad0i/HGQ)
*   GitHub (HGQ2): [https://github.com/calad0i/HGQ2](https://github.com/calad0i/HGQ2)
*   State:  Superseded / Stable

#### Google AI Edge (TFMOT & LiteRT)

Google's edge ecosystem has been significantly updated. The long-standing **TensorFlow Lite (TFLite)** runtime has been deprecated as of 2025.
*   **LiteRT:** The successor to TFLite, LiteRT is a new, independent, and multi-framework runtime designed to convert and run models from TensorFlow, PyTorch, and JAX. It forms the core of the **Google AI Edge** suite of tools. The goal is to provide a universal runtime with simplified hardware acceleration.
    *   GitHub: [https://github.com/google-ai-edge/LiteRT](https://github.com/google-ai-edge/LiteRT)
    *   State: Active & Strategic (Replacing TFLite)
*   **TensorFlow Model Optimization Toolkit (TFMOT):** This toolkit is still used for quantization but now targets the new LiteRT runtime.
    *   GitHub: [https://github.com/tensorflow/model-optimization](https://github.com/tensorflow/model-optimization)
    *   State: Stable / Mature

### PyTorch Ecosystem

#### Brevitas

A PyTorch-based QAT library from AMD-Xilinx, and the frontend for the FINN compiler. It models the reduced-precision hardware data-path during training. It is an active project and the QAT frontend for the FINN compiler.
*   GitHub: [https://github.com/Xilinx/brevitas](https://github.com/Xilinx/brevitas)
*   State: Active & Strategic

#### HAWQ (Hessian Aware Quantization)

A mixed-precision quantization library that uses second-order information (the Hessian) to determine bit-widths for different layers. This is a dormant academic project.
*   GitHub: [https://github.com/Zhen-Dong/HAWQ](https://github.com/Zhen-Dong/HAWQ)
*   State: Dormant

#### PyTorch Edge Stack (torchao & ExecuTorch)

The PyTorch ecosystem for edge deployment has been completely overhauled in 2025.
*   **torchao:** The legacy `torch.ao.quantization` library is deprecated. **`torchao`** is the new, centralized library for all model optimization. It provides a unified workflow for Quantization-Aware Training (QAT), Post-Training Quantization (PTQ), and sparsity. It supports modern low-precision formats (e.g., INT8, INT4, FP8) and is designed to prepare models for deployment.
    *   GitHub: [https://github.com/pytorch/ao](https://github.com/pytorch/ao)
    *   State: Active & Strategic (Replacing torch.ao.quantization)
*   **ExecuTorch:** Now at a stable 1.0 release, ExecuTorch is PyTorch's official solution for Ahead-of-Time (AOT) compilation and inference on edge devices (mobile, embedded). It runs models prepared with `torchao` without a conversion process, and is in production use at Meta.
    *   GitHub: [https://github.com/pytorch/executorch](https://github.com/pytorch/executorch)
    *   State: Stable & Strategic

### JAX Ecosystem

#### AQT (Accurate Quantized Training)

A Google library for JAX that aims for a bit-exact match between simulated training quantization and serving quantization. AQT is under active development and is focused on research.
*   GitHub: [https://github.com/google/aqt](https://github.com/google/aqt)
*   State: Active R&D

#### Qwix

A JAX quantization library from Google supporting QAT, PTQ, and numerics like int4 and fp8, with a focus on LLMs and accelerators. Qwix is an active R&D project.
*   GitHub: [https://github.com/google/qwix](https://github.com/google/qwix)
*   State: Active R&D

## The "Glue": Intermediate Representations

### ONNX (Open Neural Network Exchange)

Allows models to be moved between frameworks. Its standard quantization support is limited to 8-bit integers. The ONNX standard is actively developed, but some converters (e.g., for TensorFlow) can be inactive.
*   GitHub: [https://github.com/onnx/onnx](https://github.com/onnx/onnx)
*   State: Active & Strategic

### QONNX (Quantized ONNX)

An extension to ONNX created by the FINN and hls4ml communities. It adds custom operators to represent arbitrary-precision fixed-point quantization. As of 2025, the standard is being extended to also represent **arbitrary-precision minifloats** (e.g., FP8, FP4) via a new `FloatQuant` operator. This is critical for representing models with high dynamic range, which are common in scientific applications, and enables new research into hardware-software co-design for FPGAs.
*   GitHub: [https://github.com/fastmachinelearning/qonnx](https://github.com/fastmachinelearning/qonnx)
*   State: Active & Strategic

### ONNX Runtime

A cross-platform compiler and runtime from Microsoft that can perform its own 8-bit quantization on a floating-point ONNX model. It is a production-ready solution for deploying ONNX models to diverse hardware.
*   GitHub: [https://github.com/microsoft/onnxruntime](https://github.com/microsoft/onnxruntime)
*   State: Active & Strategic

## End-to-End Compilers for FPGA Synthesis

### Custom FPGA layout vs. runtime instructions

1.  **Custom layout:** Used by `hls4ml` and `FINN`, this approach generates new hardware logic for the neural network, offering low latency at the cost of slow hardware synthesis times.
2.  **Runtime overlay:** Used by `Vitis AI`, this approach uses a pre-built AI accelerator (DPU or AIE) on the FPGA. The NN is compiled into instructions for this runtime, offering a faster workflow.

| Philosophy | Key Tools | Core Technology | Workflow | Pros | Cons |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Custom layout | hls4ml, FINN | NN-to-HLS/RTL Compilation | (Slow) Generate HLS/RTL -> Synthesize | Low latency, arbitrary-precision | Slow build times, complex |
| Runtime overlay | Vitis AI | DPU / AIE Accelerators | (Fast) Compile NN -> Load instructions | Faster workflow, fast iteration | Higher latency, less specialized |

### "Custom layout" Philosophy: hls4ml

A Python package that translates models into HLS C++ code for low latency. It is an active project at the center of a research ecosystem for FPGA co-design.
*   GitHub: [https://github.com/fastmachinelearning/hls4ml](https://github.com/fastmachinelearning/hls4ml)
*   State: Active & Strategic

### "Custom layout" Philosophy: FINN

A dataflow compiler from AMD-Xilinx that builds accelerators from pre-optimized hardware blocks. It is the backend for the Brevitas/QONNX pipeline. As the official AMD-Xilinx dataflow compiler, FINN is a mature and stable tool. It is actively maintained as part of the Brevitas -> QONNX -> FINN pipeline.
*   GitHub: [https://github.com/Xilinx/finn](https://github.com/Xilinx/finn)
*   HLS Library: [https://github.com/Xilinx/finn-hlslib](https://github.com/Xilinx/finn-hlslib)
*   State: Stable / Mature

### "Runtime overlay" Philosophy: Vitis AI & AI Engines

AMD-Xilinx's development stack. It uses a "Deep-Learning Processor Unit (DPU)" or "AI Engines (AIE)" as a pre-existing accelerator. The user compiles their NN to run on this hardware. This is AMD-Xilinx's 'overlay' solution for FPGAs. The core compiler is slow-moving, but the surrounding tools and tutorials are actively maintained.
*   GitHub: [https://github.com/Xilinx/Vitis-AI](https://github.com/Xilinx/Vitis-AI)
*   State: Stable / Mature

## General-Purpose Compilers for Heterogeneous Edge Inference

These compilers target a wide range of hardware, not just FPGAs.

### Apache TVM

An open-source ML compilation framework that ingests models from any framework (via ONNX) and generates machine code for targets like ARM CPUs, GPUs, and specialized accelerators.
A significant 2025 development is the **`tvm-ffi`** initiative, which creates a stable C ABI for ML kernels. This allows high-performance libraries (e.g., for custom scientific operators) to be compiled once and loaded by any major framework (PyTorch, JAX) without complex bindings, solving a major ecosystem fragmentation problem.
*   GitHub: [https://github.com/apache/tvm](https://github.com/apache/tvm)
*   State: Active (Research & Infrastructure)

### Intel OpenVINO

Intel's open-source toolkit for optimizing and deploying AI inference on Intel hardware (CPUs, iGPUs, NPUs). The 2025 releases have aggressively pivoted the tool to support modern transformer-based models. Its optimization capabilities are centered on the **Neural Network Compression Framework (NNCF)**, which has been upgraded to support advanced weight compression (INT4/INT8, NF4) and other techniques relevant for large models.
*   GitHub: [https://github.com/openvinotoolkit/openvino](https://github.com/openvinotoolkit/openvino)
*   State: Active & Strategic

### Glow

A machine learning compiler from Meta/PyTorch that used Ahead-of-Time (AOT) compilation. As part of PyTorch's consolidation of its edge strategy, the Glow repository was officially archived on July 1, 2025. ExecuTorch is now the single, official solution for AOT compilation on edge devices.
*   GitHub: [https://github.com/pytorch/glow](https://github.com/pytorch/glow)
*   State: Archived / Deprecated

## Advanced Optimization & Specialized Tooling

### Hardware-Aware Pruning & Quantization: pQuant

A tool for "End-to-End Hardware-Aware Model Compression" that streamlines both pruning and quantization before feeding a model to a backend like `hls4ml`. This is an active research project.
*   GitHub: [https://github.com/nroope/PQuant](https://github.com/nroope/PQuant)
*   State: Active R&D

### Low-Level Arithmetic Optimization: da4ml

A library for "Distributed Arithmetic (DA)," a "multiplier-less" technique for matrix-vector multiplication using adders and Look-Up Tables (LUTs), trading DSP blocks for LUTs on an FPGA. A library for 'multiplier-less' arithmetic and a plugin for the `hls4ml` ecosystem.
*   GitHub: [https://github.com/calad0i/da4ml](https://github.com/calad0i/da4ml)
*   State: Active R&D

### FPGA-Specific Inference Optimization: NeuraLUT-Assemble

A framework for efficient neural network inference on FPGAs using lookup tables (LUTs). It combines mixed-precision techniques with assembling larger neurons from smaller units and incorporating skip-connections to improve accuracy and reduce area-delay product.
*   GitHub: [https://github.com/MartaAndronic/NeuraLUT](https://github.com/MartaAndronic/NeuraLUT)
*   State: Active R&D

### Pre-Synthesis Resource & Latency Estimation: rule4ml

Provides pre-synthesis estimates of FPGA resource utilization and latency using ML models. This project is currently dormant.
*   GitHub: [https://github.com/IMPETUS-UdeS/rule4ml](https://github.com/IMPETUS-UdeS/rule4ml)
*   State: Dormant

## Alternative Model Paradigms

### Boosted Decision Trees: conifer

A project related to `hls4ml` that translates trained Boosted Decision Trees (from scikit-learn, xgboost, etc.) into FPGA firmware for low-latency inference. It is actively maintained.
*   GitHub: [https://github.com/thesps/conifer](https://github.com/thesps/conifer)
*   State: Active

### Symbolic Regression

Instead of compressing a large NN, Symbolic Regression (SR) tools discover a symbolic formula that fits the data. This formula can be implemented in HLS, resulting in a small and efficient hardware model.
*   **PySR:** A high-performance SR library using genetic programming. This is an active project with a growing research community.
    *   GitHub: [https://github.com/MilesCranmer/PySR](https://github.com/MilesCranmer/PySR)
    *   State: Active R&D
*   **SymbolNet:** A neural symbolic regression tool that uses an NN to help discover the symbolic formula. This is a dormant, single-commit academic repository.
    *   GitHub: [https://github.com/hftsoi/SymbolNet](https://github.com/hftsoi/SymbolNet)
    *   State: Dormant

## Hardware Abstraction, Runtimes, and Deployment

### PYNQ (Python Productivity for Zynq)

An open-source project from AMD-Xilinx providing a bootable Linux environment for its SoC platforms. It allows users to control FPGA accelerators (like a Vitis AI DPU or an `hls4ml` design) from Python code running on the ARM CPU. This is the official AMD-Xilinx project for controlling SoCs with Python.
*   GitHub: [https://github.com/Xilinx/PYNQ](https://github.com/Xilinx/PYNQ)
*   State: Active & Strategic

### DPU-PYNQ

The specific PYNQ package that acted as the driver for the Vitis AI DPU. This project is archived and no longer maintained. Teams must migrate to modern Vitis AI runtimes.
*   GitHub: [https://github.com/Xilinx/DPU-PYNQ](https://github.com/Xilinx/DPU-PYNQ)
*   State: Archived / Deprecated

### Vivado / Vitis

The AMD-Xilinx hardware design suites. Compilers like `hls4ml` and `FINN` generate HLS code that is fed into Vitis/Vivado for synthesis into a hardware bitstream.

## Workflow

*   **Goal: Mobile phones & general edge devices (CPUs, GPUs, NPUs).**
    *   **Google Ecosystem (TF, JAX, PyTorch):** Quantize with TFMOT (or `ai-edge-torch` for PyTorch) and deploy with **LiteRT**.
    *   **PyTorch Ecosystem:** Optimize (quantize, sparsify) with **torchao** and deploy with **ExecuTorch**.
*   **Goal: Intel edge devices (CPUs, iGPUs, NPUs).**
    *   Use the **Intel OpenVINO** toolchain, which can ingest models from PyTorch, TF, or ONNX.
*   **Goal: Maximum portability across many different hardware backends.**
    *   Export to ONNX and use **ONNX Runtime**.
    *   For maximum performance and control, use **Apache TVM** to compile an ONNX model for a specific target.
*   **Goal: AMD/Xilinx FPGA with an easy, "software-like" workflow.**
    *   Use the "Overlay" philosophy with **Vitis AI** and the DPU/AIE accelerators.
*   **Goal: AMD/Xilinx FPGA with the absolute lowest latency and custom precision.**
    *   Use the "Custom Hardware" (dataflow) philosophy.
    *   **Keras:** Use QKeras/HGQ with `hls4ml`.
    *   **PyTorch:** Use **Brevitas** -> **QONNX** -> **FINN**.
*   **Goal: "My model has high dynamic range and needs floating-point precision on an FPGA."**
    *   Use the Brevitas/QONNX/FINN or hls4ml toolchains, leveraging the new **minifloat** support in QONNX.
*   **Goal: "My model is a Boosted Decision Tree."**
    *   Use `conifer`.
*   **Goal: "I need a simpler, more interpretable model than a neural network."**
    *   Explore Symbolic Regression with **PySR**.
