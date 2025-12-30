네, 요청하신 대로 'How to Run' 섹션을 제외하고 나머지 핵심 내용을 담은 README.md 파일 내용을 정리해 드립니다. 아래 내용을 복사하여 파일로 만드시면 됩니다.

ArrhyMamba: Towards Explainable Unsupervised PPG Anomaly Detection in Arrhythmia
This repository contains the official implementation of ArrhyMamba, a hybrid SSM-Transformer framework designed for detecting arrhythmias in PPG signals through unsupervised time-series anomaly detection (TSAD).

1. Overview
ArrhyMamba leverages the long-sequence modeling capabilities of Mamba and the robust attention mechanism of Transformers (integrated with RoPE) to identify irregular cardiac patterns in PPG data. The model is specifically optimized for high-frequency biomedical signals where capturing both global dependencies and local features is critical.

2. Installation & Requirements
To ensure the CUDA extensions for Mamba (specifically selective_scan_cuda) are correctly compiled, the following environment is required.

Python Version: 3.9 or higher.

Hardware: NVIDIA GPU with a compatible CUDA Toolkit for PyTorch 2.3.0.

Core Dependencies: Detailed versions for torch, mamba-ssm, heartpy, and x-transformers are listed in requirements.txt.

Bash

# Install dependencies from the provided requirements file
pip install -r requirements.txt
Note: The mamba-ssm and causal-conv1d packages require a C++ compiler and CUDA environment to build specialized kernels during installation.

3. Data Availability
Public Dataset: Raw PPG signals are sourced from the VitalDB open repository (https://vitaldb.net/).

Restricted Data: The specialized arrhythmia annotations and preprocessed datasets generated for this study are not publicly available due to ethical restrictions and patient privacy concerns.

Access Requests: Researchers may contact the corresponding author for access to specific metadata, which will be granted upon reasonable request and institutional approval.

4. Repository Structure
models/: Implementation of the hybrid Mamba-Transformer architecture including RoPE (Rotary Positional Embedding).

data/: Data loading utilities and preprocessing scripts utilizing heartpy for signal filtering.

utils/: Custom CUDA kernels for selective scan and training helper functions.

configs/: YAML files containing hyperparameter settings to ensure experimental reproducibility.
