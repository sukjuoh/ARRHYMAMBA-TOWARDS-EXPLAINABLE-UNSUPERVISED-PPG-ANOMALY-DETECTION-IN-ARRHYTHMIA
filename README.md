<h1 align="center">
  <img src="files/arrhymamba_logo.png" width="70" style="vertical-align: middle; margin-right: 10px;">
  ArrhyMamba: Towards Explainable Unsupervised PPG Anomaly Detection in Arrhythmia
</h1>

<p align="center">
  <strong>Official Implementation of the Hybrid SSM-Transformer Framework</strong>
</p>

## 1. Overview

* **Architecture**: ArrhyMamba leverages the long-sequence modeling of **Mamba** and the robust attention mechanism of **Transformers** with **RoPE** (Rotary Positional Embedding).
* **Optimization**: The model is specifically optimized for high-frequency biomedical signals, capturing both global dependencies and local features.

## 2. Requirements & Installation

To ensure the CUDA extensions for Mamba (specifically `selective_scan_cuda`) are correctly compiled, the following environment is required:

* **Python Version**: 3.10 or higher.
* **Hardware**: NVIDIA GPU with a compatible CUDA Toolkit for PyTorch 2.4.0.
* **Dependencies**: Install core libraries using the provided `requirements.txt`.

```bash
# Install all dependencies
pip install -r requirements.txt
```

## 3. Data Availability
* **Public Dataset**: Raw PPG signals are sourced from the **VitalDB** open repository ([https://vitaldb.net/](https://vitaldb.net/)).
* **Restricted Data**: Please note that the **refined datasets and expert-labeled annotations** generated for this study are **strictly not publicly available** due to ethical restrictions, patient privacy concerns, and institutional policies.
* **Access**: Researchers may contact the corresponding author for reasonable access requests regarding metadata, subject to institutional approval.

## 4. Repository Structure

* **`models/`**: Implementation of the hybrid Mamba-Transformer architecture including RoPE (Rotary Positional Embedding).
* **`data/`**: Data loading utilities and preprocessing scripts utilizing `heartpy` for signal filtering.
* **`utils/`**: Custom CUDA kernels for selective scan and training helper functions.
* **`configs/`**: YAML files containing hyperparameter settings to ensure experimental reproducibility.

## 5. Inference & Evaluation

To facilitate immediate testing, the pre-trained checkpoint and sample test data are included directly in this repository.

### 5.1. Model Checkpoint
The pre-trained weight file, **`ArrhyMamba.ckpt`**, is provided in the file list above. 

> **Note on Training Procedure:** Our model follows a **two-stage training process**. The weights from Stage 1 are used as the foundation for Stage 2 training. The final `ArrhyMamba.ckpt` includes the fully integrated weights from both stages, representing the complete optimized model.



### 5.2. Sample Dataset for Anomaly Detection
Example test data is located in `preprocess/PPG_dataset/`.
* **Purpose**: These samples demonstrate how ArrhyMamba performs **arrhythmia detection** using an **unsupervised anomaly detection** framework.
* **Functionality**: We provide these examples to show the model's ability to identify irregular PPG patterns (arrhythmias) by detecting them as temporal anomalies within the signal flow.

### 5.3. Running Evaluation
You can verify the model's performance by running the evaluation script with the provided sample dataset:

```bash
# Run evaluation using the included checkpoint and sample dataset (index 4)
python eval.py --dataset_ind 4
