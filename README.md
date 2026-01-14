<h1 align="center">
  <sub><img src="files/arrhymamba_logo.png" width="70"></sub>
  ArrhyMamba: Towards Explainable Unsupervised PPG Anomaly Detection in Arrhythmia

<p align="center">
  <strong>The Official PyTorch Implementation of ArrhyMamba</strong><br>


## 📄 Abstract
Photoplethysmography (PPG) signals exhibit significant inter- and intra-individual variability, making
arrhythmia detection particularly challenging. Due to this variability and the severe class imbalance
inherent in arrhythmia datasets, existing supervised learning models often struggle to generalize,
limiting their ability to detect diverse arrhythmia across different individuals. In this paper, we propose
ArrhyMamba, a PPG-based model leveraging time series anomaly detection (TSAD). By utilizing
exclusively on normal PPG data, it enables a single model to detect various types of arrhythmias across
diverse patients. Additionally, ArrhyMamba enhances explainability by generating counterfactual
waveforms for detected arrhythmias. Unlike traditional approaches, ArrhyMamba adopts a hybrid
architecture that combines the strengths of Mamba and Transformer models, enabling efficient and
effective sequence modeling. On the VitalDB PPG dataset, ArrhyMamba achieved an accuracy
of 0.915, recall of 0.937, F1-score of 0.732, AUC of 0.924, and specificity of 0.911. Compared
to TimeVQVAE-AD, the state-of-the-art (SOTA) TSAD model, ArrhyMamba showed substantial
performance gains — improving accuracy by 1.5%, recall by 29%, F1-score by 22%, and AUC by
12%, with only 30% of the parameters.


## 1. Model Architecture

![Overview of the proposed ArrhyMamba architecture](files/architecture.png)
*Figure 1: Overview of the proposed ArrhyMamba architecture.*

ArrhyMamba is a hybrid SSM-Transformer framework designed for detecting arrhythmias in PPG signals through unsupervised time-series anomaly detection (TSAD).

### (a) Stage 1: VQ-Tokenizer
Stage 1 follows the standard **VQ-VAE** structure, consisting of a CNN encoder, vector quantization layer, and CNN decoder to learn discrete latent representations of PPG signals.

### (b) Stage 2: Masked Token Prediction
Following **MaskGIT**, this stage employs the **Mamba-Transformer** to predict masked tokens in the latent space.

### (c) Mamba-Transformer (The Hybrid Core)
The **Mamba-Transformer** is a hybrid architecture that integrates:
* **Bidirectional Mamba**: Composed of forward and backward structured state-space models (SSMs), enabling efficient modeling of long-range dependencies.
* **RoFormer**: A Transformer-based architecture that incorporates **Rotary Positional Embedding (RoPE)** to capture precise local patterns in PPG signals.

---

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
