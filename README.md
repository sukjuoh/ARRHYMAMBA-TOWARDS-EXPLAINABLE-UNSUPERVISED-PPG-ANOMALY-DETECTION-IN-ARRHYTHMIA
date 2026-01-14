<h1 align="center">
  <sub><img src="files/arrhymamba_logo.png" width="70"></sub>
  ArrhyMamba: Towards Explainable Unsupervised PPG Anomaly Detection in Arrhythmia
</h1>
<p align="center">
  <strong>The Official PyTorch Implementation of ArrhyMamba</strong><br>
</p>

---

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

---

## 1. Model Architecture

![Overview of the proposed ArrhyMamba architecture](files/architecture.png)

ArrhyMamba is a hybrid SSM-Transformer framework designed for detecting arrhythmias in PPG signals through unsupervised time-series anomaly detection (TSAD).

### (a) Stage 1: VQ-Tokenizer
Stage 1 follows the standard **VQ-VAE** structure, consisting of a CNN encoder, vector quantization layer, and CNN decoder to learn discrete latent representations of PPG signals.

### (b) Stage 2: Masked Token Prediction
Following **MaskGIT**, this stage employs the **Mamba-Transformer** to predict masked tokens in the latent space.

### (c) Mamba-Transformer
The **Mamba-Transformer** is a hybrid architecture that integrates:
* **Bidirectional Mamba**: Composed of forward and backward structured state-space models (SSMs), enabling efficient modeling of long-range dependencies.
* **RoFormer**: A Transformer-based architecture that incorporates **Rotary Positional Embedding (RoPE)** to capture precise local patterns in PPG signals.

---

## 2. Key Results

**ArrhyMamba achieves state-of-the-art performance, outperforming existing deep learning and non-deep learning models in both Global and Personalized model settings.**

### Table 1: Performance Comparison on Global Model Setting
| Method | Accuracy | Recall | F1-score | AUC | Specificity |
| :--- | :---: | :---: | :---: | :---: | :---: |
| (Non-DL) RCF | 0.900 | 0.246 | 0.293 | 0.606 | 0.967 |
| (Non-DL) MERLIN | 0.878 | 0.029 | 0.056 | 0.506 | 0.996 |
| (Non-DL) Matrix Profile STUMPY | 0.889 | 0.253 | 0.332 | 0.614 | 0.974 |
| (Non-DL) Matrix Profile SCRIMP++ | 0.889 | 0.253 | 0.332 | 0.613 | 0.974 |
| (DL) MAD-GAN | 0.895 | 0.146 | 0.243 | 0.569 | 0.993 |
| (DL) LSTM-AD | 0.905 | 0.201 | 0.331 | 0.600 | **0.998** |
| (DL) DAGMM | 0.896 | 0.125 | 0.219 | 0.561 | 0.997 |
| (DL) USAD | 0.900 | 0.267 | 0.382 | 0.625 | **0.998** |
| (DL) TranAD | 0.907 | 0.249 | 0.387 | 0.622 | 0.995 |
| (DL) TimeVQVAE-AD | 0.901 | 0.726 | 0.598 | 0.827 | 0.927 |
| **(DL) ArrhyMamba (Ours)** | **0.915** | **0.937** | **0.732** | **0.924** | 0.911 |

### Table 2: Performance Comparison on Personalized Model Setting
| Method | Accuracy | Recall | F1-score | AUC | Specificity |
| :--- | :---: | :---: | :---: | :---: | :---: |
| (DL) MAD-GAN | 0.906 | 0.373 | 0.457 | 0.637 | 0.988 |
| (DL) LSTM-AD | 0.916 | 0.432 | 0.548 | 0.698 | **0.994** |
| (DL) DAGMM | 0.910 | 0.385 | 0.484 | 0.678 | 0.989 |
| (DL) USAD | 0.926 | 0.460 | 0.597 | 0.725 | 0.989 |
| (DL) TranAD | 0.927 | 0.489 | 0.617 | 0.733 | 0.990 |
| (DL) TimeVQVAE-AD | 0.913 | 0.853 | 0.766 | 0.886 | 0.918 |
| **(DL) ArrhyMamba (Ours)** | **0.954** | **0.985** | **0.841** | **0.967** | 0.949 |

---


## 3. Requirements & Installation

To ensure the CUDA extensions for Mamba (specifically `selective_scan_cuda`) are correctly compiled, the following environment is required:

* **Python Version**: 3.10 or higher.
* **Hardware**: NVIDIA GPU with a compatible CUDA Toolkit for PyTorch 2.4.0.
* **Dependencies**: Install core libraries using the provided `requirements.txt`.

```bash
# Install all dependencies
pip install -r requirements.txt
```

## 4. Data Availability
* **Public Dataset**: Raw PPG signals are sourced from the **VitalDB** open repository ([https://vitaldb.net/](https://vitaldb.net/)).
* **Restricted Data**: Please note that the **refined datasets and expert-labeled annotations** generated for this study are **strictly not publicly available** due to ethical restrictions, patient privacy concerns, and institutional policies.
* **Access**: Researchers may contact the corresponding author for reasonable access requests regarding metadata, subject to institutional approval.

## 5. Repository Structure

* **`models/`**: Implementation of the hybrid Mamba-Transformer architecture including RoPE (Rotary Positional Embedding).
* **`data/`**: Data loading utilities and preprocessing scripts utilizing `heartpy` for signal filtering.
* **`utils/`**: Custom CUDA kernels for selective scan and training helper functions.
* **`configs/`**: YAML files containing hyperparameter settings to ensure experimental reproducibility.

## 6. Inference & Evaluation

To facilitate immediate testing, the pre-trained checkpoint and sample test data are included directly in this repository.

### 6.1. Model Checkpoint
The pre-trained weight file, **`ArrhyMamba.ckpt`**, is provided in the file list above. 

> **Note on Training Procedure:** Our model follows a **two-stage training process**. The weights from Stage 1 are used as the foundation for Stage 2 training. The final `ArrhyMamba.ckpt` includes the fully integrated weights from both stages, representing the complete optimized model.



### 6.2. Sample Dataset for Anomaly Detection
Example test data is located in `preprocess/PPG_dataset/`.
* **Purpose**: These samples demonstrate how ArrhyMamba performs **arrhythmia detection** using an **unsupervised anomaly detection** framework.
* **Functionality**: We provide these examples to show the model's ability to identify irregular PPG patterns (arrhythmias) by detecting them as temporal anomalies within the signal flow.

### 6.3. Running Evaluation
You can verify the model's performance by running the evaluation script with the provided sample dataset:

```bash
# Run evaluation using the included checkpoint and sample dataset (index 4)
python eval.py --dataset_ind 4
```
![Overview of the proposed ArrhyMamba architecture](evaluation/results/4_techtycardia.png)
