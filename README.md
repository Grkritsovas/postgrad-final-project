# Project
XAI with traditional ML for music emotion recognition **VA** (valence-arousal).

**Main Data:** DEAM

**Features:** 260 openSMILE LLDs over 45s (or full song) at 2 Hz (IS13_ComParE_lld-func.conf).

**ML prep:** roll time series into statistical descriptors - default **8** and extended **15**.

**Secondary Data:** Deezer - available splits can be downloaded from https://github.com/deezer/deezer_mood_detection_dataset/tree/master

--- 
## Notebook map

### 01 - Build datasets
Generates 8 feature configs:

1) **2080** = 260 LLD × **8** stats (min, max, q25, q75, mean, std, kurtosis, skew)
2) **3900** = 260 LLD × **15** stats (adds median, range, trend, variation, …)
3) **1257** = (1) with **perceptual-group decorrelation**
4) **1762** = (2) with **perceptual-group decorrelation**
5) **Per-group PCA** on (1) to **95% var** → **347** total PCs (shared among groups)
6) **Per-group PCA** on (2) to **95% var** → **516** total PCs (shared among groups)
7) **Global PCA** on (1) to **95% var** → ~**242 PCs**
8) **Global PCA** on (2) to **95% var** → ~**317 PCs**

> PCA is scale-sensitive → for (5–8) we use the **custom split** only (fit scaler+PCA on train, transform val/test) to avoid duplicating datasets.
> Also generate key/mode metadata (mode will be used for minorness, weighted by key_confidence - mode prediction with madmom)

---

### 02 - EDA
Label quality; feature/metadata correlations; PCA visuals.

---

### 03 - Baselines
Pre-selection baselines and sanity checks.

---

### 04 — Intra Feature Selection (LLD × stats)

For the four non-PCA bases: 1_2080, 2_3900, 3_decorrelated_2080, 4_decorrelated_3900

- Step A – CV ranking: nested CV; within each fold fit RF, compute SHAP (joint V-A), rank stats per LLD, sweep k.

- Step B – Dev re-rank: re-rank top-k on dev (train-median imputation; joint V-A objective).

- Output: final per-base X and fold-aware SHAP explanations.

PCA routes:

- 5_per-group_PCA_2080, 6_per-group_PCA_3900

  - Parse {group}_PC*, CV-choose m PCs/group (joint V-A).

  - Compare RF/GBR/Ridge/ENet/SVR on dev CV.

  - Group-level SHAP: sum |SHAP| of PCs within each group.

- 7_global_PCA_2080, 8_global_PCA_3900

  - Sweep n global PCs with leak-free CV to pick best_n.

  - Compare model families; permutation tests; PC↔V/A correlations; 2D PC scatter colored by V/A.

---
### 05_Final_Comparisons

- Brings together results from all pipelines (raw descriptors, decorrelated, PCA).
- Compares performance across datasets and methods.
- Identifies best-performing and most interpretable setups.

---
### 06 – Transfer Learning: Dataset Preparation

- Prepares external embeddings from large pre-trained music models and preprocessed mel spectrograms that match shapes of computer vision models.
- Aligns them with DEAM’s VA labels.
- Produces train/val/test splits for downstream transfer learning experiments.

---
### 07 - Transfer Learning: Training 

- Trains computer vision backbone models with mel-spectrograms created in Notebook 06 (styles: AST, PANNs, Musicnn, CLAP, VGGish) on DEAM with a regressor head.
- Also tested with adding a pretraining step from the Deezer dataset (DIY pretraining) and gradual unfreezing of layers from the pre-trained models before fine-tuning on DEAM.
- Serves as a DL baseline for the task

---
### 08 – MERT + Music2Emo

- Uses state-of-the-art music embeddings (e.g., MERT) to train a regressor on DEAM.
- Also tested with adding a pretraining step from Deezer to see if the regressor can become better aligned with the domain task of V-A predictions.
- Zeroshot tested an efficient multimodal pre-trained model, but only provided it with the MERT embeddings (5th and 6th layer), while Music2Emo usually prefers to be also given chords and keys, instead those were padded with 0s.
- Provides a benchmark against classical ML + explainable pipelines.
---
