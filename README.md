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

### 04 - Intra Feature Selection (LLD × stats)

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

### 06-08_Deep_Learning

- Deezer: Creating the Deezer dataset with metadata extracted from the Deezer API and GENIUS for lyrics (unused at this stage)
- Deezer_Analysis: EDA on Deezer
- 06: Dataset preparation for Deep learning benchmarks (mel spectrograms and MERT embeddings)
- 07: Training computer vision backbones on DEAM with a regressor head (+tested pretraining on Deezer)
- 08: Train a regressor on MERT embeddings from DEAM (+with pretrain on Deezer) + zeroshot evaluation of Music2Emo (https://huggingface.co/amaai-lab/music2emo)

---
