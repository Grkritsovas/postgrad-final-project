# Project XAI with traditional ML for music emotion recognition (valence–arousal - **VA**).

**Data:** DEAM **Features:** 260 openSMILE LLDs over 45s (or full song) at 2 Hz (IS13 COMPARE LLD-FUNC.CONF).

**ML prep:** roll time series into statistical descriptors — default **8** and extended **15**.
--- 
## Notebook map ### 01 - Build datasets Generates 8 feature configs:
1) **2080** = 260 LLD × **8** stats (min, max, q25, q75, mean, std, kurtosis, skew)
2) **3900** = 260 LLD × **15** stats (adds median, range, trend, variation, …)
3) **1257** = (1) with **perceptual-group decorrelation**
4) **1762** = (2) with **perceptual-group decorrelation**
5) **Per-group PCA** on (1) to **95% var** → **347** total PCs (shared among groups)
6) **Per-group PCA** on (2) to **95% var** → **516** total PCs (shared among groups)
7) **Global PCA** on (1) to **95% var** → ~**242 PCs**
8) **Global PCA** on (2) to **95% var** → ~**317 PCs**
 > PCA is scale-sensitive → for (5–8) we use the **custom split** only (fit scaler+PCA on train, transform val/test) to avoid duplicating datasets.
---
### 02 - EDA Label quality; feature/metadata correlations; PCA visuals.
---
### 03 - Baselines Pre-selection baselines and sanity checks.
---
### 04[1–4] - Intra feature selection (LLD × stats) Same pipeline across the four non-PCA datasets: 
- **Step A (CV):** choose best **k** stats per LLD via nested CV. Inside each fold: - fit RF per column, rank by their **SHAPley values** (joint VA) (sort them descending).
- **Step B (dev re-rank):** on dev set, re-rank the top-k and keep the final selection (leak-free imputation with train medians; joint VA objective).
- **Output:** final per-base **X** and SHAP explanations.
---
### 04[5-6] - Per-group PCA route
- Parse {group}_PC* columns, CV to choose **m PCs per group** (joint VA).
- Build fold designs with **train medians only**; compare RF/GBR/Ridge/ENet/SVR on dev CV.
- Group-level SHAP by summing PCs’ |SHAP| within each group.
---
### 04[7-8] — Global PCA route - Sweep **n PCs** globally with leak-free CV to pick **best_n**.
- Compare model families on the chosen n.
- Permutation test for significance; 2D PC scatter colored by VA/AR; PC↔VA/AR correlations.
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
### 07 – Transfer Learning: Training 
- Trains computer vision backbone models with mel-spectrograms created in Notebook 06 (styles: AST, PANNs, Musicnn, CLAP, VGGish) on DEAM with a regressor head.
- Also tested with adding a pretraining step from the Deezer dataset (DIY pretraining) and gradual unfreezing of layers from the pre-trained models before fine-tuning on DEAM.
- Serves as a DL baseline for the task
---
### 08 – MERT + Music2Emo
- Uses state-of-the-art music embeddings (e.g., MERT) to train a regressor on DEAM.
- Also tested with adding a pretraining step from Deezer to see if the regressor can become better aligned with the domain task of V-A predictions.
- Zeroshot tested an efficient multimodal pre-trained model, but only provided it with the MERT embeddings (5th and 6th layer), while Music2Emo usually prefers to be also given chords and keys, instead those were padded with 0s.
- Provides a benchmark against classical ML + explainable pipelines. ---
