# Master Prompt / Project Context (ME44312)

Use this file as the single source of truth for any coding agent working on this repo (especially in Jupyter notebooks).

## 1) Project Context & Motivation

- **Course:** Machine Learning for Transport and Multi Machine Systems (ME44312)
- **Authors:** Lucas Braxhoofden, Dirk Geertman, Stos van Lynden, Julius van Commenee, Ties Roorda
- **Core problem:** Evaluate model generalizability under **regional domain shift**
- **Goal:** Quantify how models trained in one environment perform in **unseen regions**, and whether **transfer learning** mitigates performance drops

## 2) Data Structure & Characteristics

- **Dataset file:** `equipment_failure_data_1.csv`
- **Target:** `EQUIPMENT_FAILURE` (binary classification)
- **Sequential nature:** This is time-series / sequential data
  - Always sort by grouping key (`ID` or `WELL_GROUP`) and then by `DATE` **before** splitting, feature engineering, or analysis
- **Feature columns**
  - **Categorical:** `REGION_CLUSTER`, `MAINTENANCE_VENDOR`, `MANUFACTURER`, `WELL_GROUP`
  - **Numerical:** `S15`, `S17`, `S13`, `S5`, `S16`, `S19`, `S18`, `S8`, `AGE_OF_EQUIPMENT`
  - **Meta / grouping:** `ID`, `DATE`

## 3) Preprocessing & Splitting Rules (Non-Negotiable)

- **Validation strategy:** Leave-One-Group-Out (LOGO) cross-validation based strictly on `REGION_CLUSTER`
- **No random global split:** Random train/test split across the full dataset is forbidden (prevents mixing regional patterns)
- **Shuffling rule:** You may shuffle **only after** splitting, and **only** the training portion (before model fitting)
- **Feature engineering (encouraged):** Derived sequential features (e.g., lag features, rolling std/mean per well/ID, short-term deltas)
  - Build them in a leakage-safe way: for any row, only use information available at or before its timestamp within its sequence

## 4) Modeling Progression

1. **Baseline:** Decision Tree (interpretable reference)
2. **Primary:** Random Forest (nonlinear baseline + feature stability)
3. **Advanced:** Neural Network (or comparable advanced architecture) for best predictive performance
   - Deep learning framework: PyTorch or Keras

## 5) Required Experiments (Must Be Reproducible)

- **Local benchmark:** Train on Region A, test on Region A (upper bound without domain shift)
- **Foreign test:** Train on Region A (or A+B+C+D), test on unseen Region E (measure domain-shift drop)
- **Transfer learning test:** Pretrain on multiple regions, fine-tune on limited data from target region
  - Compare against training only on that limited target data
- **Feature stability analysis:** Compare feature importance (or analogous NN attributions) across regions
- **Metrics (focus):** Track the **drop in Recall and F1** from Local → Foreign; also report precision, ROC-AUC/PR-AUC where useful

## 6) Coding Environment & Constraints

- **Primary working surface:** VS Code + Python + Jupyter notebooks (`CODE Machine Failure ME NN.ipynb`)
- **Style:** Keep code simple and concise; run in clear, sequential notebook cells
- **Preprocessing transparency:** Prefer `pandas`/`numpy` for preprocessing where feasible (use `scikit-learn` mainly for models / metrics)
- **No leakage:** Any transformation that learns parameters must be fit on training data only within each LOGO fold

## 7) Deliverable Expectations (Per Experiment / Fold)

- A clear description of the split (which `REGION_CLUSTER` held out)
- A table of metrics (at minimum Recall and F1)
- A short, explicit comparison of Local vs Foreign results with absolute dates/regions where relevant

