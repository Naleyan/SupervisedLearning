# Music Classification & Regression (Audio ML Project)

End-to-end machine learning project on a large-scale music dataset (≈100k tracks), combining **metadata**, **listening statistics**, and **audio descriptors** to perform **genre classification** and **track duration regression**.

---

## Project Overview

This project merges four raw data sources into a single cleaned and feature-engineered dataset:

* **Tracks metadata**
* **Genres taxonomy**
* **Echonest audio features**
* **Spectral audio features**

On top of this unified dataset, several supervised learning models are trained and evaluated on three distinct tasks:

* **Task 1 – Fine-grained genre classification** (`genre_top`, ~16 classes)
* **Task 2 – Coarse genre classification** (`genre_coarse`, 3 classes)
* **Task 3 – Track duration regression** (`duration` / `log_duration`)

All steps of the pipeline — data preparation, exploratory data analysis (EDA), feature engineering, model training, and evaluation — are implemented in **Jupyter notebooks**.

---

## Main Results

### Task 1 – Fine-grained Genre Classification (`genre_top`)

* **Number of classes:** ~16 genres
* **Best-performing models:** Tree-based ensembles

  * Random Forest
  * Gradient Boosting (XGBoost / LightGBM, depending on the run)
* **Typical performance:**

  * Accuracy: **~0.70–0.75**
  * Macro-F1: significantly lower than accuracy due to class imbalance and rare genres

This task highlights the difficulty of reliable classification when dealing with many underrepresented genres.

---

### Task 2 – Coarse Genre Classification (`genre_coarse`)

* **Number of classes:** 3 genre families
* **Approach:**

  * Custom taxonomy grouping original genres into three balanced families (e.g. *mainstream*, *rock/roots*, *specialised*)
  * Same models and evaluation protocol as Task 1
* **Typical performance:**

  * Accuracy: **≥ 0.80**
  * Macro-F1: close to accuracy

Compared to the fine-grained setting, this task shows a clear gain in **stability, robustness, and interpretability**.

---

### Task 3 – Track Duration Regression

* **Target variables:**

  * `duration`
  * `log_duration`
* **Models evaluated:**

  * Linear regression with regularisation
  * Tree-based regressors
  * Gradient boosting regressors
* **Key findings:**

  * Combining listening statistics with audio features consistently reduces prediction error compared to simple baselines
  * Tree-based and gradient boosting models achieve the best **RMSE** and **R²** scores on the test set

---

## Repository Structure

```text
notebooks/
├── phase1.ipynb
│   Load and merge raw data (tracks, genres, echonest, spectral),
│   clean columns, handle missing values, engineer features
│   (log-duration, ratios, encoded genres), and export CSV files
│   to data/intermediate/.
│
├── genre_top.ipynb
│   Task 1: Fine-grained genre classification.
│   Train multiple classifiers, compare accuracy, macro-F1,
│   and analyse confusion matrices for `genre_top`.
│
├── genre_coarse.ipynb
│   Task 2: Coarse genre classification.
│   Define the 3-group taxonomy (`genre_coarse`), retrain the same
│   models, and analyse performance gains compared to Task 1.
│
├── track_duration.ipynb
│   Task 3: Track duration regression.
│   Compare different feature sets (metadata vs. audio) using
│   RMSE, MAE, and R².
```

---

## How to Run

1. **Create a Python 3 environment** and install the required dependencies:

   * `pandas`
   * `numpy`
   * `scikit-learn`
   * `matplotlib`
   * `seaborn`
   * `xgboost` and/or `lightgbm` (optional, for gradient boosting experiments)

2. **Place the raw data files** in the following directory:

   ```text
   data/raw/
   ├── tracks
   ├── echonest_features
   ├── spectral_features
   └── genres
   ```

3. **Run the data preparation notebook**:

   ```text
   notebooks/phase1.ipynb
   ```

   This generates the following intermediate datasets in `data/intermediate/`:

   * `df_phase1.csv`
   * `df_classif_phase1.csv`
   * `df_audio_phase1.csv`

4. **Run the task-specific notebooks** to reproduce the results:

   * `notebooks/genre_top.ipynb`
   * `notebooks/genre_coarse.ipynb`
   * `notebooks/track_duration.ipynb`

---

## Notes

* Results may vary slightly depending on random seeds, feature selection, and model hyperparameters.
* The project is designed as an end-to-end applied ML workflow, with a strong focus on **comparative modeling**, **evaluation**, and **practical trade-offs between task complexity and performance**.
