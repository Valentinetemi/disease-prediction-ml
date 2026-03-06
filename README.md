# 🩺 Disease Prediction from Symptoms

A machine learning system that predicts diseases based on patient-reported symptoms, trained on real medical data using Decision Tree and Random Forest classifiers.

---

## 📌 Overview

Given a set of symptoms, this model predicts the most likely disease out of 41 possible conditions. Built with Python and scikit-learn, and fully documented as a public build log on Medium.

---

## 📂 Dataset

- **Source:** [Disease Symptom Description Dataset — Kaggle](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset)
- **4,920 rows** of patient symptom records
- **41 disease classes**
- Raw symptoms stored as text across 17 columns → preprocessed into **132 binary features**

---

## ⚙️ Preprocessing

The raw dataset stores symptoms as text strings across 17 columns (Symptom_1 through Symptom_17). Preprocessing transforms this into a binary feature matrix where each unique symptom becomes its own column filled with 1 (present) or 0 (absent).

**Final shape after preprocessing:** `(4920, 132)`

---

## 🤖 Models

| Model | Type |
|-------|------|
| Decision Tree | Single classifier — interpretable, good baseline |
| Random Forest | Ensemble method — more accurate, reduces overfitting |

---

## 🛠️ Tech Stack

- Python
- pandas, numpy
- scikit-learn
- Jupyter Notebook / Google Colab

---

## 📖 Build Log

This project is being documented publicly on Medium:

- **Part 1:** [Dataset breakdown & model selection](https://medium.com/@temiloluwaval/building-a-disease-prediction-ai-from-scratch-phase-1-fef7894fdd00) ← update with link
- **Part 2:** Preprocessing, training & evaluation ← coming soon

---

## 🚀 How to Run

1. Clone the repo
```bash
git clone https://github.com/Valentinetemi/disease-prediction-ml.git
```

2. Open the notebook in Google Colab or Jupyter

3. Download the dataset from Kaggle and place it in the root folder

4. Run cells in order

---

*Built and documented by [Temi](https://github.com/Valentinetemi)*
