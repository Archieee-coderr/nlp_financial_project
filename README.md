# Financial Sentiment Analysis — Financial PhraseBank (75% Agreement)

## Overview

This project performs sentiment classification on the Financial PhraseBank dataset (75% annotator agreement subset), classifying financial sentences into three categories: **Negative (0)**, **Neutral (1)**, and **Positive (2)**.

Two embedding models are compared across five classifiers.

---

## Pipeline

```
Raw Data → Class Balancing → Embedding Extraction → Classification → Evaluation
```

1. **Data Loading** — `Sentences_75Agree.txt`, parsed and label-mapped
2. **Class Balancing** — Training set only: random oversampling of minority classes (Negative, Positive) to match Neutral count; all classifiers use `class_weight='balanced'`
3. **Embedding** — Sentences encoded into dense vectors via pre-trained transformer models
4. **Classification** — Five classifiers trained and evaluated on a fixed 80/20 train/test split
5. **Evaluation** — Accuracy, precision, recall, F1-score per class, confusion matrices

> **Note:** Oversampling is applied **after** the train/test split to prevent data leakage. The test set always reflects the original class distribution.

---

## Models

| | all-MiniLM-L6-v2 | FinBERT (ProsusAI/finbert) |
|---|---|---|
| Type | General-purpose | Financial domain |
| Embedding dim | 384 | 768 |
| Representation | Sentence embedding | [CLS] token |

---

## Dataset

**Original label distribution (3,453 sentences):**

```
Neutral  (1): 2146  (62.1%)
Positive (2):  887  (25.7%)
Negative (0):  420  (12.2%)
```

**After oversampling (training set only):**

```
Neutral  (1): 1717
Positive (2): 1717
Negative (0): 1717
Total train : 5151
Total test  :  691  (original distribution)
```

---

## Results

### MiniLM — Data Analysis

**Label Distribution & Sentence Length**

![Label Distribution](images/75Agree/image-10.png)
![Sentence Length Distribution](images/75Agree/image-11.png)

**Embedding Visualisation**

| PCA | t-SNE |
|-----|-------|
| ![PCA](images/75Agree/image-12.png) | ![t-SNE](images/75Agree/image-13.png) |

**Class centroid distances (MiniLM):**
```
Negative vs Neutral:  0.4237
Negative vs Positive: 0.2901
Neutral  vs Positive: 0.2601
```

---

### MiniLM — Classifier Results

| Classifier | Accuracy |
|---|---|
| Logistic Regression | 83.07% |
| Linear SVM | 83.65% |
| **RBF SVM** | **87.70%** |
| Random Forest | 81.33% |
| Naive Bayes | 73.95% |

**Confusion Matrices:**

| Logistic Regression | Linear SVM |
|---|---|
| ![CM LR](images/75Agree/image-14.png) | ![CM Linear SVM](images/75Agree/image-15.png) |

| RBF SVM | Random Forest |
|---|---|
| ![CM RBF SVM](images/75Agree/image-16.png) | ![CM RF](images/75Agree/image-17.png) |

| Naive Bayes |
|---|
| ![CM NB](images/75Agree/image-18.png) |

**Full classification report (MiniLM):**
```
                     precision  recall  f1-score  support

Logistic Regression
  Negative (0)          0.74    0.86      0.80       84
  Neutral  (1)          0.91    0.85      0.88      429
  Positive (2)          0.71    0.78      0.74      178
  macro avg             0.79    0.83      0.81      691

Linear SVM
  Negative (0)          0.72    0.87      0.79       84
  Neutral  (1)          0.92    0.85      0.88      429
  Positive (2)          0.73    0.79      0.76      178
  macro avg             0.79    0.84      0.81      691

RBF SVM
  Negative (0)          0.89    0.75      0.81       84
  Neutral  (1)          0.88    0.96      0.92      429
  Positive (2)          0.86    0.73      0.79      178
  macro avg             0.88    0.81      0.84      691

Random Forest
  Negative (0)          0.83    0.60      0.69       84
  Neutral  (1)          0.80    0.97      0.88      429
  Positive (2)          0.85    0.53      0.65      178
  macro avg             0.83    0.70      0.74      691

Naive Bayes
  Negative (0)          0.59    0.76      0.66       84
  Neutral  (1)          0.84    0.83      0.83      429
  Positive (2)          0.58    0.51      0.54      178
  macro avg             0.67    0.70      0.68      691
```

---

### FinBERT — Data Analysis

**Embedding Visualisation**

| PCA | t-SNE |
|-----|-------|
| ![PCA](images/75Agree/image-19.png) | ![t-SNE](images/75Agree/image-20.png) |

**Class centroid distances (FinBERT):**
```
Negative vs Neutral:  23.6850
Negative vs Positive: 23.4216
Neutral  vs Positive: 18.1887
```

---

### FinBERT — Classifier Results

| Classifier | Accuracy |
|---|---|
| Logistic Regression | 96.53% |
| Linear SVM | 95.95% |
| **RBF SVM** | **96.67%** |
| **Random Forest** | **96.67%** |
| Naive Bayes | 96.38% |

**Confusion Matrices:**

| Logistic Regression | Linear SVM |
|---|---|
| ![CM LR](images/75Agree/image-21.png) | ![CM Linear SVM](images/75Agree/image-22.png) |

| RBF SVM | Random Forest |
|---|---|
| ![CM RBF SVM](images/75Agree/image-23.png) | ![CM RF](images/75Agree/image-24.png) |

| Naive Bayes |
|---|
| ![CM NB](images/75Agree/image-25.png) |

**Full classification report (FinBERT):**
```
                     precision  recall  f1-score  support

Logistic Regression
  Negative (0)          0.92    0.95      0.94       84
  Neutral  (1)          0.98    0.98      0.98      429
  Positive (2)          0.95    0.94      0.94      178
  macro avg             0.95    0.96      0.95      691

Linear SVM
  Negative (0)          0.93    0.93      0.93       84
  Neutral  (1)          0.98    0.97      0.97      429
  Positive (2)          0.93    0.94      0.94      178
  macro avg             0.95    0.95      0.95      691

RBF SVM
  Negative (0)          0.94    0.95      0.95       84
  Neutral  (1)          0.98    0.98      0.98      429
  Positive (2)          0.94    0.95      0.95      178
  macro avg             0.96    0.96      0.96      691

Random Forest
  Negative (0)          0.95    0.96      0.96       84
  Neutral  (1)          0.97    0.98      0.98      429
  Positive (2)          0.95    0.93      0.94      178
  macro avg             0.96    0.96      0.96      691

Naive Bayes
  Negative (0)          0.93    0.96      0.95       84
  Neutral  (1)          0.98    0.98      0.98      429
  Positive (2)          0.95    0.93      0.94      178
  macro avg             0.95    0.96      0.95      691
```

---

## Summary

| Classifier | MiniLM | FinBERT |
|---|---|---|
| Logistic Regression | 83.07% | 96.53% |
| Linear SVM | 83.65% | 95.95% |
| RBF SVM | 87.70% | **96.67%** |
| Random Forest | 81.33% | **96.67%** |
| Naive Bayes | 73.95% | 96.38% |

