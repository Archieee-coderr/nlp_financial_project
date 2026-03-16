# Financial PhraseBank Sentiment Classification

A lightweight NLP mini‑project exploring **representation learning** in the financial domain.  
The pipeline extracts sentence embeddings, visualizes them, and evaluates several shallow classifiers.

---

## Dataset  
**Financial PhraseBank**  
- ~4,800 financial news sentences  
- Labels: *Positive*, *Negative*, *Neutral*  
- Source: https://huggingface.co/datasets/takala/financial_phrasebank  

---

## Method

### **1. Embedding Extraction**
- Model: `sentence-transformers/all-MiniLM-L6-v2`  
- Output: 384‑dim dense embeddings for each sentence  

### **2. Dimensionality Reduction**
- PCA (2D) for visualization  
- Scatter plot colored by sentiment labels  

### **3. Classification Models**
Evaluated on extracted embeddings using scikit‑learn:

- Logistic Regression  
- Linear SVM  
- RBF SVM  
- Random Forest  
- Naive Bayes  

Metrics: **Accuracy**, **Macro F1**, **Confusion Matrix**

---

## Results Summary

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| Logistic Regression | 0.7649 | 0.72 |
| **Linear SVM** | **0.7845** | **0.75** |
| RBF SVM | 0.7783 | 0.74 |
| Random Forest | 0.7165 | 0.60 |
| Naive Bayes | 0.6649 | 0.60 |

**Linear SVM performs best**, suggesting the embedding space is close to linearly separable.

Confusion matrices and PCA plots are included in `/images/`.

---

## Project Structure

```
Financial_NLP_Project/
│
├── main.py
├── requirements.txt
├── README.md
│
├── images/
│   ├── PAC.png
|   ├── t-SNE.png
│   ├── confusion matrix logistic regression.png
│   ├── confusion matrix linear svm.png
│   ├── confusion matrix RBF svm.png
│   ├── confusion matrix random forest.png
│   └── cm_confusion matrix naive bayes.png
│
└── data/
    └── Sentences_50Agree.txt
```

Outputs include:
- Embeddings  
- PCA visualization  
- Classification metrics  
- Confusion matrices  
