# =========================
# 1. 导入库
# =========================
import pandas as pd
from sentence_transformers import SentenceTransformer

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# 2. 加载 & 清洗数据
# =========================
# 读取 txt 文件
df = pd.read_csv(
    "data/Sentences_50Agree.txt",
    sep="@",
    header=None,
    names=["sentence", "label"],
    encoding="latin1"
)

df["sentence"] = df["sentence"].str.strip()
df["label"] = df["label"].str.strip()

# 标签映射
label_map = {"negative": 0, "neutral": 1, "positive": 2}
df["label_id"] = df["label"].map(label_map)

print(df.head())
print("\nLabel distribution:")
print(df["label_id"].value_counts())


# =========================
# 3. 生成 Embedding
# =========================
# 加载预训练模型
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 提取 embedding（向量）
sentences = df["sentence"].tolist()
embeddings = model.encode(sentences, show_progress_bar=True)

print("Embedding shape:", embeddings.shape)


# =========================
# 4. 可视化（PCA / t-SNE）
# =========================
# PCA 降维到 2 维
pca = PCA(n_components=2)
reduced = pca.fit_transform(embeddings)

# 绘图
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    reduced[:, 0],
    reduced[:, 1],
    c=df["label_id"],
    cmap="viridis",
    alpha=0.7
)

plt.title("PCA Visualization of Financial PhraseBank Embeddings")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(scatter, ticks=[0, 1, 2], label="label_id")
plt.show()

# tsne = TSNE(n_components=2, perplexity=40, random_state=42)
# tsne_result = tsne.fit_transform(embeddings)
# 先降到 50 维
pca_50 = PCA(n_components=50)
embeddings_50 = pca_50.fit_transform(embeddings)

# 再做 t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_result = tsne.fit_transform(embeddings_50)
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    tsne_result[:, 0],
    tsne_result[:, 1],
    c=df["label_id"],
    cmap="viridis",
    alpha=0.7
)

plt.title("t-SNE Visualization of Financial PhraseBank Embeddings")
plt.colorbar(scatter, ticks=[0, 1, 2], label="label_id")
plt.show()

# 定义混淆矩阵函数
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Negative", "Neutral", "Positive"],
                yticklabels=["Negative", "Neutral", "Positive"])
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

# =========================
# 5. 训练 Logistic Regression
# =========================

# 划分训练集 / 测试集
X = embeddings
y = df["label_id"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# 训练 Logistic Regression 模型
clf = LogisticRegression(max_iter=2000) # 迭代2000次
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
plot_confusion_matrix(y_test, y_pred, "Confusion Matrix - Logistic Regression")


# =========================
# 6. 训练 Linear SVM
# =========================
svm_clf = LinearSVC()
svm_clf.fit(X_train, y_train)

# 预测
y_pred_svm = svm_clf.predict(X_test)

# 输出结果
print("Linear SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print("\nLinear SVM Classification Report:\n", classification_report(y_test, y_pred_svm))
plot_confusion_matrix(y_test, y_pred_svm, "Confusion Matrix - Linear SVM")


# =========================
# 7. 训练 RBF SVM
# =========================
# 1) PCA 降维到 50 维
pca = PCA(n_components=50, random_state=42)
X_reduced = pca.fit_transform(X)

# 2) 划分训练集 / 测试集
X_train_rbf, X_test_rbf, y_train_rbf, y_test_rbf = train_test_split(
    X_reduced, y, test_size=0.2, random_state=42, stratify=y
)

# 3) 训练 RBF SVM
rbf_clf = SVC(kernel="rbf", C=3, gamma="scale")
rbf_clf.fit(X_train_rbf, y_train_rbf)

# 4) 预测
y_pred_rbf = rbf_clf.predict(X_test_rbf)

# 5) 输出结果
print("RBF SVM Accuracy:", accuracy_score(y_test_rbf, y_pred_rbf))
print("\nRBF SVM Classification Report:\n", classification_report(y_test_rbf, y_pred_rbf))
plot_confusion_matrix(y_test_rbf, y_pred_rbf, "Confusion Matrix - RBF SVM")


# =========================
# 8. 训练 Random Forest
# =========================
rf_clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
rf_clf.fit(X_train, y_train)

# 预测
y_pred_rf = rf_clf.predict(X_test)

# 输出结果
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\nRandom Forest Classification Report:\n", classification_report(y_test, y_pred_rf))
plot_confusion_matrix(y_test, y_pred_rf, "Confusion Matrix - Random Forest")

# =========================
# 9. 训练 Naive Bayes
# =========================
nb_clf = GaussianNB()
nb_clf.fit(X_train, y_train)

# 预测
y_pred_nb = nb_clf.predict(X_test)

# 输出结果
print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
print("\nNaive Bayes Classification Report:\n", classification_report(y_test, y_pred_nb))
plot_confusion_matrix(y_test, y_pred_nb, "Confusion Matrix - Naive Bayes")

