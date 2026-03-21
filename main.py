# # =========================
# # 1. 导入库
# # =========================
# import pandas as pd
# import numpy as np
# from sentence_transformers import SentenceTransformer
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.svm import LinearSVC
# from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.metrics import confusion_matrix
# import seaborn as sns
# from transformers import AutoTokenizer, AutoModel
# import torch
# from sklearn.utils import resample


# # =========================
# # 2. 加载 & 清洗数据
# # =========================
# # 读取 txt 文件
# df = pd.read_csv(
#     "data/Sentences_75Agree.txt",
#     sep="@",
#     header=None,
#     names=["sentence", "label"],
#     encoding="latin1"
# )

# df["sentence"] = df["sentence"].str.strip()
# df["label"] = df["label"].str.strip()

# # 标签映射
# label_map = {"negative": 0, "neutral": 1, "positive": 2}
# df["label_id"] = df["label"].map(label_map)


# # -----------------------------
# # 上采样 minority class
# # -----------------------------
# df_majority = df[df.label_id == 1]  # neutral
# df_negative = df[df.label_id == 0]  # negative
# df_positive = df[df.label_id == 2]  # positive

# # 上采样少数类
# df_negative_upsampled = resample(
#     df_negative,
#     replace=True,
#     n_samples=len(df_majority),
#     random_state=42
# )
# df_positive_upsampled = resample(
#     df_positive,
#     replace=True,
#     n_samples=len(df_majority),
#     random_state=42
# )

# # 合并
# df_balanced = pd.concat([df_majority, df_negative_upsampled, df_positive_upsampled])
# df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# print("Balanced Label Distribution:")
# print(df_balanced["label_id"].value_counts())

# # print("\nLabel distribution:")
# # print(df["label_id"].value_counts())


# # =========================
# # 3. 生成 Embedding
# # =========================
# # 加载预训练模型
# model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# # 提取 embedding（向量）
# # sentences = df["sentence"].tolist()
# # embeddings = model.encode(sentences, show_progress_bar=True)
# sentences_balanced = df_balanced["sentence"].tolist()
# labels_balanced = df_balanced["label_id"].values

# embeddings_balanced = model.encode(sentences_balanced, show_progress_bar=True)

# print("Embedding shape:", embeddings_balanced.shape)

# # =========================
# # 4. 数据分析
# # =========================
# print("\n===== Data Analysis =====")

# # 4.1 类别分布
# print("\nLabel Distribution:")
# print(df_balanced["label_id"].value_counts())

# plt.figure(figsize=(5,4))
# sns.countplot(x=df_balanced["label_id"])
# plt.title("Label Distribution")
# plt.xlabel("Label")
# plt.ylabel("Count")
# plt.show()

# # 4.2 句子长度分布
# df["length"] = df["sentence"].str.split().apply(len)

# print("\nSentence Length Statistics by Class:")
# print(df.groupby("label_id")["length"].describe())

# plt.figure(figsize=(6,4))
# sns.boxplot(x=df_balanced["label_id"], y=df["length"])
# plt.title("Sentence Length Distribution by Class")
# plt.xlabel("Label")
# plt.ylabel("Sentence Length")
# plt.show()

# # 4.3 每类 embedding 的均值向量
# mean_vecs = {}
# for label in [0, 1, 2]:
#     mean_vecs[label] = embeddings_balanced[df_balanced["label_id"] == label].mean(axis=0)

# # 计算类中心距离
# dist_01 = np.linalg.norm(mean_vecs[0] - mean_vecs[1])
# dist_02 = np.linalg.norm(mean_vecs[0] - mean_vecs[2])
# dist_12 = np.linalg.norm(mean_vecs[1] - mean_vecs[2])

# print("\nMean Vector Distances:")
# print(f"Negative vs Neutral: {dist_01:.4f}")
# print(f"Negative vs Positive: {dist_02:.4f}")
# print(f"Neutral vs Positive: {dist_12:.4f}")

# # 4.4 每类 embedding 的方差
# print("\nEmbedding Variance by Class:")
# for label in [0, 1, 2]:
#     var = np.var(embeddings_balanced[df_balanced["label_id"] == label], axis=0).mean()
#     print(f"Label {label}: {var:.6f}")

# # =========================
# # 5. 可视化（PCA / t-SNE）
# # =========================
# # PCA 降维到 2 维
# pca = PCA(n_components=2)
# reduced = pca.fit_transform(embeddings_balanced)

# # 绘图
# plt.figure(figsize=(8, 6))
# scatter = plt.scatter(
#     reduced[:, 0],
#     reduced[:, 1],
#     c=df_balanced["label_id"],
#     cmap="viridis",
#     alpha=0.7
# )

# plt.title("PCA Visualization of Financial PhraseBank Embeddings")
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.colorbar(scatter, ticks=[0, 1, 2], label="label_id")
# plt.show()

# # tsne = TSNE(n_components=2, perplexity=40, random_state=42)
# # tsne_result = tsne.fit_transform(embeddings)
# # 先降到 50 维
# pca_50 = PCA(n_components=50)
# embeddings_50 = pca_50.fit_transform(embeddings_balanced)

# # 再做 t-SNE

# tsne = TSNE(n_components=2, perplexity=30, random_state=42)
# tsne_result = tsne.fit_transform(embeddings_50)
# plt.figure(figsize=(8, 6))
# scatter = plt.scatter(
#     tsne_result[:, 0],
#     tsne_result[:, 1],
#     c=df_balanced["label_id"],
#     cmap="viridis",
#     alpha=0.7
# )
# plt.title("t-SNE Visualization of Financial PhraseBank Embeddings")
# plt.colorbar(scatter, ticks=[0, 1, 2], label="label_id")
# plt.show()
# # plt.figure(figsize=(8, 6))
# # scatter = plt.scatter(
# #     tsne_result[:, 0],
# #     tsne_result[:, 1],
# #     c=df["label_id"],
# #     cmap="tab10",        # 更适合分类
# #     s=8,                 # 更小的点
# #     alpha=0.6            # 更高透明度
# # )

# # plt.title("t-SNE Visualization of Financial PhraseBank Embeddings")
# # plt.colorbar(scatter, ticks=[0, 1, 2], label="label_id")
# # plt.tight_layout()
# # plt.show()



# # 定义混淆矩阵函数
# def plot_confusion_matrix(y_true, y_pred, title):
#     cm = confusion_matrix(y_true, y_pred)
#     plt.figure(figsize=(6, 5))
#     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
#                 xticklabels=["Negative", "Neutral", "Positive"],
#                 yticklabels=["Negative", "Neutral", "Positive"])
#     plt.title(title)
#     plt.xlabel("Predicted Label")
#     plt.ylabel("True Label")
#     plt.show()

# # =========================
# # 6. 训练 Logistic Regression
# # =========================

# # 划分训练集 / 测试集
# # X = embeddings_balanced
# # y = df["label_id"].values

# # X_train, X_test, y_train, y_test = train_test_split(
# #     X, y, test_size=0.2, random_state=42, stratify=y
# # )
# X = embeddings_balanced
# y = labels_balanced

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )
# # 训练 Logistic Regression 模型 不加 balanced
# # clf = LogisticRegression(max_iter=2000) # 迭代2000次
# # clf.fit(X_train, y_train)

# # 加 balanced
# clf = LogisticRegression(max_iter=2000, class_weight="balanced")
# clf.fit(X_train, y_train)

# y_pred = clf.predict(X_test)

# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("\nClassification Report:\n", classification_report(y_test, y_pred))
# plot_confusion_matrix(y_test, y_pred, "Confusion Matrix - Logistic Regression")


# # =========================
# # 7. 训练 Linear SVM
# # =========================
# svm_clf = LinearSVC(class_weight="balanced")
# svm_clf.fit(X_train, y_train)

# # 预测
# y_pred_svm = svm_clf.predict(X_test)

# # 输出结果
# print("Linear SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
# print("\nLinear SVM Classification Report:\n", classification_report(y_test, y_pred_svm))
# plot_confusion_matrix(y_test, y_pred_svm, "Confusion Matrix - Linear SVM")


# # =========================
# # 8. 训练 RBF SVM
# # =========================
# # 1) PCA 降维到 50 维
# pca = PCA(n_components=50, random_state=42)
# X_reduced = pca.fit_transform(X)

# # 2) 划分训练集 / 测试集
# X_train_rbf, X_test_rbf, y_train_rbf, y_test_rbf = train_test_split(
#     X_reduced, y, test_size=0.2, random_state=42, stratify=y
# )

# # 3) 训练 RBF SVM
# rbf_clf = SVC(kernel="rbf", C=3, gamma="scale", class_weight="balanced")
# rbf_clf.fit(X_train_rbf, y_train_rbf)

# # 4) 预测
# y_pred_rbf = rbf_clf.predict(X_test_rbf)

# # 5) 输出结果
# print("RBF SVM Accuracy:", accuracy_score(y_test_rbf, y_pred_rbf))
# print("\nRBF SVM Classification Report:\n", classification_report(y_test_rbf, y_pred_rbf))
# plot_confusion_matrix(y_test_rbf, y_pred_rbf, "Confusion Matrix - RBF SVM")


# # =========================
# # 9. 训练 Random Forest
# # =========================
# rf_clf = RandomForestClassifier(
#     n_estimators=300,
#     max_depth=None,
#     random_state=42,
#     n_jobs=-1,
#     class_weight="balanced"
# )
# rf_clf.fit(X_train, y_train)

# # 预测
# y_pred_rf = rf_clf.predict(X_test)

# # 输出结果
# print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
# print("\nRandom Forest Classification Report:\n", classification_report(y_test, y_pred_rf))
# plot_confusion_matrix(y_test, y_pred_rf, "Confusion Matrix - Random Forest")

# # =========================
# # 10. 训练 Naive Bayes
# # =========================
# nb_clf = GaussianNB()
# nb_clf.fit(X_train, y_train)

# # 预测
# y_pred_nb = nb_clf.predict(X_test)

# # 输出结果
# print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
# print("\nNaive Bayes Classification Report:\n", classification_report(y_test, y_pred_nb))
# plot_confusion_matrix(y_test, y_pred_nb, "Confusion Matrix - Naive Bayes")



# -------------------------------------------new Clouded-------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from Embeddings import get_minilm_embeddings, get_finbert_embeddings
from Classifiers import run_all_classifiers


# =========================
# 配置：在这里切换模型
# =========================
# 可选值："minilm" 或 "finbert" 或 "both"
EMBEDDING_MODEL = "both"


# =========================
# 1. 加载 & 清洗数据
# =========================
df = pd.read_csv(
    "data/Sentences_75Agree.txt",
    sep="@",
    header=None,
    names=["sentence", "label"],
    encoding="latin1"
)
df["sentence"] = df["sentence"].str.strip()
df["label"] = df["label"].str.strip()

label_map = {"negative": 0, "neutral": 1, "positive": 2}
df["label_id"] = df["label"].map(label_map)

print("Original Label Distribution:")
print(df["label_id"].value_counts())

# 直接用原始数据，不在这里做上采样
# 上采样移到 classifiers.py 里，在 split 之后只对训练集做
sentences = df["sentence"].tolist()
labels = df["label_id"].values


# =========================
# 2. 数据分析与可视化
# =========================
def run_data_analysis(df_orig, embeddings, model_name):
    print(f"\n===== Data Analysis [{model_name}] =====")

    # 2.1 类别分布
    plt.figure(figsize=(5, 4))
    sns.countplot(x=df_orig["label_id"])
    plt.title(f"Label Distribution [{model_name}]")
    plt.tight_layout()
    plt.show()

    # 2.2 句子长度分布
    df_orig = df_orig.copy()
    df_orig["length"] = df_orig["sentence"].str.split().apply(len)
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df_orig["label_id"], y=df_orig["length"])
    plt.title(f"Sentence Length Distribution [{model_name}]")
    plt.tight_layout()
    plt.show()

    # 2.3 类中心距离
    mean_vecs = {
        label: embeddings[df_orig["label_id"] == label].mean(axis=0)
        for label in [0, 1, 2]
    }
    print(f"Negative vs Neutral:  {np.linalg.norm(mean_vecs[0] - mean_vecs[1]):.4f}")
    print(f"Negative vs Positive: {np.linalg.norm(mean_vecs[0] - mean_vecs[2]):.4f}")
    print(f"Neutral  vs Positive: {np.linalg.norm(mean_vecs[1] - mean_vecs[2]):.4f}")

    # 2.4 PCA 可视化
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        reduced[:, 0], reduced[:, 1],
        c=df_orig["label_id"], cmap="viridis", alpha=0.7
    )
    plt.title(f"PCA Visualization [{model_name}]")
    plt.colorbar(scatter, ticks=[0, 1, 2], label="label_id")
    plt.tight_layout()
    plt.show()

    # 2.5 t-SNE 可视化
    pca_50 = PCA(n_components=50)
    embeddings_50 = pca_50.fit_transform(embeddings)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_result = tsne.fit_transform(embeddings_50)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        tsne_result[:, 0], tsne_result[:, 1],
        c=df_orig["label_id"], cmap="viridis", alpha=0.7
    )
    plt.title(f"t-SNE Visualization [{model_name}]")
    plt.colorbar(scatter, ticks=[0, 1, 2], label="label_id")
    plt.tight_layout()
    plt.show()


# =========================
# 3. 汇总对比
# =========================
def print_summary(all_results: dict):
    print("\n" + "=" * 60)
    print("  FINAL COMPARISON SUMMARY")
    print("=" * 60)
    for model_name, results in all_results.items():
        print(f"\n{model_name}:")
        for r in results:
            print(f"  {r['model']:<45} Accuracy: {r['accuracy']:.4f}")


# =========================
# 4. 主流程
# =========================
all_results = {}

if EMBEDDING_MODEL in ("minilm", "both"):
    embeddings_minilm = get_minilm_embeddings(sentences)
    run_data_analysis(df, embeddings_minilm, "MiniLM")
    all_results["MiniLM"] = run_all_classifiers(embeddings_minilm, labels, "MiniLM")

if EMBEDDING_MODEL in ("finbert", "both"):
    embeddings_finbert = get_finbert_embeddings(sentences)
    run_data_analysis(df, embeddings_finbert, "FinBERT")
    all_results["FinBERT"] = run_all_classifiers(embeddings_finbert, labels, "FinBERT")

print_summary(all_results)