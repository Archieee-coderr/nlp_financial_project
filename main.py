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