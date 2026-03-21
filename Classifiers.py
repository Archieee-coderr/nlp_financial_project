import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import resample


# =========================
# 混淆矩阵绘制
# =========================
def plot_confusion_matrix(y_true, y_pred, title: str):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Neutral", "Positive"],
        yticklabels=["Negative", "Neutral", "Positive"],
    )
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()


# =========================
# 对训练集做上采样
# =========================
def oversample_train(X_train: np.ndarray, y_train: np.ndarray):
    """
    只对训练集做上采样，测试集完全不动。
    把少数类复制到和多数类（neutral）一样多。
    """
    # 找到多数类的数量
    unique, counts = np.unique(y_train, return_counts=True)
    n_majority = counts.max()

    X_parts, y_parts = [], []
    for label in unique:
        X_cls = X_train[y_train == label]
        y_cls = y_train[y_train == label]
        if len(X_cls) < n_majority:
            # 少数类：上采样
            X_cls, y_cls = resample(
                X_cls, y_cls,
                replace=True,
                n_samples=n_majority,
                random_state=42
            )
        X_parts.append(X_cls)
        y_parts.append(y_cls)

    X_balanced = np.vstack(X_parts)
    y_balanced = np.concatenate(y_parts)

    # 打乱顺序
    idx = np.random.RandomState(42).permutation(len(y_balanced))
    return X_balanced[idx], y_balanced[idx]


# =========================
# 单个分类器训练 & 评估
# =========================
def evaluate_classifier(clf, X_train, X_test, y_train, y_test, name: str) -> dict:
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n{'=' * 40}")
    print(f"{name} Accuracy: {acc:.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    plot_confusion_matrix(y_test, y_pred, f"Confusion Matrix - {name}")

    return {"model": name, "accuracy": acc}


# =========================
# 运行全部分类器
# =========================
def run_all_classifiers(embeddings: np.ndarray, labels: np.ndarray, model_name: str) -> list:
    """
    正确流程：先 split → 只对训练集上采样 → 测试集保持原始分布
    """
    print(f"\n{'#' * 50}")
    print(f"  Running classifiers with: {model_name}")
    print(f"{'#' * 50}")

    results = []

    # 第一步：先划分（用原始不平衡数据）
    X_train_raw, X_test, y_train_raw, y_test = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # 第二步：只对训练集上采样
    X_train, y_train = oversample_train(X_train_raw, y_train_raw)

    print(f"\nTraining set size after oversampling: {len(y_train)}")
    print(f"Test set size (original distribution): {len(y_test)}")

    # 1. Logistic Regression
    results.append(evaluate_classifier(
        LogisticRegression(max_iter=2000, class_weight="balanced"),
        X_train, X_test, y_train, y_test,
        f"[{model_name}] Logistic Regression"
    ))

    # 2. Linear SVM
    results.append(evaluate_classifier(
        LinearSVC(class_weight="balanced"),
        X_train, X_test, y_train, y_test,
        f"[{model_name}] Linear SVM"
    ))

    # 3. RBF SVM（先 PCA 降维）
    pca = PCA(n_components=100, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)   # 注意：用 transform 不是 fit_transform
    results.append(evaluate_classifier(
        SVC(kernel="rbf", C=3, gamma="scale", class_weight="balanced"),
        X_train_pca, X_test_pca, y_train, y_test,
        f"[{model_name}] RBF SVM"
    ))

    # 4. Random Forest
    results.append(evaluate_classifier(
        RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced"),
        X_train, X_test, y_train, y_test,
        f"[{model_name}] Random Forest"
    ))

    # 5. Naive Bayes
    results.append(evaluate_classifier(
        GaussianNB(),
        X_train, X_test, y_train, y_test,
        f"[{model_name}] Naive Bayes"
    ))

    return results