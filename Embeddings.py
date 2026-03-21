import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel


def get_minilm_embeddings(sentences: list) -> np.ndarray:
    """
    使用 all-MiniLM-L6-v2 生成句子 embedding。
    输出维度：384
    """
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(sentences, show_progress_bar=True)
    print(f"[MiniLM] Embedding shape: {embeddings.shape}")
    return embeddings


def get_finbert_embeddings(sentences: list, batch_size: int = 32) -> np.ndarray:
    """
    使用 FinBERT 生成句子 embedding，取 [CLS] token 向量。
    输出维度：768
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[FinBERT] Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModel.from_pretrained("ProsusAI/finbert")
    model.eval()
    model.to(device)

    all_embeddings = []

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]

        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            output = model(**encoded)

        # 取 [CLS] token 作为句子表示
        cls_embeddings = output.last_hidden_state[:, 0, :]
        all_embeddings.append(cls_embeddings.cpu().numpy())

    embeddings = np.vstack(all_embeddings)
    print(f"[FinBERT] Embedding shape: {embeddings.shape}")
    return embeddings