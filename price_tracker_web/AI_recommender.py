import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

# -------------------------
# 1. 商品データ
# -------------------------
data = {
    "product_id": [101, 102, 103, 104, 105, 106],
    "product_name": [
        "Wireless Earbuds",
        "Bluetooth Headphones",
        "Smart Watch",
        "Fitness Tracker",
        "USB-C Charger",
        "Portable Power Bank"
    ],
    "category": [
        "Audio",
        "Audio",
        "Wearable",
        "Wearable",
        "Accessories",
        "Accessories"
    ],
    "description": [
        "Wireless earbuds with noise cancellation and long battery life",
        "Over-ear bluetooth headphones with high quality sound",
        "Smart watch with heart rate monitor and fitness tracking",
        "Lightweight fitness tracker for daily activity monitoring",
        "Fast charging USB-C charger for smartphones and tablets",
        "Portable power bank with high capacity and fast charging"
    ]
}

df = pd.DataFrame(data)

# -------------------------
# 2. TF-IDF
# -------------------------
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df["description"])

# -------------------------
# 3. Cosine Similarity
# -------------------------
cosine_sim = cosine_similarity(tfidf_matrix)

# -------------------------
# 4. 推薦関数
# -------------------------
def recommend_products(product_id, top_n=3):
    idx = df.index[df["product_id"] == product_id][0]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    indices = [i[0] for i in scores]
    return df.iloc[indices][["product_id", "product_name", "category"]]

print("\n🔍 推薦結果：")
print(recommend_products(101))

# ==================================================
# 📊 グラフ①：商品類似度（棒グラフ）
# ==================================================
def plot_similarity_bar(product_id):
    idx = df.index[df["product_id"] == product_id][0]
    plt.figure()
    plt.bar(df["product_name"], cosine_sim[idx])
    plt.xticks(rotation=45, ha="right")
    plt.title("Product Similarity Scores")
    plt.tight_layout()
    plt.show()

plot_similarity_bar(101)

# ==================================================
# 🔥 グラフ②：商品類似度ヒートマップ
# ==================================================
plt.figure()
sns.heatmap(
    cosine_sim,
    xticklabels=df["product_name"],
    yticklabels=df["product_name"],
    annot=True,
    cmap="coolwarm"
)
plt.title("Product Similarity Heatmap")
plt.tight_layout()
plt.show()

# ==================================================
# 🏷️ グラフ③：カテゴリ別商品数
# ==================================================
plt.figure()
df["category"].value_counts().plot(kind="bar")
plt.title("Product Count by Category")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# ==================================================
# 📈 グラフ④：TF-IDF 重要キーワード
# ==================================================
feature_names = tfidf.get_feature_names_out()
importance = tfidf_matrix.sum(axis=0).A1
keywords = pd.Series(importance, index=feature_names).sort_values(ascending=False)[:10]

plt.figure()
keywords.plot(kind="bar")
plt.title("Top 10 TF-IDF Keywords")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()

# ==================================================
# 🧭 グラフ⑤：商品分布（PCA 2D）
# ==================================================
pca = PCA(n_components=2)
reduced = pca.fit_transform(tfidf_matrix.toarray())

plt.figure()
plt.scatter(reduced[:, 0], reduced[:, 1])

for i, name in enumerate(df["product_name"]):
    plt.text(reduced[i, 0], reduced[i, 1], name)

plt.title("Product Distribution (PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.show()