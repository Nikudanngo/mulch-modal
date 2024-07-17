import MeCab
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
 
# 形態素解析
def tokenize(text):
    mecab = MeCab.Tagger("-Owakati")
    return mecab.parse(text).strip().split()
 
# 外部テキストファイルからデータを読み込む
def load_abstracts(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        abstracts = file.read().strip().split("\n\n")
    return [abstract.replace('\n', ' ') for abstract in abstracts]
 
# データの読み込み（例としてのファイルパス）
file_path = 'abstracts.txt'
abstracts = load_abstracts(file_path)
 
# アブストラクトがあることを確認
if len(abstracts) == 0:
    raise ValueError("ファイルにアブストラクトが含まれていません。")
 
# 少なくとも3つのアブストラクトがあることを確認
if len(abstracts) < 4:
    raise ValueError("少なくとも4つのアブストラクトが必要です。")
 
# TF-IDF特徴量の抽出
vectorizer = TfidfVectorizer(tokenizer=tokenize)
X = vectorizer.fit_transform(abstracts)
 
# 次元削減
n_samples, n_features = X.shape
print(f"Number of samples: {n_samples}, Number of features: {n_features}")
 
pca = PCA(n_components=min(n_samples, n_features, 2))
X_pca = pca.fit_transform(X.toarray())
 
# クラスタリング
kmeans = KMeans(n_clusters=4)  # 4つのクラスタに分割
clusters = kmeans.fit_predict(X_pca)
 
# クラスタリング結果の視覚化
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('クラスタリング結果')
 
# 各ポイントに対応するアブストラクトのインデックスを表示
for i, text in enumerate(abstracts):
    plt.annotate(str(i), (X_pca[i, 0], X_pca[i, 1]))
 
plt.colorbar(scatter, label='Cluster')
plt.show()