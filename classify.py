import MeCab
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import precision_score, recall_score, f1_score

def tokenize(text):
    """形態素解析を行い、トークンのリストを返す。"""
    mecab = MeCab.Tagger("-Owakati")
    return mecab.parse(text).strip().split()

def load_abstracts_from_csv(file_path):
    """CSVファイルからデータを読み込み、アブストラクトとラベルのリストを返す。"""
    abstracts = []
    labels = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                labels.append(int(row[0]))
                abstracts.append(row[1])
        return abstracts, labels
    except Exception as e:
        raise RuntimeError(f"ファイルの読み込み中にエラーが発生しました: {e}")

def check_abstracts(abstracts):
    """アブストラクトが適切に読み込まれているかを確認する。"""
    if not abstracts:
        raise ValueError("ファイルにアブストラクトが含まれていません。")
    if len(abstracts) < 4:
        raise ValueError("少なくとも4つのアブストラクトが必要です。")

def extract_features(abstracts):
    """TF-IDF特徴量を抽出し、次元削減を行う。"""
    vectorizer = TfidfVectorizer(tokenizer=tokenize)
    X = vectorizer.fit_transform(abstracts)
    
    n_samples, n_features = X.shape
    print(f"Number of samples: {n_samples}, Number of features: {n_features}")
    
    pca = PCA(n_components=min(n_samples, n_features, 2))
    X_pca = pca.fit_transform(X.toarray())
    
    return X_pca

def cluster_and_visualize(X_pca, abstracts, n_clusters=4):
    """クラスタリングを行い、結果を視覚化する。"""
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(X_pca)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('クラスタリング結果')
    
    for i, text in enumerate(abstracts):
        plt.annotate(str(i), (X_pca[i, 0], X_pca[i, 1]))
    
    plt.colorbar(scatter, label='Cluster')
    plt.show()

    return clusters

def evaluate_clustering(clusters, true_labels):
    """クラスタリングの結果を評価する。"""
    cluster_labels = np.zeros_like(clusters)
    for i in range(4):
        mask = (clusters == i)
        true_labels_in_cluster = np.array(true_labels)[mask]
        if len(true_labels_in_cluster) > 0:
            most_common_label = np.bincount(true_labels_in_cluster).argmax()
            cluster_labels[mask] = most_common_label
    
    accuracy = np.sum(cluster_labels == true_labels) / len(true_labels)
    print(f"Clustering accuracy: {accuracy * 100:.2f}%")
    
    for cluster_num in range(4):
        mask = (clusters == cluster_num)
        true_labels_in_cluster = np.array(true_labels)[mask]
        indices_in_cluster = np.where(mask)[0]
        print(f"Cluster {cluster_num}:")
        print(f"  True indices and labels: {list(zip(indices_in_cluster, true_labels_in_cluster))}")
        print(f"  Predicted labels: {cluster_labels[mask]}")
        
        precision = precision_score(true_labels_in_cluster, cluster_labels[mask], average='macro', zero_division=0)
        recall = recall_score(true_labels_in_cluster, cluster_labels[mask], average='macro', zero_division=0)
        f1 = f1_score(true_labels_in_cluster, cluster_labels[mask], average='macro', zero_division=0)
        
        print(f"  Precision: {precision:.2f}")
        print(f"  Recall: {recall:.2f}")
        print(f"  F1 Score: {f1:.2f}")
    
    return accuracy

def main(file_path):
    abstracts, true_labels = load_abstracts_from_csv(file_path)
    check_abstracts(abstracts)
    X_pca = extract_features(abstracts)
    clusters = cluster_and_visualize(X_pca, abstracts)

    evaluate_clustering(clusters, true_labels)

# データの読み込み（例としてのファイルパス）
file_path = 'abstracts.csv'
main(file_path)
