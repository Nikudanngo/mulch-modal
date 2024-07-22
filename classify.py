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
    indices = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                labels.append(int(row[0]))
                abstracts.append(row[1])
                indices.append(reader.line_num)
        return abstracts, labels, indices
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
    plt.title('Clustering Result')
    
    for i, text in enumerate(abstracts):
        plt.annotate(str(i), (X_pca[i, 0], X_pca[i, 1]))
    
    plt.colorbar(scatter, label='Cluster')
    plt.show()

    return clusters

def evaluate_clustering(clusters, labels):
    """クラスタリングの結果を評価する。"""
    n_clusters = len(np.unique(clusters))
    cluster_labels = np.zeros_like(clusters)
    input_labels = np.array(labels)
    f1_score_list = []
    
    print(f"Number of clusters: {n_clusters}")
    print(input_labels)
    
    # 各クラスタ(0~3)ごとに順番にラベルを割り当てる
    for i in range(n_clusters):
        mask = (clusters == i)
        labels_in_cluster = np.array(labels)[mask]
        if len(labels_in_cluster) > 0:
            most_common_label = np.bincount(labels_in_cluster).argmax()
            cluster_labels[mask] = most_common_label
    
    print(cluster_labels)
    
    # クラスタリングの精度を計算
    accuracy = np.sum(cluster_labels == labels) / len(labels)
    print(f"Clustering accuracy: {accuracy * 100:.2f}%")
    
    # # 各クラスタ(0~3)の適合率、再現率、F1スコアを計算
    for i in range(n_clusters):
        precision = precision_score(input_labels, cluster_labels, labels=[i+1], average='micro')
        recall = recall_score(input_labels, cluster_labels, labels=[i+1], average='micro')
        f1 = f1_score(input_labels, cluster_labels, labels=[i+1], average='micro')
        f1_score_list.append(f1)
        print(f"Cluster {i+1} precision: {precision * 100:.2f}%")
        print(f"Cluster {i+1} recall: {recall * 100:.2f}%")
        print(f"Cluster {i+1} F1 score: {f1 * 100:.2f}%")

    print(f"Average F1 score: {np.mean(f1_score_list) * 100:.2f}%")

    return accuracy


def main(file_path):
    abstracts, labels, indices = load_abstracts_from_csv(file_path)

    check_abstracts(abstracts)
    X_pca = extract_features(abstracts)
    clusters = cluster_and_visualize(X_pca, abstracts)

    evaluate_clustering(clusters, labels)

# データの読み込み（例としてのファイルパス）
file_path = 'abstracts.csv'
main(file_path)
