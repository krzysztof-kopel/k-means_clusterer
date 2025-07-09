import numpy as np
from sklearn.datasets import load_wine
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# scroll down to the bottom to implement your solution


def plot_comparison(data: np.ndarray, predicted_clusters: np.ndarray, true_clusters: np.ndarray = None,
                    centers: np.ndarray = None, show: bool = True):

    # Use this function to visualize the results on Stage 6.

    if true_clusters is not None:
        plt.figure(figsize=(20, 10))

        plt.subplot(1, 2, 1)
        sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=predicted_clusters, palette='deep')
        if centers is not None:
            sns.scatterplot(x=centers[:, 0], y=centers[:, 1], marker='X', color='k', s=200)
        plt.title('Predicted clusters')
        plt.xlabel('alcohol')
        plt.ylabel('malic_acid')
        plt.grid()

        plt.subplot(1, 2, 2)
        sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=true_clusters, palette='deep')
        if centers is not None:
            sns.scatterplot(x=centers[:, 0], y=centers[:, 1], marker='X', color='k', s=200)
        plt.title('Ground truth')
        plt.xlabel('alcohol')
        plt.ylabel('malic_acid')
        plt.grid()
    else:
        plt.figure(figsize=(10, 10))
        sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=predicted_clusters, palette='deep')
        if centers is not None:
            sns.scatterplot(x=centers[:, 0], y=centers[:, 1], marker='X', color='k', s=200)
        plt.title('Predicted clusters')
        plt.xlabel('alcohol')
        plt.ylabel('malic_acid')
        plt.grid()

    plt.savefig('Visualization.png', bbox_inches='tight')
    if show:
        plt.show()

class CustomKMeans:
    def __init__(self, k: int):
        self.k = k
        self.centers = None

    def fit(self, wine_samples: np.ndarray, eps=1e-6) -> None:
        self.centers = wine_samples[:self.k].copy()
        old_centers = None
        while old_centers is None or np.max(np.linalg.norm(self.centers - old_centers, axis=1)) > eps:
            centers_labels = self.predict(wine_samples)
            old_centers = self.centers.copy()
            for i in range(self.k):
                self.centers[i] = np.mean(wine_samples[centers_labels == i], axis=0)

    def predict(self, wine_samples: np.ndarray) -> np.ndarray:
        distances = np.linalg.norm(wine_samples[:, None, :] - self.centers[None, :, :], axis=2)
        return np.argmin(distances, axis=1)

    def inertia(self, wine_samples: np.ndarray) -> float:
        labels = self.predict(wine_samples)
        return np.sum((wine_samples - self.centers[labels]) ** 2)

    def silhouette(self, wine_samples: np.ndarray) -> float:
        labels = self.predict(wine_samples)
        return silhouette_score(wine_samples, labels)



if __name__ == '__main__':

    # Load data
    data = load_wine(as_frame=True, return_X_y=True)
    X_full, y_full = data

    # Permutate it to make things more interesting
    rnd = np.random.RandomState(42)
    permutations = rnd.permutation(len(X_full))
    X_full = X_full.iloc[permutations]
    y_full = y_full.iloc[permutations]

    # From dataframe to ndarray
    X_full = X_full.values
    y_full = y_full.values

    # Scale data
    scaler = StandardScaler()
    X_full = scaler.fit_transform(X_full)
    
    k_range = range(2, 11)

    inertias = []
    for k in k_range:
        clusterer = CustomKMeans(k)
        clusterer.fit(X_full)
        inertias.append(clusterer.inertia(X_full))
    # print([float(i) for i in inertias])

    silhouettes = []
    for k in k_range:
        clusterer = CustomKMeans(k)
        clusterer.fit(X_full)
        silhouettes.append(clusterer.silhouette(X_full))
    # print([float(i) for i in silhouettes])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(10, 5))
    ax1.set_title("Inertia vs k")
    ax1.set_xlabel("k")
    ax1.set_ylabel("inertia")
    ax1.scatter(k_range, inertias)
    
    ax2.set_title("Silhouette score vs k")
    ax2.set_xlabel("k")
    ax2.set_ylabel("silhouette score")
    ax2.scatter(k_range, silhouettes)

    fig.show()

    optimal_k = 3
    optimal_clusterer = CustomKMeans(optimal_k)
    optimal_clusterer.fit(X_full)
    predictions = optimal_clusterer.predict(X_full[:20]).tolist()
    print(predictions)

    plot_comparison(X_full[:20], predictions, centers=optimal_clusterer.centers, true_clusters=y_full[:20])
