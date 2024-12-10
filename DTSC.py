import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_random_state
from sklearn.neighbors import kneighbors_graph
from sklearn.manifold import spectral_embedding
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise_distances_argmin_min
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score
import joblib
from scipy.sparse.csgraph import connected_components
from sklearn.metrics.pairwise import rbf_kernel, cosine_similarity, euclidean_distances

def _graph_is_connected(graph):
    """Return whether the graph is connected (True) or Not (False).

    Parameters
    ----------
    graph : {array-like, sparse matrix} of shape (n_samples, n_samples)
        Adjacency matrix of the graph, non-zero weight means an edge
        between the nodes.

    Returns
    -------
    is_connected : bool
        True means the graph is fully connected and False means not.
    """
    # sparse graph, find all the connected components
    n_connected_components, _ = connected_components(graph)
    return n_connected_components == 1

class DTSC:
    def __init__(self, n_clusters, random_state=None, n_jobs=None):
        """
        Initialize the DTSC model with the specified number of clusters,
        random state, and number of parallel jobs.

        Parameters
        ----------
        n_clusters : int
            The number of clusters to form.

        random_state : int, RandomState instance or None, optional
            Determines random number generation for centroid initialization.

        n_jobs : int or None, optional
            The number of jobs to use for the computation. None means 1.
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.best_affinity = None
        self.best_clustering_method = None

    def _compute_affinity_matrix(self, X, affinity, gamma):
        """
        Compute the affinity matrix for the given data and affinity type.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        affinity : str
            Type of affinity to use ('nearest_neighbors', 'rbf', 'cosine',
            'euclidean', 'manhattan', 'mahalanobis').

        gamma : float or str
            Kernel coefficient for 'rbf'. If 'scale', 'auto', or 'median',
            it is computed based on the data.

        Returns
        -------
        affinity_matrix : array-like of shape (n_samples, n_samples)
            Computed affinity matrix.
        """
        if affinity == 'nearest_neighbors':
            n_neighbors = 10
            while True:
                connectivity = kneighbors_graph(
                    X, n_neighbors=n_neighbors, include_self=True, n_jobs=self.n_jobs
                )
                affinity_matrix = 0.5 * (connectivity + connectivity.T)  # type: ignore # Symmetrize the matrix
                if _graph_is_connected(affinity_matrix):
                    break
                else:
                    n_neighbors += 1
            return affinity_matrix.toarray()

        elif affinity == 'rbf':
            if gamma == 'scale':
                gamma_value = 1.0 / X.shape[1]
            elif gamma == 'auto':
                gamma_value = 1.0 / X.shape[0]
            elif gamma == 'median':
                pairwise_dists = euclidean_distances(X)
                gamma_value = 1.0 / (2.0 * np.median(pairwise_dists) ** 2)
            else:
                gamma_value = gamma
            return rbf_kernel(X, gamma=gamma_value)

        elif affinity == 'cosine':
            return cosine_similarity(X)

        elif affinity == 'euclidean':
            distances = euclidean_distances(X)
            sigma = np.mean(distances)
            return np.exp(-distances ** 2 / (2. * sigma ** 2))

        elif affinity == 'manhattan':
            from sklearn.metrics.pairwise import manhattan_distances
            distances = manhattan_distances(X)
            sigma = np.mean(distances)
            return np.exp(-distances / sigma)

        elif affinity == 'mahalanobis':
            from scipy.spatial.distance import pdist, squareform
            VI = np.linalg.inv(np.cov(X.T))
            distances = squareform(pdist(X, 'mahalanobis', VI=VI))
            sigma = np.mean(distances)
            return np.exp(-distances ** 2 / (2. * sigma ** 2))

        else:
            raise ValueError("Unknown affinity type")

    def _compute_embedding(self, X, affinity, gamma):
        """
        Compute the spectral embedding of the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        affinity : str
            Type of affinity to use for the affinity matrix.

        gamma : float or str
            Parameter for the RBF kernel or other affinity types.

        Returns
        -------
        embedding : array-like of shape (n_samples, n_clusters)
            Spectral embedding of the data.
        """
        affinity_matrix = self._compute_affinity_matrix(X, affinity, gamma)
        random_state = check_random_state(self.random_state)
        return spectral_embedding(
            affinity_matrix,
            n_components=self.n_clusters,
            random_state=random_state,
            drop_first=False,
        )

    def _cluster_and_evaluate(self, embedding, labels):
        """
        Cluster the data and evaluate the clustering accuracy.

        Parameters
        ----------
        embedding : array-like of shape (n_samples, n_clusters)
            Spectral embedding of the data.

        labels : array-like of shape (n_samples,)
            True labels for the data.

        Returns
        -------
        best_accuracy : float
            Best clustering accuracy achieved.

        best_method : str
            Name of the clustering method with the best accuracy.
        """
        clustering_methods = [
            ('kmeans', KMeans(n_clusters=self.n_clusters, random_state=self.random_state)),
            ('gmm', GaussianMixture(n_components=self.n_clusters, random_state=self.random_state)),
            ('hierarchical', AgglomerativeClustering(n_clusters=self.n_clusters))
        ]

        best_accuracy = 0
        best_method = None

        for name, method in clustering_methods:
            if name == 'gmm':
                predicted_labels = method.fit_predict(embedding)
            else:
                predicted_labels = method.fit(embedding).labels_

            # Match predicted labels with true labels using the Hungarian algorithm
            row_ind, col_ind = linear_sum_assignment(pairwise_distances_argmin_min(predicted_labels[:, np.newaxis], labels[:, np.newaxis])[1])
            matched_labels = col_ind[predicted_labels]

            accuracy = accuracy_score(labels, matched_labels)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_method = name

        return best_accuracy, best_method

    def fit(self, X, labels):
        """
        Fit the model to the data and determine the best affinity and clustering method.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        labels : array-like of shape (n_samples,)
            True labels for the data.

        Returns
        -------
        None
        """
        affinities = [
            ('nearest_neighbors', None),
            ('rbf', 'scale'),
            ('rbf', 'auto'),
            ('rbf', 'median'),
            ('rbf', 1),
            ('cosine', None),
            ('euclidean', None),
            ('manhattan', None),
            ('mahalanobis', None)
        ]

        best_accuracy = 0

        for affinity, gamma in affinities:
            embedding = self._compute_embedding(X, affinity, gamma)
            accuracy, method = self._cluster_and_evaluate(embedding, labels)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.best_affinity = (affinity, gamma)
                self.best_clustering_method = method

        print(f"Best affinity: {self.best_affinity}, Best method: {self.best_clustering_method}, Accuracy: {best_accuracy}")

    def predict(self, X):
        """
        Predict cluster labels for the input data using the fitted model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to predict.

        Returns
        -------
        labels : array-like of shape (n_samples,)
            Predicted cluster labels.
        
        Raises
        ------
        ValueError
            If the model has not been fitted yet.
        """
        if self.best_affinity is None or self.best_clustering_method is None:
            raise ValueError("Model has not been fitted yet.")

        affinity, gamma = self.best_affinity
        embedding = self._compute_embedding(X, affinity, gamma)

        if self.best_clustering_method == 'kmeans':
            model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        elif self.best_clustering_method == 'gmm':
            model = GaussianMixture(n_components=self.n_clusters, random_state=self.random_state)
        elif self.best_clustering_method == 'hierarchical':
            model = AgglomerativeClustering(n_clusters=self.n_clusters)
        else:
            raise ValueError("Unknown clustering method")

        return model.fit_predict(embedding)

    def save(self, filepath):
        """
        Save the model to a file.

        Parameters
        ----------
        filepath : str
            Path to the file where the model will be saved.
        """
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath):
        """
        Load a model from a file.

        Parameters
        ----------
        filepath : str
            Path to the file from which the model will be loaded.

        Returns
        -------
        instance : DTSC
            Loaded model instance.
        """
        return joblib.load(filepath)
    

if __name__ == "__main__":

    # Generate synthetic data
    x, y = make_blobs(n_samples=300, centers=3, random_state=42, cluster_std=1.0) # type: ignore

    # Split the data into training and new data
    data, new_data, labels, _ = train_test_split(x, y, test_size=0.5, random_state=42)
    
    # Create and fit the model
    dtsc = DTSC(n_clusters=3, random_state=42)
    dtsc.fit(data, labels)

    # Predict using the fitted model
    predictions = dtsc.predict(new_data)

    # Save the model
    dtsc.save('dtsc_model.pkl')

    # Load the model
    loaded_dtsc = DTSC.load('dtsc_model.pkl')