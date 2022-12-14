import collections
import numpy as np
from sklearn.cluster import DBSCAN


def clean_predictions(prediction, cluster_diameter=9, threshold=450):
    """
    :param prediction: An ndarray that contains the prediction
    :param cluster_diameter: The max distance between two points in the same cluster
    :param threshold: The max amount of points inside clusters that we will clean

    :returns: The prediction without small clusters
    """
    x, y = np.where(prediction > 0)
    if len(x) == 0:
        return prediction

    X = []
    for i in range(len(x)):
        X.append([x[i], y[i]])

    labels = DBSCAN(eps=cluster_diameter).fit_predict(np.array(X))
    cluster_frequency = collections.Counter(labels)

    large_clusters = []
    for c in cluster_frequency:
        if cluster_frequency[c] >= threshold:
            large_clusters.append(c)

    mask = np.isin(labels, large_clusters)

    cleaned_predictions = np.zeros_like(prediction)
    cleaned_predictions[x[mask], y[mask]] = 255
    return cleaned_predictions
