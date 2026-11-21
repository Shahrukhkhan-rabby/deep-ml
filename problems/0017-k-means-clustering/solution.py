import numpy as np


def k_means_clustering(
    points: list[tuple[float, ...]],
    k: int,
    initial_centroids: list[tuple[float, ...]],
    max_iterations: int,
) -> list[tuple[float, ...]]:
    pts = np.asarray(points, dtype=float)
    centroids = np.asarray(initial_centroids, dtype=float)

    for _ in range(max_iterations):
        # Compute Euclidean distances from each point (m, 1, d) to each centroid (1, k, d)
        distances = np.linalg.norm(pts[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(distances, axis=1)

        new_centroids = centroids.copy()
        for cluster_idx in range(k):
            cluster_points = pts[labels == cluster_idx]
            if len(cluster_points) > 0:
                new_centroids[cluster_idx] = np.mean(cluster_points, axis=0)

        if np.allclose(new_centroids, centroids):
            centroids = new_centroids
            break

        centroids = new_centroids

    return [tuple(round(float(coord), 4) for coord in c) for c in centroids]