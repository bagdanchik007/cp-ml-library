#include "ml/algorithms/kmeans.hpp"

#include <iostream>

namespace ml {

KMeans::KMeans(int n_clusters, int max_iters, unsigned seed)
    : k_(n_clusters),
      max_iters_(max_iters),
      seed_(seed) {
    if (k_ <= 0) {
        throw std::invalid_argument(
            "n_clusters must be positive"
        );
    }

    if (max_iters_ <= 0) {
        throw std::invalid_argument(
            "max_iters must be positive"
        );
    }
}

void KMeans::initialize_centroids(const Matrix& X) {
    // Simple random initialization.
    // Can be upgraded to k-means++ later.
    std::mt19937 gen(seed_);

    std::uniform_int_distribution<size_t> dist(
        0,
        X.rows - 1
    );

    centroids_ = Matrix(k_, X.cols);

    std::vector<bool> used(X.rows, false);

    for (int c = 0; c < k_; ++c) {
        size_t index;

        do {
            index = dist(gen);
        } while (used[index]);

        used[index] = true;

        centroids_.set_row(
            c,
            X.row(index)
        );
    }
}

int KMeans::closest_centroid(
    const std::vector<double>& point
) const {
    int best = 0;

    double best_distance =
        std::numeric_limits<double>::max();

    for (int c = 0; c < k_; ++c) {
        const double distance =
            euclidean_distance(
                point,
                centroids_.row(c)
            );

        if (distance < best_distance) {
            best_distance = distance;
            best = c;
        }
    }

    return best;
}

void KMeans::fit(const Matrix& X) {
    // Validate input data.
    if (X.rows == 0) {
        throw std::invalid_argument(
            "Input data must contain samples"
        );
    }

    if (X.cols == 0) {
        throw std::invalid_argument(
            "Input data must contain features"
        );
    }

    if (X.rows < static_cast<size_t>(k_)) {
        throw std::invalid_argument(
            "Not enough samples for the requested number of clusters"
        );
    }

    initialize_centroids(X);

    labels_.assign(
        X.rows,
        -1
    );

    for (int iter = 0; iter < max_iters_; ++iter) {
        bool changed = false;

        // -------------------------------------------------
        // Assignment step
        // -------------------------------------------------
        for (size_t i = 0; i < X.rows; ++i) {
            const int new_label =
                closest_centroid(
                    X.row(i)
                );

            if (labels_[i] != new_label) {
                labels_[i] = new_label;
                changed = true;
            }
        }

        // Stop when assignments no longer change.
        if (!changed) {
            break;
        }

        // -------------------------------------------------
        // Update step
        // -------------------------------------------------
        Matrix new_centroids =
            Matrix::zeros(
                k_,
                X.cols
            );

        std::vector<int> counts(
            k_,
            0
        );

        for (size_t i = 0; i < X.rows; ++i) {
            const int cluster =
                labels_[i];

            for (size_t j = 0; j < X.cols; ++j) {
                new_centroids(
                    cluster,
                    j
                ) += X(i, j);
            }

            counts[cluster]++;
        }

        // Calculate mean for every cluster.
        for (int c = 0; c < k_; ++c) {
            if (counts[c] > 0) {
                for (size_t j = 0; j < X.cols; ++j) {
                    new_centroids(c, j) /=
                        counts[c];
                }
            } else {
                // Re-initialize empty clusters.
                std::mt19937 gen(
                    seed_ + iter + c
                );

                std::uniform_int_distribution<size_t> dist(
                    0,
                    X.rows - 1
                );

                new_centroids.set_row(
                    c,
                    X.row(
                        dist(gen)
                    )
                );
            }
        }

        centroids_ = new_centroids;
    }

    fitted_ = true;
}

std::vector<int> KMeans::predict(
    const Matrix& X
) const {
    if (!fitted_) {
        throw std::runtime_error(
            "Model has not been fitted yet"
        );
    }

    if (X.cols != centroids_.cols) {
        throw std::invalid_argument(
            "Input feature count does not match fitted model"
        );
    }

    std::vector<int> result(
        X.rows
    );

    for (size_t i = 0; i < X.rows; ++i) {
        result[i] =
            closest_centroid(
                X.row(i)
            );
    }

    return result;
}

std::vector<int> KMeans::fit_predict(
    const Matrix& X
) {
    fit(X);

    return labels_;
}

} // namespace ml
