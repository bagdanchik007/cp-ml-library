#pragma once

#include "ml/core/matrix/matrix.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ml {

/**
 * @brief Agglomerative (bottom-up) hierarchical clustering.
 *
 * Starts with every sample as its own cluster and repeatedly merges the
 * two closest clusters until only n_clusters remain. Supports single
 * linkage (distance between closest members) and complete linkage
 * (distance between farthest members).
 */
class AgglomerativeClustering {
public:
    enum class Linkage { Single, Complete };

    explicit AgglomerativeClustering(size_t n_clusters = 2, Linkage linkage = Linkage::Single)
        : n_clusters_(n_clusters), linkage_(linkage) {
        if (n_clusters_ == 0) throw std::invalid_argument("AgglomerativeClustering: n_clusters must be at least 1");
    }

    std::vector<int> fit_predict(const Matrix& X) {
        if (X.rows < n_clusters_) throw std::invalid_argument("AgglomerativeClustering: fewer samples than n_clusters");

        const size_t n = X.rows;
        std::vector<double> distance(n * n, 0.0);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                const double d = euclidean_distance(X.row(i), X.row(j));
                distance[i * n + j] = d;
                distance[j * n + i] = d;
            }
        }

        std::vector<std::vector<size_t>> clusters(n);
        for (size_t i = 0; i < n; ++i) clusters[i] = {i};

        while (clusters.size() > n_clusters_) {
            size_t best_a = 0, best_b = 1;
            double best_distance = std::numeric_limits<double>::infinity();

            for (size_t a = 0; a < clusters.size(); ++a) {
                for (size_t b = a + 1; b < clusters.size(); ++b) {
                    const double d = cluster_distance(clusters[a], clusters[b], distance, n);
                    if (d < best_distance) {
                        best_distance = d;
                        best_a = a;
                        best_b = b;
                    }
                }
            }

            clusters[best_a].insert(clusters[best_a].end(), clusters[best_b].begin(), clusters[best_b].end());
            clusters.erase(clusters.begin() + static_cast<std::ptrdiff_t>(best_b));
        }

        std::vector<int> labels(n, -1);
        for (size_t cluster_id = 0; cluster_id < clusters.size(); ++cluster_id) {
            for (size_t index : clusters[cluster_id]) labels[index] = static_cast<int>(cluster_id);
        }
        return labels;
    }

private:
    size_t n_clusters_;
    Linkage linkage_;

    static double euclidean_distance(const std::vector<double>& a, const std::vector<double>& b) {
        double sum = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            const double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }

    double cluster_distance(
        const std::vector<size_t>& a,
        const std::vector<size_t>& b,
        const std::vector<double>& distance,
        size_t n
    ) const {
        double result = linkage_ == Linkage::Single
            ? std::numeric_limits<double>::infinity()
            : -std::numeric_limits<double>::infinity();

        for (size_t i : a) {
            for (size_t j : b) {
                const double d = distance[i * n + j];
                result = linkage_ == Linkage::Single ? std::min(result, d) : std::max(result, d);
            }
        }
        return result;
    }
};

} // namespace ml
