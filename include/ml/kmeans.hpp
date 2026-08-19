#pragma once

#include "matrix/matrix.hpp"
#include <vector>
#include <random>
#include <limits>
#include <stdexcept>

namespace ml {

/**
 * @brief K-Means clustering implemented from scratch.
 *
 * Classic Lloyd's algorithm with k-means++ style initialization option.
 * Transparent and interview-friendly implementation.
 */
class KMeans {
public:
    explicit KMeans(int n_clusters = 3, int max_iters = 300, unsigned seed = 42);

    /**
     * @brief Fit the model to the data.
     * @param X  Data matrix (n_samples × n_features)
     */
    void fit(const Matrix& X);

    /**
     * @brief Predict cluster labels for new data.
     */
    std::vector<int> predict(const Matrix& X) const;

    /**
     * @brief Convenience: fit + predict on the same data.
     */
    std::vector<int> fit_predict(const Matrix& X);

    // Accessors
    const Matrix& centroids() const { return centroids_; }
    const std::vector<int>& labels() const { return labels_; }
    bool is_fitted() const { return fitted_; }
    int n_clusters() const { return k_; }

private:
    int k_;
    int max_iters_;
    unsigned seed_;
    Matrix centroids_;
    std::vector<int> labels_;
    bool fitted_ = false;

    void initialize_centroids(const Matrix& X);
    int closest_centroid(const std::vector<double>& point) const;
};

} // namespace ml
