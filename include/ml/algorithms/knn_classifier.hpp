#pragma once

#include "ml/core/matrix/matrix.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <vector>

namespace ml {

/**
 * @brief K-Nearest-Neighbors classifier implemented from scratch.
 *
 * Lazy learner: fit() only stores the training data, all work happens
 * at prediction time using a brute-force Euclidean distance search.
 */
class KNNClassifier {
public:
    explicit KNNClassifier(size_t n_neighbors = 5) : n_neighbors_(n_neighbors) {
        if (n_neighbors_ == 0) throw std::invalid_argument("KNNClassifier: n_neighbors must be at least 1");
    }

    void fit(const Matrix& X, const std::vector<int>& y) {
        if (X.rows != y.size()) throw std::invalid_argument("KNNClassifier::fit: X and y must have matching row counts");
        if (X.rows < n_neighbors_) throw std::invalid_argument("KNNClassifier::fit: fewer samples than n_neighbors");
        X_ = X;
        y_ = y;
        fitted_ = true;
    }

    std::vector<int> predict(const Matrix& X) const {
        if (!fitted_) throw std::runtime_error("KNNClassifier::predict: model is not fitted");
        if (X.cols != X_.cols) throw std::invalid_argument("KNNClassifier::predict: feature count mismatch");

        std::vector<int> predictions;
        predictions.reserve(X.rows);
        for (size_t i = 0; i < X.rows; ++i) {
            predictions.push_back(predict_single(X.row(i)));
        }
        return predictions;
    }

    bool is_fitted() const { return fitted_; }

private:
    size_t n_neighbors_;
    Matrix X_;
    std::vector<int> y_;
    bool fitted_ = false;

    static double squared_euclidean_distance(const std::vector<double>& a, const std::vector<double>& b) {
        double sum = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            const double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum;
    }

    int predict_single(const std::vector<double>& sample) const {
        std::vector<std::pair<double, int>> distances;
        distances.reserve(X_.rows);
        for (size_t i = 0; i < X_.rows; ++i) {
            distances.emplace_back(squared_euclidean_distance(sample, X_.row(i)), y_[i]);
        }
        std::partial_sort(
            distances.begin(),
            distances.begin() + static_cast<std::ptrdiff_t>(n_neighbors_),
            distances.end(),
            [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; }
        );

        std::map<int, size_t> votes;
        for (size_t i = 0; i < n_neighbors_; ++i) {
            ++votes[distances[i].second];
        }

        int best_label = votes.begin()->first;
        size_t best_count = 0;
        for (const auto& [label, count] : votes) {
            if (count > best_count) {
                best_count = count;
                best_label = label;
            }
        }
        return best_label;
    }
};

} // namespace ml
