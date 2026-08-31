#pragma once

#include "ml/core/matrix/matrix.hpp"

#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <vector>

namespace ml {

/**
 * @brief Gaussian Naive Bayes classifier implemented from scratch.
 *
 * Assumes each feature is conditionally independent and normally
 * distributed given the class label. Fast to train (a single pass
 * over the data) and works well as a baseline classifier.
 */
class GaussianNaiveBayes {
public:
    void fit(const Matrix& X, const std::vector<int>& y) {
        if (X.rows != y.size() || X.rows == 0) {
            throw std::invalid_argument("GaussianNaiveBayes::fit: X and y must be non-empty with matching row counts");
        }

        class_stats_.clear();
        std::map<int, size_t> counts;
        for (size_t i = 0; i < X.rows; ++i) ++counts[y[i]];

        const size_t n_features = X.cols;
        const auto n_samples = static_cast<double>(X.rows);

        for (const auto& [label, count] : counts) {
            ClassStats stats;
            stats.prior = static_cast<double>(count) / n_samples;
            stats.mean.assign(n_features, 0.0);
            stats.variance.assign(n_features, 0.0);

            for (size_t i = 0; i < X.rows; ++i) {
                if (y[i] != label) continue;
                for (size_t j = 0; j < n_features; ++j) stats.mean[j] += X(i, j);
            }
            for (double& value : stats.mean) value /= static_cast<double>(count);

            for (size_t i = 0; i < X.rows; ++i) {
                if (y[i] != label) continue;
                for (size_t j = 0; j < n_features; ++j) {
                    const double diff = X(i, j) - stats.mean[j];
                    stats.variance[j] += diff * diff;
                }
            }
            for (double& value : stats.variance) {
                value /= static_cast<double>(count);
                value += 1e-9; // avoid division by zero for constant features
            }

            class_stats_[label] = std::move(stats);
        }
        fitted_ = true;
    }

    std::vector<int> predict(const Matrix& X) const {
        if (!fitted_) throw std::runtime_error("GaussianNaiveBayes::predict: model is not fitted");

        std::vector<int> predictions;
        predictions.reserve(X.rows);
        for (size_t i = 0; i < X.rows; ++i) predictions.push_back(predict_single(X.row(i)));
        return predictions;
    }

    bool is_fitted() const { return fitted_; }

private:
    struct ClassStats {
        double prior = 0.0;
        std::vector<double> mean;
        std::vector<double> variance;
    };

    std::map<int, ClassStats> class_stats_;
    bool fitted_ = false;

    static double log_gaussian_density(double x, double mean, double variance) {
        static constexpr double two_pi = 6.283185307179586;
        return -0.5 * std::log(two_pi * variance) - ((x - mean) * (x - mean)) / (2.0 * variance);
    }

    int predict_single(const std::vector<double>& sample) const {
        int best_label = class_stats_.begin()->first;
        double best_log_prob = -std::numeric_limits<double>::infinity();

        for (const auto& [label, stats] : class_stats_) {
            double log_prob = std::log(stats.prior);
            for (size_t j = 0; j < sample.size(); ++j) {
                log_prob += log_gaussian_density(sample[j], stats.mean[j], stats.variance[j]);
            }
            if (log_prob > best_log_prob) {
                best_log_prob = log_prob;
                best_label = label;
            }
        }
        return best_label;
    }
};

} // namespace ml
