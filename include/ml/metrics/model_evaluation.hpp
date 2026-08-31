#pragma once

#include "ml/core/matrix/matrix.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ml {

inline void validate_predictions(const std::vector<double>& actual, const std::vector<double>& predicted) {
    if (actual.empty() || actual.size() != predicted.size()) {
        throw std::invalid_argument("metrics: actual and predicted must be non-empty and equally sized");
    }
}

inline double mean_squared_error(const std::vector<double>& actual, const std::vector<double>& predicted) {
    validate_predictions(actual, predicted);
    double sum = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) { const double error = actual[i] - predicted[i]; sum += error * error; }
    return sum / static_cast<double>(actual.size());
}

inline double mean_absolute_error(const std::vector<double>& actual, const std::vector<double>& predicted) {
    validate_predictions(actual, predicted);
    double sum = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) sum += std::abs(actual[i] - predicted[i]);
    return sum / static_cast<double>(actual.size());
}

inline double r2_score(const std::vector<double>& actual, const std::vector<double>& predicted) {
    validate_predictions(actual, predicted);
    double mean = 0.0;
    for (double value : actual) mean += value;
    mean /= static_cast<double>(actual.size());
    double residual = 0.0, total = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) { const double d = actual[i] - mean; total += d * d; const double e = actual[i] - predicted[i]; residual += e * e; }
    return total == 0.0 ? (residual == 0.0 ? 1.0 : 0.0) : 1.0 - residual / total;
}

inline double accuracy_score(const std::vector<int>& actual, const std::vector<int>& predicted) {
    if (actual.empty() || actual.size() != predicted.size()) throw std::invalid_argument("accuracy_score: inputs must be non-empty and equally sized");
    size_t correct = 0;
    for (size_t i = 0; i < actual.size(); ++i) if (actual[i] == predicted[i]) ++correct;
    return static_cast<double>(correct) / static_cast<double>(actual.size());
}

/**
 * @brief Precision for the positive class (label == 1) in binary classification.
 *
 * precision = true_positives / (true_positives + false_positives)
 */
inline double precision_score(const std::vector<int>& actual, const std::vector<int>& predicted, int positive_label = 1) {
    if (actual.empty() || actual.size() != predicted.size()) throw std::invalid_argument("precision_score: inputs must be non-empty and equally sized");
    size_t true_positive = 0, predicted_positive = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (predicted[i] == positive_label) {
            ++predicted_positive;
            if (actual[i] == positive_label) ++true_positive;
        }
    }
    return predicted_positive == 0 ? 0.0 : static_cast<double>(true_positive) / static_cast<double>(predicted_positive);
}

/**
 * @brief Recall for the positive class (label == 1) in binary classification.
 *
 * recall = true_positives / (true_positives + false_negatives)
 */
inline double recall_score(const std::vector<int>& actual, const std::vector<int>& predicted, int positive_label = 1) {
    if (actual.empty() || actual.size() != predicted.size()) throw std::invalid_argument("recall_score: inputs must be non-empty and equally sized");
    size_t true_positive = 0, actual_positive = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] == positive_label) {
            ++actual_positive;
            if (predicted[i] == positive_label) ++true_positive;
        }
    }
    return actual_positive == 0 ? 0.0 : static_cast<double>(true_positive) / static_cast<double>(actual_positive);
}

/**
 * @brief Harmonic mean of precision and recall for the positive class.
 */
inline double f1_score(const std::vector<int>& actual, const std::vector<int>& predicted, int positive_label = 1) {
    const double precision = precision_score(actual, predicted, positive_label);
    const double recall = recall_score(actual, predicted, positive_label);
    return (precision + recall) == 0.0 ? 0.0 : 2.0 * precision * recall / (precision + recall);
}

/**
 * @brief Root Mean Squared Error, i.e. sqrt(mean_squared_error(...)).
 */
inline double root_mean_squared_error(const std::vector<double>& actual, const std::vector<double>& predicted) {
    return std::sqrt(mean_squared_error(actual, predicted));
}

/**
 * @brief Square confusion matrix for integer class labels 0..n_classes-1.
 *
 * Row i, column j holds the count of samples with true label i predicted as label j.
 */
inline Matrix confusion_matrix(const std::vector<int>& actual, const std::vector<int>& predicted, size_t n_classes) {
    if (actual.empty() || actual.size() != predicted.size()) throw std::invalid_argument("confusion_matrix: inputs must be non-empty and equally sized");
    Matrix result(n_classes, n_classes, 0.0);
    for (size_t i = 0; i < actual.size(); ++i) {
        const auto true_label = static_cast<size_t>(actual[i]);
        const auto pred_label = static_cast<size_t>(predicted[i]);
        if (true_label >= n_classes || pred_label >= n_classes) {
            throw std::invalid_argument("confusion_matrix: label out of range for n_classes");
        }
        result(true_label, pred_label) += 1.0;
    }
    return result;
}

/**
 * @brief Mean silhouette coefficient over all samples for a clustering result.
 *
 * For each sample, compares the average distance to points in its own
 * cluster (a) against the average distance to points in the nearest
 * other cluster (b): s = (b - a) / max(a, b). Values range from -1
 * (poor clustering) to 1 (well-separated, dense clusters).
 */
inline double silhouette_score(const Matrix& X, const std::vector<int>& labels) {
    if (X.rows != labels.size() || X.rows < 2) {
        throw std::invalid_argument("silhouette_score: X and labels must have matching, non-trivial row counts");
    }

    std::vector<int> unique_labels = labels;
    std::sort(unique_labels.begin(), unique_labels.end());
    unique_labels.erase(std::unique(unique_labels.begin(), unique_labels.end()), unique_labels.end());
    if (unique_labels.size() < 2) throw std::invalid_argument("silhouette_score: requires at least 2 clusters");

    auto distance = [&X](size_t i, size_t j) {
        double sum = 0.0;
        for (size_t c = 0; c < X.cols; ++c) {
            const double diff = X(i, c) - X(j, c);
            sum += diff * diff;
        }
        return std::sqrt(sum);
    };

    double total = 0.0;
    for (size_t i = 0; i < X.rows; ++i) {
        double intra_sum = 0.0;
        size_t intra_count = 0;
        std::map<int, std::pair<double, size_t>> inter_sums;

        for (size_t j = 0; j < X.rows; ++j) {
            if (i == j) continue;
            const double d = distance(i, j);
            if (labels[j] == labels[i]) {
                intra_sum += d;
                ++intra_count;
            } else {
                auto& entry = inter_sums[labels[j]];
                entry.first += d;
                entry.second += 1;
            }
        }

        const double a = intra_count == 0 ? 0.0 : intra_sum / static_cast<double>(intra_count);
        double b = std::numeric_limits<double>::infinity();
        for (const auto& [label, sum_and_count] : inter_sums) {
            const double mean_distance = sum_and_count.first / static_cast<double>(sum_and_count.second);
            b = std::min(b, mean_distance);
        }

        const double denominator = std::max(a, b);
        total += denominator == 0.0 ? 0.0 : (b - a) / denominator;
    }

    return total / static_cast<double>(X.rows);
}

} // namespace ml
