#pragma once

#include "ml/core/matrix/matrix.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace ml {

/**
 * @brief Multiclass logistic regression (softmax regression) trained
 * from scratch with batch gradient descent on the cross-entropy loss.
 *
 * Class labels must be contiguous integers starting at 0 (0, 1, ..., k-1).
 */
class SoftmaxRegression {
public:
    SoftmaxRegression() = default;

    void fit(
        const Matrix& X,
        const std::vector<int>& y,
        size_t n_classes,
        double learning_rate = 0.1,
        int epochs = 1000
    ) {
        if (X.rows != y.size() || X.rows == 0) {
            throw std::invalid_argument("SoftmaxRegression::fit: X and y must be non-empty with matching row counts");
        }
        if (n_classes < 2) throw std::invalid_argument("SoftmaxRegression::fit: n_classes must be at least 2");

        n_classes_ = n_classes;
        weights_.assign(n_classes_, std::vector<double>(X.cols, 0.0));
        bias_.assign(n_classes_, 0.0);
        const auto n_samples = static_cast<double>(X.rows);

        for (int epoch = 0; epoch < epochs; ++epoch) {
            std::vector<std::vector<double>> weight_gradient(n_classes_, std::vector<double>(X.cols, 0.0));
            std::vector<double> bias_gradient(n_classes_, 0.0);

            for (size_t i = 0; i < X.rows; ++i) {
                const auto probabilities = softmax(scores(X.row(i)));
                for (size_t c = 0; c < n_classes_; ++c) {
                    const double error = probabilities[c] - (static_cast<size_t>(y[i]) == c ? 1.0 : 0.0);
                    for (size_t j = 0; j < X.cols; ++j) weight_gradient[c][j] += error * X(i, j);
                    bias_gradient[c] += error;
                }
            }

            for (size_t c = 0; c < n_classes_; ++c) {
                for (size_t j = 0; j < X.cols; ++j) weights_[c][j] -= learning_rate * weight_gradient[c][j] / n_samples;
                bias_[c] -= learning_rate * bias_gradient[c] / n_samples;
            }
        }
        fitted_ = true;
    }

    std::vector<std::vector<double>> predict_proba(const Matrix& X) const {
        if (!fitted_) throw std::runtime_error("SoftmaxRegression::predict_proba: model is not fitted");
        std::vector<std::vector<double>> result;
        result.reserve(X.rows);
        for (size_t i = 0; i < X.rows; ++i) result.push_back(softmax(scores(X.row(i))));
        return result;
    }

    std::vector<int> predict(const Matrix& X) const {
        const auto probabilities = predict_proba(X);
        std::vector<int> labels;
        labels.reserve(probabilities.size());
        for (const auto& row : probabilities) {
            const auto max_it = std::max_element(row.begin(), row.end());
            labels.push_back(static_cast<int>(std::distance(row.begin(), max_it)));
        }
        return labels;
    }

    bool is_fitted() const { return fitted_; }

private:
    size_t n_classes_ = 0;
    std::vector<std::vector<double>> weights_;
    std::vector<double> bias_;
    bool fitted_ = false;

    std::vector<double> scores(const std::vector<double>& row) const {
        std::vector<double> result(n_classes_, 0.0);
        for (size_t c = 0; c < n_classes_; ++c) {
            double sum = bias_[c];
            for (size_t j = 0; j < row.size(); ++j) sum += weights_[c][j] * row[j];
            result[c] = sum;
        }
        return result;
    }

    static std::vector<double> softmax(std::vector<double> logits) {
        const double max_logit = *std::max_element(logits.begin(), logits.end());
        double sum = 0.0;
        for (double& value : logits) {
            value = std::exp(value - max_logit);
            sum += value;
        }
        for (double& value : logits) value /= sum;
        return logits;
    }
};

} // namespace ml
