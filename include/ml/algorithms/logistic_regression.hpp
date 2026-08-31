#pragma once

#include "ml/core/matrix/matrix.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace ml {

/**
 * @brief Binary Logistic Regression implemented from scratch.
 *
 * Trained with batch gradient descent on the cross-entropy loss.
 * Predicts class labels in {0, 1}.
 */
class LogisticRegression {
public:
    LogisticRegression() = default;

    void fit(
        const Matrix& X,
        const std::vector<int>& y,
        double learning_rate = 0.1,
        int epochs = 1000
    ) {
        if (X.rows != y.size() || X.rows == 0) {
            throw std::invalid_argument("LogisticRegression::fit: X and y must be non-empty with matching row counts");
        }

        weights_.assign(X.cols, 0.0);
        bias_ = 0.0;
        const auto n_samples = static_cast<double>(X.rows);

        for (int epoch = 0; epoch < epochs; ++epoch) {
            std::vector<double> weight_gradient(X.cols, 0.0);
            double bias_gradient = 0.0;

            for (size_t i = 0; i < X.rows; ++i) {
                const double prediction = sigmoid(dot(X.row(i)) + bias_);
                const double error = prediction - static_cast<double>(y[i]);
                for (size_t j = 0; j < X.cols; ++j) weight_gradient[j] += error * X(i, j);
                bias_gradient += error;
            }

            for (size_t j = 0; j < X.cols; ++j) weights_[j] -= learning_rate * weight_gradient[j] / n_samples;
            bias_ -= learning_rate * bias_gradient / n_samples;
        }
        fitted_ = true;
    }

    std::vector<double> predict_proba(const Matrix& X) const {
        if (!fitted_) throw std::runtime_error("LogisticRegression::predict_proba: model is not fitted");
        std::vector<double> probabilities;
        probabilities.reserve(X.rows);
        for (size_t i = 0; i < X.rows; ++i) probabilities.push_back(sigmoid(dot(X.row(i)) + bias_));
        return probabilities;
    }

    std::vector<int> predict(const Matrix& X, double threshold = 0.5) const {
        const auto probabilities = predict_proba(X);
        std::vector<int> labels;
        labels.reserve(probabilities.size());
        for (double probability : probabilities) labels.push_back(probability >= threshold ? 1 : 0);
        return labels;
    }

    const std::vector<double>& weights() const { return weights_; }
    double bias() const { return bias_; }
    bool is_fitted() const { return fitted_; }

private:
    std::vector<double> weights_;
    double bias_ = 0.0;
    bool fitted_ = false;

    static double sigmoid(double z) {
        return 1.0 / (1.0 + std::exp(-z));
    }

    double dot(const std::vector<double>& row) const {
        double sum = 0.0;
        for (size_t j = 0; j < row.size(); ++j) sum += weights_[j] * row[j];
        return sum;
    }
};

} // namespace ml
