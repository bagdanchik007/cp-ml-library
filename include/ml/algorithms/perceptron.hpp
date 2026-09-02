#pragma once

#include "ml/core/matrix/matrix.hpp"

#include <stdexcept>
#include <vector>

namespace ml {

/**
 * @brief Classic single-layer Perceptron for binary classification.
 *
 * Labels are expected as {0, 1}. Internally trained on {-1, +1} targets
 * using the perceptron learning rule: weights are nudged towards
 * misclassified samples until the data is separated or epochs run out.
 */
class Perceptron {
public:
    Perceptron() = default;

    void fit(const Matrix& X, const std::vector<int>& y, double learning_rate = 0.1, int epochs = 100) {
        if (X.rows != y.size() || X.rows == 0) {
            throw std::invalid_argument("Perceptron::fit: X and y must be non-empty with matching row counts");
        }

        weights_.assign(X.cols, 0.0);
        bias_ = 0.0;

        for (int epoch = 0; epoch < epochs; ++epoch) {
            bool made_update = false;
            for (size_t i = 0; i < X.rows; ++i) {
                const double target = y[i] == 1 ? 1.0 : -1.0;
                const double activation = predict_raw(X.row(i));
                if (activation * target <= 0.0) {
                    for (size_t j = 0; j < X.cols; ++j) weights_[j] += learning_rate * target * X(i, j);
                    bias_ += learning_rate * target;
                    made_update = true;
                }
            }
            if (!made_update) break; // converged: data is linearly separable
        }
        fitted_ = true;
    }

    std::vector<int> predict(const Matrix& X) const {
        if (!fitted_) throw std::runtime_error("Perceptron::predict: model is not fitted");
        std::vector<int> predictions;
        predictions.reserve(X.rows);
        for (size_t i = 0; i < X.rows; ++i) predictions.push_back(predict_raw(X.row(i)) >= 0.0 ? 1 : 0);
        return predictions;
    }

    const std::vector<double>& weights() const { return weights_; }
    double bias() const { return bias_; }
    bool is_fitted() const { return fitted_; }

private:
    std::vector<double> weights_;
    double bias_ = 0.0;
    bool fitted_ = false;

    double predict_raw(const std::vector<double>& row) const {
        double sum = bias_;
        for (size_t j = 0; j < row.size(); ++j) sum += weights_[j] * row[j];
        return sum;
    }
};

} // namespace ml
