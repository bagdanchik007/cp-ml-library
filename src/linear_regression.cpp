#include "ml/linear_regression.hpp"
#include <iostream>
#include <numeric>

namespace ml {

void LinearRegression::fit(const Matrix& X, const std::vector<double>& y,
                           double learning_rate, int epochs, bool verbose) {
    if (X.rows != y.size())
        throw std::invalid_argument("Number of samples in X and y must match");
    if (X.rows == 0)
        throw std::invalid_argument("Empty training set");

    const size_t n = X.rows;
    const size_t m = X.cols;

    weights_.assign(m, 0.0);
    bias_ = 0.0;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Forward pass
        std::vector<double> predictions = (*this).predict(X);  // uses current weights

        // Compute gradients
        std::vector<double> dw(m, 0.0);
        double db = 0.0;

        for (size_t i = 0; i < n; ++i) {
            double error = predictions[i] - y[i];
            for (size_t j = 0; j < m; ++j)
                dw[j] += error * X(i, j);
            db += error;
        }

        // Average and update
        for (size_t j = 0; j < m; ++j)
            weights_[j] -= (learning_rate / static_cast<double>(n)) * dw[j];
        bias_ -= (learning_rate / static_cast<double>(n)) * db;

        if (verbose && (epoch % 100 == 0 || epoch == epochs - 1)) {
            double loss = compute_mse(predictions, y);
            std::cout << "Epoch " << epoch << " | MSE: " << loss << "\n";
        }
    }

    fitted_ = true;
}

std::vector<double> LinearRegression::predict(const Matrix& X) const {
    if (!fitted_ && weights_.empty()) {
        // Allow calling predict during training (internal use)
        // but public callers should check is_fitted()
    }

    auto preds = X * weights_;          // matrix-vector product
    for (auto& p : preds)
        p += bias_;
    return preds;
}

double LinearRegression::score(const Matrix& X, const std::vector<double>& y) const {
    if (!fitted_)
        throw std::runtime_error("Model has not been fitted yet");
    auto preds = predict(X);
    return compute_mse(preds, y);
}

double LinearRegression::compute_mse(const std::vector<double>& predictions,
                                     const std::vector<double>& y) const {
    if (predictions.size() != y.size())
        throw std::invalid_argument("Size mismatch in MSE calculation");
    double sum = 0.0;
    for (size_t i = 0; i < y.size(); ++i) {
        double d = predictions[i] - y[i];
        sum += d * d;
    }
    return sum / static_cast<double>(y.size());
}

} // namespace ml
