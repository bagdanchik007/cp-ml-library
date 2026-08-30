#include "ml/algorithms/linear_regression.hpp"

#include <iostream>

namespace ml {

void LinearRegression::fit(
    const Matrix& X,
    const std::vector<double>& y,
    double learning_rate,
    int epochs,
    bool verbose
) {
    if (X.rows == 0 || X.cols == 0) {
        throw std::invalid_argument(
            "Training data must not be empty"
        );
    }

    if (X.rows != y.size()) {
        throw std::invalid_argument(
            "Number of samples in X and y must match"
        );
    }

    if (learning_rate <= 0.0) {
        throw std::invalid_argument(
            "Learning rate must be positive"
        );
    }

    if (epochs <= 0) {
        throw std::invalid_argument(
            "Number of epochs must be positive"
        );
    }

    const size_t n = X.rows;
    const size_t m = X.cols;

    weights_.assign(m, 0.0);
    bias_ = 0.0;
    fitted_ = false;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Forward pass.
        std::vector<double> predictions(n, 0.0);

        for (size_t i = 0; i < n; ++i) {
            double prediction = bias_;

            for (size_t j = 0; j < m; ++j) {
                prediction += X(i, j) * weights_[j];
            }

            predictions[i] = prediction;
        }

        // Compute gradients.
        std::vector<double> dw(m, 0.0);
        double db = 0.0;

        for (size_t i = 0; i < n; ++i) {
            const double error = predictions[i] - y[i];

            for (size_t j = 0; j < m; ++j) {
                dw[j] += error * X(i, j);
            }

            db += error;
        }

        // Gradient descent update.
        const double scale =
            learning_rate / static_cast<double>(n);

        for (size_t j = 0; j < m; ++j) {
            weights_[j] -= scale * dw[j];
        }

        bias_ -= scale * db;

        if (verbose &&
            (epoch % 100 == 0 || epoch == epochs - 1)) {

            const double loss =
                compute_mse(predictions, y);

            std::cout
                << "Epoch " << epoch
                << " | MSE: " << loss
                << '\n';
        }
    }

    fitted_ = true;
}

void LinearRegression::fit(
    const Matrix& X,
    const Matrix& y
) {
    if (y.cols != 1) {
        throw std::invalid_argument(
            "LinearRegression expects a single target column"
        );
    }

    if (y.rows != X.rows) {
        throw std::invalid_argument(
            "Number of samples in X and y must match"
        );
    }

    std::vector<double> target(y.rows);

    for (size_t i = 0; i < y.rows; ++i) {
        target[i] = y(i, 0);
    }

    fit(X, target);
}

std::vector<double> LinearRegression::predict_vector(
    const Matrix& X
) const {
    if (!fitted_) {
        throw std::runtime_error(
            "Model has not been fitted yet"
        );
    }

    if (X.rows == 0 || X.cols == 0) {
        throw std::invalid_argument(
            "Prediction data must not be empty"
        );
    }

    if (X.cols != weights_.size()) {
        throw std::invalid_argument(
            "Number of features in X does not match "
            "the trained model"
        );
    }

    std::vector<double> predictions =
        X * weights_;

    for (double& prediction : predictions) {
        prediction += bias_;
    }

    return predictions;
}

Matrix LinearRegression::predict(
    const Matrix& X
) const {
    const auto predictions = predict_vector(X);

    Matrix result(predictions.size(), 1);

    for (size_t i = 0; i < predictions.size(); ++i) {
        result(i, 0) = predictions[i];
    }

    return result;
}

double LinearRegression::score(
    const Matrix& X,
    const std::vector<double>& y
) const {
    if (!fitted_) {
        throw std::runtime_error(
            "Model has not been fitted yet"
        );
    }

    if (X.rows != y.size()) {
        throw std::invalid_argument(
            "Number of samples in X and y must match"
        );
    }

    if (X.rows == 0 || X.cols == 0) {
        throw std::invalid_argument(
            "Evaluation data must not be empty"
        );
    }

    const auto predictions = predict_vector(X);

    return compute_mse(predictions, y);
}

double LinearRegression::compute_mse(
    const std::vector<double>& predictions,
    const std::vector<double>& y
) const {
    if (predictions.size() != y.size()) {
        throw std::invalid_argument(
            "Size mismatch in MSE calculation"
        );
    }

    if (y.empty()) {
        throw std::invalid_argument(
            "Cannot compute MSE for empty data"
        );
    }

    double sum = 0.0;

    for (size_t i = 0; i < y.size(); ++i) {
        const double error =
            predictions[i] - y[i];

        sum += error * error;
    }

    return sum / static_cast<double>(y.size());
}

} // namespace ml