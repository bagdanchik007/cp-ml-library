#pragma once

#include "ml/core/matrix/matrix.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace ml {

/**
 * @brief Lasso Regression (L1-regularized linear regression) trained
 * with cyclic coordinate descent, since L1 regularization has no
 * closed-form normal-equation solution like Ridge.
 */
class LassoRegression {
public:
    explicit LassoRegression(double alpha = 1.0) : alpha_(alpha) {
        if (alpha_ < 0.0) throw std::invalid_argument("LassoRegression: alpha must be non-negative");
    }

    void fit(const Matrix& X, const std::vector<double>& y, int max_iterations = 1000, double tolerance = 1e-6) {
        if (X.rows != y.size() || X.rows == 0) {
            throw std::invalid_argument("LassoRegression::fit: X and y must be non-empty with matching row counts");
        }

        const auto n_samples = static_cast<double>(X.rows);

        // Center features and target so coordinate descent can ignore the
        // bias term entirely; the bias is recovered afterwards from the means.
        std::vector<double> feature_mean(X.cols, 0.0);
        for (size_t j = 0; j < X.cols; ++j) {
            double sum = 0.0;
            for (size_t i = 0; i < X.rows; ++i) sum += X(i, j);
            feature_mean[j] = sum / n_samples;
        }
        const double target_mean = mean(y);

        Matrix centered_X(X.rows, X.cols);
        for (size_t i = 0; i < X.rows; ++i) {
            for (size_t j = 0; j < X.cols; ++j) centered_X(i, j) = X(i, j) - feature_mean[j];
        }
        std::vector<double> residual(X.rows);
        for (size_t i = 0; i < X.rows; ++i) residual[i] = y[i] - target_mean;

        weights_.assign(X.cols, 0.0);

        for (int iteration = 0; iteration < max_iterations; ++iteration) {
            double max_change = 0.0;

            for (size_t j = 0; j < X.cols; ++j) {
                double column_norm_sq = 0.0;
                double rho = 0.0;
                for (size_t i = 0; i < X.rows; ++i) {
                    const double x_ij = centered_X(i, j);
                    column_norm_sq += x_ij * x_ij;
                    rho += x_ij * (residual[i] + weights_[j] * x_ij);
                }
                if (column_norm_sq < 1e-12) continue;

                const double old_weight = weights_[j];
                const double new_weight = soft_threshold(rho, alpha_ * n_samples) / column_norm_sq;

                for (size_t i = 0; i < X.rows; ++i) residual[i] -= (new_weight - old_weight) * centered_X(i, j);

                weights_[j] = new_weight;
                max_change = std::max(max_change, std::abs(new_weight - old_weight));
            }

            if (max_change < tolerance) break;
        }

        bias_ = target_mean;
        for (size_t j = 0; j < X.cols; ++j) bias_ -= weights_[j] * feature_mean[j];
        fitted_ = true;
    }

    std::vector<double> predict(const Matrix& X) const {
        if (!fitted_) throw std::runtime_error("LassoRegression::predict: model is not fitted");
        std::vector<double> predictions;
        predictions.reserve(X.rows);
        for (size_t i = 0; i < X.rows; ++i) {
            double value = bias_;
            for (size_t j = 0; j < X.cols; ++j) value += weights_[j] * X(i, j);
            predictions.push_back(value);
        }
        return predictions;
    }

    const std::vector<double>& weights() const { return weights_; }
    double bias() const { return bias_; }
    bool is_fitted() const { return fitted_; }

private:
    double alpha_;
    std::vector<double> weights_;
    double bias_ = 0.0;
    bool fitted_ = false;

    static double mean(const std::vector<double>& values) {
        double sum = 0.0;
        for (double value : values) sum += value;
        return sum / static_cast<double>(values.size());
    }

    static double soft_threshold(double value, double threshold) {
        if (value > threshold) return value - threshold;
        if (value < -threshold) return value + threshold;
        return 0.0;
    }
};

} // namespace ml
