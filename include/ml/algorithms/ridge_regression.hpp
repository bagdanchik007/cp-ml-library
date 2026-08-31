#pragma once

#include "ml/core/matrix/matrix.hpp"

#include <stdexcept>
#include <vector>

namespace ml {

/**
 * @brief Ridge Regression (L2-regularized linear regression) implemented from scratch.
 *
 * Solved in closed form via the normal equations:
 *   w = (XtX + alpha * I)^-1 * Xt * y
 * using the existing Matrix::inverse implementation. The bias term is
 * handled by augmenting X with a constant column of ones which is not
 * penalized.
 */
class RidgeRegression {
public:
    explicit RidgeRegression(double alpha = 1.0) : alpha_(alpha) {
        if (alpha_ < 0.0) throw std::invalid_argument("RidgeRegression: alpha must be non-negative");
    }

    void fit(const Matrix& X, const std::vector<double>& y) {
        if (X.rows != y.size() || X.rows == 0) {
            throw std::invalid_argument("RidgeRegression::fit: X and y must be non-empty with matching row counts");
        }

        const Matrix design = augment_with_bias(X);
        const Matrix design_transpose = design.transpose();

        Matrix gram = design_transpose * design;
        for (size_t i = 1; i < gram.rows; ++i) gram(i, i) += alpha_; // do not penalize the bias term

        Matrix target_column(y.size(), 1);
        target_column.set_column(0, y);
        const Matrix xty = design_transpose * target_column;

        const Matrix coefficients = gram.inverse() * xty;

        bias_ = coefficients(0, 0);
        weights_.assign(X.cols, 0.0);
        for (size_t j = 0; j < X.cols; ++j) weights_[j] = coefficients(j + 1, 0);
        fitted_ = true;
    }

    std::vector<double> predict(const Matrix& X) const {
        if (!fitted_) throw std::runtime_error("RidgeRegression::predict: model is not fitted");
        if (X.cols != weights_.size()) throw std::invalid_argument("RidgeRegression::predict: feature count mismatch");

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

    static Matrix augment_with_bias(const Matrix& X) {
        Matrix design(X.rows, X.cols + 1, 1.0);
        for (size_t i = 0; i < X.rows; ++i) {
            for (size_t j = 0; j < X.cols; ++j) design(i, j + 1) = X(i, j);
        }
        return design;
    }
};

} // namespace ml
