#pragma once

#include "matrix/matrix.hpp"
#include <vector>
#include <stdexcept>
#include <cmath>

namespace ml {

/**
 * @brief Multivariate Linear Regression implemented from scratch.
 *
 * Supports:
 *  - Gradient Descent (recommended for this pure-C++ version)
 *  - Basic feature scaling helper
 *
 * This class is deliberately kept simple and transparent so that
 * the mathematical foundations remain visible – ideal for interviews
 * and portfolio reviews.
 */
class LinearRegression {
public:
    LinearRegression() = default;

    /**
     * @brief Fit the model using batch gradient descent.
     * @param X  Feature matrix (n_samples × n_features)
     * @param y  Target vector (n_samples)
     * @param learning_rate  Step size
     * @param epochs         Number of full passes over the data
     * @param verbose        Print loss every 100 epochs
     */
    void fit(const Matrix& X, const std::vector<double>& y,
             double learning_rate = 0.01,
             int epochs = 1000,
             bool verbose = false);

    /**
     * @brief Predict targets for new samples.
     */
    std::vector<double> predict(const Matrix& X) const;

    /**
     * @brief Mean Squared Error on a given dataset.
     */
    double score(const Matrix& X, const std::vector<double>& y) const;

    // Accessors
    const std::vector<double>& weights() const { return weights_; }
    double bias() const { return bias_; }
    bool is_fitted() const { return fitted_; }

private:
    std::vector<double> weights_;
    double bias_ = 0.0;
    bool fitted_ = false;

    double compute_mse(const std::vector<double>& predictions,
                       const std::vector<double>& y) const;
};

} // namespace ml
