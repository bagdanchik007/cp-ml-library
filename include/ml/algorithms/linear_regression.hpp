#pragma once

#include "ml/core/estimators/regressor.hpp"

#include <vector>
#include <stdexcept>

namespace ml {

/**
 * @brief Multivariate Linear Regression implemented from scratch.
 *
 * Uses batch gradient descent for training.
 */
class LinearRegression : public Regressor {
public:
    LinearRegression() = default;

    /**
     * @brief Estimator-compatible training interface.
     *
     * Expects y to contain a single target column.
     */
    void fit(const Matrix& X, const Matrix& y) override;

    /**
     * @brief Train the model using batch gradient descent.
     */
    void fit(const Matrix& X,
             const std::vector<double>& y,
             double learning_rate = 0.01,
             int epochs = 1000,
             bool verbose = false);

    /**
     * @brief Estimator-compatible prediction interface.
     *
     * Returns one prediction per row as a single-column matrix.
     */
    Matrix predict(const Matrix& X) const override;

    /**
     * @brief Predict targets and return them as a vector.
     */
    std::vector<double> predict_vector(const Matrix& X) const;

    /**
     * @brief Mean Squared Error on a given dataset.
     */
    double score(const Matrix& X,
                 const std::vector<double>& y) const;

    const std::vector<double>& weights() const {
        return weights_;
    }

    double bias() const {
        return bias_;
    }

    bool is_fitted() const {
        return fitted_;
    }

private:
    std::vector<double> weights_;
    double bias_ = 0.0;
    bool fitted_ = false;

    double compute_mse(
        const std::vector<double>& predictions,
        const std::vector<double>& y
    ) const;
};

} // namespace ml