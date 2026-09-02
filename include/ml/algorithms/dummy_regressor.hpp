#pragma once

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace ml {

/**
 * @brief Baseline regressor that ignores input features entirely.
 *
 * Supports three strategies:
 *   - "mean": always predicts the training target mean
 *   - "median": always predicts the training target median
 *   - "constant": always predicts a user-supplied constant value
 */
class DummyRegressor {
public:
    explicit DummyRegressor(std::string strategy = "mean", double constant_value = 0.0)
        : strategy_(std::move(strategy)), constant_value_(constant_value) {
        if (strategy_ != "mean" && strategy_ != "median" && strategy_ != "constant") {
            throw std::invalid_argument("DummyRegressor: strategy must be 'mean', 'median' or 'constant'");
        }
    }

    void fit(const std::vector<double>& y) {
        if (strategy_ == "constant") {
            prediction_value_ = constant_value_;
            fitted_ = true;
            return;
        }

        if (y.empty()) throw std::invalid_argument("DummyRegressor::fit: y must not be empty");

        if (strategy_ == "mean") {
            double sum = 0.0;
            for (double value : y) sum += value;
            prediction_value_ = sum / static_cast<double>(y.size());
        } else { // median
            std::vector<double> sorted(y);
            std::sort(sorted.begin(), sorted.end());
            const size_t middle = sorted.size() / 2;
            prediction_value_ = sorted.size() % 2 == 0
                ? (sorted[middle - 1] + sorted[middle]) / 2.0
                : sorted[middle];
        }
        fitted_ = true;
    }

    std::vector<double> predict(size_t n_samples) const {
        if (!fitted_) throw std::runtime_error("DummyRegressor::predict: model is not fitted");
        return std::vector<double>(n_samples, prediction_value_);
    }

    double prediction_value() const { return prediction_value_; }
    bool is_fitted() const { return fitted_; }

private:
    std::string strategy_;
    double constant_value_;
    double prediction_value_ = 0.0;
    bool fitted_ = false;
};

} // namespace ml
