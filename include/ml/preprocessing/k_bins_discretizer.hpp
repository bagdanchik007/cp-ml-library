#pragma once

#include "ml/core/matrix/matrix.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace ml {

/**
 * @brief Discretizes continuous features into equal-width bins.
 *
 * Each column is independently split into n_bins equal-width intervals
 * based on the min/max seen during fit(). transform() maps each value
 * to its bin index (0 .. n_bins - 1), clamping out-of-range values.
 */
class KBinsDiscretizer {
public:
    explicit KBinsDiscretizer(size_t n_bins = 5) : n_bins_(n_bins) {
        if (n_bins_ < 2) throw std::invalid_argument("KBinsDiscretizer: n_bins must be at least 2");
    }

    void fit(const Matrix& data) {
        if (data.rows == 0 || data.cols == 0) throw std::invalid_argument("KBinsDiscretizer::fit: input data must not be empty");

        min_.assign(data.cols, 0.0);
        width_.assign(data.cols, 0.0);

        for (size_t column = 0; column < data.cols; ++column) {
            double min_value = data(0, column);
            double max_value = data(0, column);
            for (size_t row = 1; row < data.rows; ++row) {
                min_value = std::min(min_value, data(row, column));
                max_value = std::max(max_value, data(row, column));
            }
            min_[column] = min_value;
            const double range = max_value - min_value;
            width_[column] = range > 1e-12 ? range / static_cast<double>(n_bins_) : 1.0;
        }
        fitted_ = true;
    }

    Matrix transform(const Matrix& data) const {
        if (!fitted_) throw std::logic_error("KBinsDiscretizer::transform: discretizer has not been fitted");
        if (data.cols != min_.size()) throw std::invalid_argument("KBinsDiscretizer::transform: feature count mismatch");

        Matrix result(data.rows, data.cols);
        for (size_t row = 0; row < data.rows; ++row) {
            for (size_t column = 0; column < data.cols; ++column) {
                auto bin = static_cast<long long>((data(row, column) - min_[column]) / width_[column]);
                bin = std::max<long long>(0, std::min<long long>(bin, static_cast<long long>(n_bins_) - 1));
                result(row, column) = static_cast<double>(bin);
            }
        }
        return result;
    }

    Matrix fit_transform(const Matrix& data) {
        fit(data);
        return transform(data);
    }

    size_t n_bins() const { return n_bins_; }

private:
    size_t n_bins_;
    std::vector<double> min_;
    std::vector<double> width_;
    bool fitted_ = false;
};

} // namespace ml
