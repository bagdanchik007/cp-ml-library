#pragma once

#include "ml/core/matrix/matrix.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace ml {

class MaxAbsScaler {
public:
    MaxAbsScaler() = default;

    void fit(
        const Matrix& data
    ) {
        if (data.rows == 0 || data.cols == 0) {
            throw std::invalid_argument(
                "MaxAbsScaler::fit: input data must not be empty"
            );
        }

        max_abs_values_.resize(
            data.cols
        );

        for (size_t column = 0;
             column < data.cols;
             ++column) {

            double max_abs = 0.0;

            for (size_t row = 0;
                 row < data.rows;
                 ++row) {

                const double value =
                    std::abs(data(row, column));

                if (value > max_abs) {
                    max_abs = value;
                }
            }

            max_abs_values_[column] =
                max_abs;
        }

        fitted_ = true;
    }

    Matrix transform(
        const Matrix& data
    ) const {
        if (!fitted_) {
            throw std::logic_error(
                "MaxAbsScaler::transform: "
                "scaler has not been fitted"
            );
        }

        if (data.rows == 0 || data.cols == 0) {
            throw std::invalid_argument(
                "MaxAbsScaler::transform: "
                "input data must not be empty"
            );
        }

        if (data.cols != max_abs_values_.size()) {
            throw std::invalid_argument(
                "MaxAbsScaler::transform: "
                "input column count does not match fitted data"
            );
        }

        Matrix result(data.rows, data.cols);

        for (size_t row = 0;
             row < data.rows;
             ++row) {

            for (size_t column = 0;
                 column < data.cols;
                 ++column) {

                const double max_abs =
                    max_abs_values_[column];

                if (max_abs == 0.0) {
                    result(row, column) = 0.0;
                } else {
                    result(row, column) =
                        data(row, column) / max_abs;
                }
            }
        }

        return result;
    }

    Matrix fit_transform(
        const Matrix& data
    ) {
        fit(data);

        return transform(data);
    }

private:
    std::vector<double> max_abs_values_;
    bool fitted_ = false;
};

} // namespace ml
