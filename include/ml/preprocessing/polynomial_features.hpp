#pragma once

#include "ml/core/matrix/matrix.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace ml {

class PolynomialFeatures {
public:
    explicit PolynomialFeatures(
        size_t degree = 2
    )
        : degree_(degree) {
        if (degree_ == 0) {
            throw std::invalid_argument(
                "PolynomialFeatures: degree must be greater than zero"
            );
        }
    }

    Matrix transform(
        const Matrix& data
    ) const {
        if (data.rows == 0 || data.cols == 0) {
            throw std::invalid_argument(
                "PolynomialFeatures::transform: input data must not be empty"
            );
        }

        Matrix result(
            data.rows,
            data.cols * degree_
        );

        for (size_t row = 0; row < data.rows; ++row) {
            for (size_t column = 0; column < data.cols; ++column) {
                const double value =
                    data(row, column);

                for (size_t power = 1;
                     power <= degree_;
                     ++power) {
                    const size_t result_column =
                        column * degree_ + (power - 1);

                    result(row, result_column) =
                        std::pow(
                            value,
                            static_cast<double>(power)
                        );
                }
            }
        }

        return result;
    }

    Matrix fit_transform(
        const Matrix& data
    ) const {
        return transform(data);
    }

private:
    size_t degree_;
};

} // namespace ml