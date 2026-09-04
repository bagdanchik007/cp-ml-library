#pragma once

#include "ml/core/matrix/matrix.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace ml {

class PolynomialFeatures {

public:

    explicit PolynomialFeatures(
        size_t degree = 2,
        bool include_bias = true
    )
        : degree_(degree),
          include_bias_(include_bias)
    {
        if (degree_ == 0) {
            throw std::invalid_argument(
                "PolynomialFeatures: degree must be positive"
            );
        }
    }

    void fit(
        const Matrix& data
    )
    {
        if (
            data.rows == 0 ||
            data.cols == 0
        ) {
            throw std::invalid_argument(
                "PolynomialFeatures::fit: "
                "input data must not be empty"
            );
        }

        feature_count_ = data.cols;

        fitted_ = true;
    }

    Matrix transform(
        const Matrix& data
    ) const
    {
        validate_transform_input(data);

        const size_t output_features =
            calculate_output_features();

        Matrix result(
            data.rows,
            output_features
        );

        for (
            size_t row = 0;
            row < data.rows;
            ++row
        ) {
            size_t output_column = 0;

            if (include_bias_) {
                result(
                    row,
                    output_column
                ) = 1.0;

                ++output_column;
            }

            for (
                size_t power = 1;
                power <= degree_;
                ++power
            ) {
                for (
                    size_t column = 0;
                    column < data.cols;
                    ++column
                ) {
                    result(
                        row,
                        output_column
                    ) = std::pow(
                        data(row, column),
                        static_cast<double>(
                            power
                        )
                    );

                    ++output_column;
                }
            }
        }

        return result;
    }

    bool is_fitted() const noexcept
    {
        return fitted_;
    }

    size_t degree() const noexcept
    {
        return degree_;
    }

    bool include_bias() const noexcept
    {
        return include_bias_;
    }

private:

    void validate_transform_input(
        const Matrix& data
    ) const
    {
        if (!fitted_) {
            throw std::logic_error(
                "PolynomialFeatures::transform: "
                "transformer has not been fitted"
            );
        }

        if (
            data.rows == 0 ||
            data.cols == 0
        ) {
            throw std::invalid_argument(
                "PolynomialFeatures::transform: "
                "input data must not be empty"
            );
        }

        if (
            data.cols != feature_count_
        ) {
            throw std::invalid_argument(
                "PolynomialFeatures::transform: "
                "feature count mismatch"
            );
        }
    }

    size_t calculate_output_features() const
    {
        return
            feature_count_ * degree_ +
            (include_bias_ ? 1 : 0);
    }

    size_t degree_;

    bool include_bias_;

    size_t feature_count_ = 0;

    bool fitted_ = false;

};

} // namespace ml