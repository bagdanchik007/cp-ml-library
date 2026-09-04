#pragma once

#include "ml/core/matrix/matrix.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

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

        combinations_.clear();

        generate_combinations(
            0,
            degree_,
            {}
        );

        fitted_ = true;
    }

    Matrix transform(
        const Matrix& data
    ) const
    {
        validate_transform_input(data);

        const size_t output_features =
            combinations_.size() +
            (include_bias_ ? 1 : 0);

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
                const auto& combination :
                combinations_
            ) {
                double value = 1.0;

                for (
                    size_t column :
                    combination
                ) {
                    value *= data(
                        row,
                        column
                    );
                }

                result(
                    row,
                    output_column
                ) = value;

                ++output_column;
            }
        }

        return result;
    }

    Matrix fit_transform(
        const Matrix& data
    )
    {
        fit(data);

        return transform(data);
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

    void generate_combinations(
        size_t start,
        size_t remaining,
        std::vector<size_t> combination
    )
    {
        if (!combination.empty()) {
            combinations_.push_back(
                combination
            );
        }

        if (
            combination.size() ==
            degree_
        ) {
            return;
        }

        for (
            size_t column = start;
            column < feature_count_;
            ++column
        ) {
            combination.push_back(
                column
            );

            generate_combinations(
                column,
                remaining - 1,
                combination
            );

            combination.pop_back();
        }
    }

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

    size_t degree_;

    bool include_bias_;

    size_t feature_count_ = 0;

    std::vector<
        std::vector<size_t>
    > combinations_;

    bool fitted_ = false;

};

} // namespace ml