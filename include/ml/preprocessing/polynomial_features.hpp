#pragma once

#include "ml/core/matrix/matrix.hpp"

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

    size_t degree_;
    bool include_bias_;

    size_t feature_count_ = 0;

    bool fitted_ = false;

};

} // namespace ml