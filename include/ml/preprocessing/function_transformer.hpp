#pragma once

#include "ml/core/matrix/matrix.hpp"

#include <functional>
#include <stdexcept>

namespace ml {

/**
 * @brief Applies an arbitrary user-supplied function element-wise to a Matrix.
 *
 * Useful for one-off preprocessing steps that don't warrant a dedicated
 * transformer class (e.g. a custom clipping rule, a domain-specific
 * rescaling formula). An optional inverse function enables round-tripping,
 * mirroring the other transformers in this library.
 */
class FunctionTransformer {
public:
    using Function = std::function<double(double)>;

    explicit FunctionTransformer(Function forward, Function inverse = nullptr)
        : forward_(std::move(forward)), inverse_(std::move(inverse)) {
        if (!forward_) throw std::invalid_argument("FunctionTransformer: forward function must not be empty");
    }

    Matrix transform(const Matrix& data) const {
        Matrix result(data.rows, data.cols);
        for (size_t row = 0; row < data.rows; ++row) {
            for (size_t column = 0; column < data.cols; ++column) {
                result(row, column) = forward_(data(row, column));
            }
        }
        return result;
    }

    Matrix inverse_transform(const Matrix& data) const {
        if (!inverse_) throw std::logic_error("FunctionTransformer::inverse_transform: no inverse function was supplied");
        Matrix result(data.rows, data.cols);
        for (size_t row = 0; row < data.rows; ++row) {
            for (size_t column = 0; column < data.cols; ++column) {
                result(row, column) = inverse_(data(row, column));
            }
        }
        return result;
    }

    bool has_inverse() const { return static_cast<bool>(inverse_); }

private:
    Function forward_;
    Function inverse_;
};

} // namespace ml
