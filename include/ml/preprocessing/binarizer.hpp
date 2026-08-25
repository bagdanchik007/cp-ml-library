#pragma once

#include "ml/core/matrix/matrix.hpp"

#include <stdexcept>

namespace ml {

class Binarizer {
public:
    explicit Binarizer(
        double threshold = 0.0
    )
        : threshold_(threshold) {
    }

    Matrix transform(
        const Matrix& data
    ) const {
        if (data.rows == 0 || data.cols == 0) {
            throw std::invalid_argument(
                "Binarizer::transform: input data must not be empty"
            );
        }

        Matrix result(data.rows, data.cols);

        for (size_t row = 0; row < data.rows; ++row) {
            for (size_t column = 0;
                 column < data.cols;
                 ++column) {

                result(row, column) =
                    data(row, column) > threshold_
                        ? 1.0
                        : 0.0;
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
    double threshold_;
};

} // namespace ml