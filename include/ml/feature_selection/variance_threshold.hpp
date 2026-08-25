#pragma once

#include "ml/core/matrix/matrix.hpp"

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace ml {

class VarianceThreshold {
public:
    explicit VarianceThreshold(
        double threshold = 0.0
    )
        : threshold_(threshold) {
        if (threshold < 0.0) {
            throw std::invalid_argument(
                "VarianceThreshold: threshold must not be negative"
            );
        }
    }

    void fit(
        const Matrix& data
    ) {
        validate_input(data);

        variances_.assign(
            data.cols,
            0.0
        );

        selected_columns_.clear();

        for (size_t column = 0;
             column < data.cols;
             ++column) {

            double mean = 0.0;

            for (size_t row = 0;
                 row < data.rows;
                 ++row) {
                mean += data(row, column);
            }

            mean /= static_cast<double>(data.rows);

            double variance = 0.0;

            for (size_t row = 0;
                 row < data.rows;
                 ++row) {

                const double difference =
                    data(row, column) - mean;

                variance +=
                    difference * difference;
            }

            variance /=
                static_cast<double>(data.rows);

            variances_[column] = variance;

            if (variance > threshold_) {
                selected_columns_.push_back(column);
            }
        }

        if (selected_columns_.empty()) {
            throw std::invalid_argument(
                "VarianceThreshold: no features meet the variance threshold"
            );
        }

        fitted_ = true;
    }

    Matrix transform(
        const Matrix& data
    ) const {
        if (!fitted_) {
            throw std::logic_error(
                "VarianceThreshold::transform: "
                "selector has not been fitted"
            );
        }

        validate_input(data);

        if (data.cols != variances_.size()) {
            throw std::invalid_argument(
                "VarianceThreshold::transform: "
                "input column count does not match fitted data"
            );
        }

        Matrix result(
            data.rows,
            selected_columns_.size()
        );

        for (size_t row = 0;
             row < data.rows;
             ++row) {

            for (size_t output_column = 0;
                 output_column < selected_columns_.size();
                 ++output_column) {

                const size_t input_column =
                    selected_columns_[output_column];

                result(row, output_column) =
                    data(row, input_column);
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

    const std::vector<double>& variances() const {
        if (!fitted_) {
            throw std::logic_error(
                "VarianceThreshold::variances: "
                "selector has not been fitted"
            );
        }

        return variances_;
    }

private:
    static void validate_input(
        const Matrix& data
    ) {
        if (data.rows == 0 || data.cols == 0) {
            throw std::invalid_argument(
                "VarianceThreshold: input data must not be empty"
            );
        }
    }

private:
    double threshold_;
    std::vector<double> variances_;
    std::vector<size_t> selected_columns_;
    bool fitted_ = false;
};

} // namespace ml