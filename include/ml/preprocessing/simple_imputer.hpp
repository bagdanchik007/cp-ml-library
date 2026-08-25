#pragma once

#include "ml/core/matrix/matrix.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ml {

enum class ImputationStrategy {
    Mean,
    Median,
    Constant
};

class SimpleImputer {
public:
    explicit SimpleImputer(
        ImputationStrategy strategy =
            ImputationStrategy::Mean,
        double fill_value = 0.0
    )
        : strategy_(strategy),
          fill_value_(fill_value) {
    }

    void fit(
        const Matrix& data
    ) {
        if (data.rows == 0 || data.cols == 0) {
            throw std::invalid_argument(
                "SimpleImputer::fit: input data must not be empty"
            );
        }

        statistics_.clear();
        statistics_.resize(data.cols);

        for (size_t column = 0;
             column < data.cols;
             ++column) {

            if (strategy_ ==
                ImputationStrategy::Constant) {

                statistics_[column] =
                    fill_value_;

                continue;
            }

            std::vector<double> values;

            for (size_t row = 0;
                 row < data.rows;
                 ++row) {

                const double value =
                    data(row, column);

                if (!std::isnan(value)) {
                    values.push_back(value);
                }
            }

            if (values.empty()) {
                throw std::invalid_argument(
                    "SimpleImputer::fit: column contains only missing values"
                );
            }

            if (strategy_ ==
                ImputationStrategy::Mean) {

                double sum = 0.0;

                for (const double value : values) {
                    sum += value;
                }

                statistics_[column] =
                    sum /
                    static_cast<double>(
                        values.size()
                    );
            } else if (strategy_ ==
                       ImputationStrategy::Median) {

                std::sort(
                    values.begin(),
                    values.end()
                );

                const size_t middle =
                    values.size() / 2;

                if (values.size() % 2 == 0) {
                    statistics_[column] =
                        (
                            values[middle - 1] +
                            values[middle]
                        ) / 2.0;
                } else {
                    statistics_[column] =
                        values[middle];
                }
            }
        }

        fitted_ = true;
    }

    Matrix transform(
        const Matrix& data
    ) const {
        if (!fitted_) {
            throw std::logic_error(
                "SimpleImputer::transform: "
                "imputer has not been fitted"
            );
        }

        if (data.rows == 0 || data.cols == 0) {
            throw std::invalid_argument(
                "SimpleImputer::transform: "
                "input data must not be empty"
            );
        }

        if (data.cols != statistics_.size()) {
            throw std::invalid_argument(
                "SimpleImputer::transform: "
                "input column count does not match fitted data"
            );
        }

        Matrix result = data;

        for (size_t row = 0;
             row < data.rows;
             ++row) {

            for (size_t column = 0;
                 column < data.cols;
                 ++column) {

                if (std::isnan(
                        result(row, column)
                    )) {

                    result(row, column) =
                        statistics_[column];
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
    ImputationStrategy strategy_;
    double fill_value_;
    std::vector<double> statistics_;
    bool fitted_ = false;
};

} // namespace ml