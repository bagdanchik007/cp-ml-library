#include "ml/feature_selection/mutual_info.hpp"

#include <cmath>
#include <cstddef>
#include <map>
#include <stdexcept>

namespace ml {

namespace {

void validate_input(
    const Matrix& X,
    const Matrix& y)
{
    if (X.rows == 0 || X.cols == 0) {
        throw std::invalid_argument(
            "mutual_info: X must not be empty"
        );
    }

    if (y.rows == 0 || y.cols == 0) {
        throw std::invalid_argument(
            "mutual_info: y must not be empty"
        );
    }

    if (y.cols != 1) {
        throw std::invalid_argument(
            "mutual_info: y must contain exactly one column"
        );
    }

    if (X.rows != y.rows) {
        throw std::invalid_argument(
            "mutual_info: X and y must have the same number of rows"
        );
    }
}

double calculate_mutual_information(
    const Matrix& X,
    const Matrix& y,
    std::size_t feature)
{
    std::map<double, std::size_t> feature_counts;
    std::map<double, std::size_t> target_counts;
    std::map<std::pair<double, double>, std::size_t>
        joint_counts;

    for (std::size_t row = 0;
         row < X.rows;
         ++row) {

        const double feature_value =
            X(row, feature);

        const double target_value =
            y(row, 0);

        ++feature_counts[feature_value];
        ++target_counts[target_value];

        ++joint_counts[
            {feature_value, target_value}
        ];
    }

    const double sample_count =
        static_cast<double>(X.rows);

    double mutual_information = 0.0;

    for (const auto& [values, count] : joint_counts) {

        const double feature_value =
            values.first;

        const double target_value =
            values.second;

        const double joint_probability =
            static_cast<double>(count) /
            sample_count;

        const double feature_probability =
            static_cast<double>(
                feature_counts[feature_value]
            ) / sample_count;

        const double target_probability =
            static_cast<double>(
                target_counts[target_value]
            ) / sample_count;

        mutual_information +=
            joint_probability *
            std::log(
                joint_probability /
                (
                    feature_probability *
                    target_probability
                )
            );
    }

    return mutual_information;
}

} // namespace

MutualInfoResult mutual_info(
    const Matrix& X,
    const Matrix& y)
{
    validate_input(X, y);

    MutualInfoResult result;

    result.scores.resize(X.cols);

    for (std::size_t feature = 0;
         feature < X.cols;
         ++feature) {

        result.scores[feature] =
            calculate_mutual_information(
                X,
                y,
                feature
            );
    }

    return result;
}

} // namespace ml
