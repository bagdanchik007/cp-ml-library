#include "ml/feature_selection/chi2.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>
#include <algorithm>

namespace ml {

namespace {

double chi_square_for_feature(
    const Matrix& X,
    const Matrix& y,
    std::size_t feature)
{
    std::vector<double> feature_values;
    std::vector<double> target_values;

    for (std::size_t row = 0; row < X.rows; ++row) {
        feature_values.push_back(X(row, feature));
        target_values.push_back(y(row, 0));
    }

    std::sort(
        feature_values.begin(),
        feature_values.end()
    );

    feature_values.erase(
        std::unique(
            feature_values.begin(),
            feature_values.end()
        ),
        feature_values.end()
    );

    std::sort(
        target_values.begin(),
        target_values.end()
    );

    target_values.erase(
        std::unique(
            target_values.begin(),
            target_values.end()
        ),
        target_values.end()
    );

    if (feature_values.empty() || target_values.empty()) {
        return 0.0;
    }

    const std::size_t feature_count =
        feature_values.size();

    const std::size_t target_count =
        target_values.size();

    std::vector<std::vector<double>> observed(
        feature_count,
        std::vector<double>(target_count, 0.0)
    );

    for (std::size_t row = 0; row < X.rows; ++row) {
        const auto feature_it =
            std::lower_bound(
                feature_values.begin(),
                feature_values.end(),
                X(row, feature)
            );

        const auto target_it =
            std::lower_bound(
                target_values.begin(),
                target_values.end(),
                y(row, 0)
            );

        const std::size_t feature_index =
            static_cast<std::size_t>(
                feature_it - feature_values.begin()
            );

        const std::size_t target_index =
            static_cast<std::size_t>(
                target_it - target_values.begin()
            );

        observed[feature_index][target_index] += 1.0;
    }

    std::vector<double> row_totals(feature_count, 0.0);
    std::vector<double> column_totals(target_count, 0.0);

    double total = 0.0;

    for (std::size_t row = 0; row < feature_count; ++row) {
        for (std::size_t column = 0;
             column < target_count;
             ++column) {

            const double value =
                observed[row][column];

            row_totals[row] += value;
            column_totals[column] += value;
            total += value;
        }
    }

    if (total == 0.0) {
        return 0.0;
    }

    double statistic = 0.0;

    for (std::size_t row = 0; row < feature_count; ++row) {
        for (std::size_t column = 0;
             column < target_count;
             ++column) {

            const double expected =
                (row_totals[row] * column_totals[column])
                / total;

            if (expected == 0.0) {
                continue;
            }

            const double difference =
                observed[row][column] - expected;

            statistic +=
                (difference * difference) / expected;
        }
    }

    return statistic;
}

double approximate_p_value(
    double statistic,
    std::size_t degrees_of_freedom)
{
    if (statistic <= 0.0) {
        return 1.0;
    }

    if (std::isinf(statistic)) {
        return 0.0;
    }

    if (degrees_of_freedom == 0) {
        return 1.0;
    }

    return std::exp(
        -0.5 * statistic
    );
}

} // namespace

Chi2Result chi2(
    const Matrix& X,
    const Matrix& y)
{
    if (X.rows == 0 || X.cols == 0) {
        throw std::invalid_argument(
            "chi2: X must not be empty"
        );
    }

    if (y.rows == 0 || y.cols == 0) {
        throw std::invalid_argument(
            "chi2: y must not be empty"
        );
    }

    if (y.cols != 1) {
        throw std::invalid_argument(
            "chi2: y must contain exactly one column"
        );
    }

    if (X.rows != y.rows) {
        throw std::invalid_argument(
            "chi2: X and y must have the same number of rows"
        );
    }

    Chi2Result result;

    result.scores.resize(X.cols);
    result.p_values.resize(X.cols);

    for (std::size_t feature = 0;
         feature < X.cols;
         ++feature) {

        const double score =
            chi_square_for_feature(
                X,
                y,
                feature
            );

        result.scores[feature] = score;

        result.p_values[feature] =
            approximate_p_value(
                score,
                X.rows
            );
    }

    return result;
}

} // namespace ml
