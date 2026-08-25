#include "ml/feature_selection/f_regression.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace ml {

namespace {

double mean(const Matrix& matrix)
{
    double sum = 0.0;

    for (std::size_t row = 0; row < matrix.rows; ++row) {
        sum += matrix(row, 0);
    }

    return sum / static_cast<double>(matrix.rows);
}

double pearson_correlation(
    const Matrix& X,
    const Matrix& y,
    std::size_t feature)
{
    double x_mean = 0.0;

    for (std::size_t row = 0; row < X.rows; ++row) {
        x_mean += X(row, feature);
    }

    x_mean /= static_cast<double>(X.rows);

    const double y_mean = mean(y);

    double numerator = 0.0;
    double x_variance = 0.0;
    double y_variance = 0.0;

    for (std::size_t row = 0; row < X.rows; ++row) {
        const double x_delta = X(row, feature) - x_mean;
        const double y_delta = y(row, 0) - y_mean;

        numerator += x_delta * y_delta;
        x_variance += x_delta * x_delta;
        y_variance += y_delta * y_delta;
    }

    if (x_variance == 0.0 || y_variance == 0.0) {
        return 0.0;
    }

    return numerator /
           std::sqrt(x_variance * y_variance);
}

double f_score_from_correlation(
    double r,
    std::size_t samples)
{
    const double r_squared = r * r;

    if (r_squared >= 1.0) {
        return std::numeric_limits<double>::infinity();
    }

    const double degrees_of_freedom =
        static_cast<double>(samples - 2);

    return (r_squared / (1.0 - r_squared)) *
           degrees_of_freedom;
}

double approximate_p_value(
    double f_score,
    std::size_t samples)
{
    if (std::isinf(f_score)) {
        return 0.0;
    }

    if (f_score <= 0.0) {
        return 1.0;
    }

    const double degrees_of_freedom =
        static_cast<double>(samples - 2);

    const double t = std::sqrt(f_score);

    const double correction =
        1.0 + (t * t) / degrees_of_freedom;

    return std::pow(
        correction,
        -0.5 * degrees_of_freedom
    );
}

} // namespace

FRegressionResult f_regression(
    const Matrix& X,
    const Matrix& y)
{
    if (X.rows == 0 || X.cols == 0) {
        throw std::invalid_argument(
            "f_regression: X must not be empty"
        );
    }

    if (y.rows == 0 || y.cols == 0) {
        throw std::invalid_argument(
            "f_regression: y must not be empty"
        );
    }

    if (y.cols != 1) {
        throw std::invalid_argument(
            "f_regression: y must contain exactly one column"
        );
    }

    if (X.rows != y.rows) {
        throw std::invalid_argument(
            "f_regression: X and y must have the same number of rows"
        );
    }

    if (X.rows < 3) {
        throw std::invalid_argument(
            "f_regression: at least three samples are required"
        );
    }

    FRegressionResult result;

    result.scores.resize(X.cols);
    result.p_values.resize(X.cols);

    for (std::size_t feature = 0;
         feature < X.cols;
         ++feature)
    {
        const double correlation =
            pearson_correlation(
                X,
                y,
                feature
            );

        const double score =
            f_score_from_correlation(
                correlation,
                X.rows
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