#include "ml/feature_selection/select_fdr.hpp"

#include "ml/feature_selection/f_regression.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ml {

SelectFdr::SelectFdr(double alpha)
    : alpha_(alpha)
{
    if (!std::isfinite(alpha_)) {
        throw std::invalid_argument(
            "SelectFdr: alpha must be finite"
        );
    }

    if (alpha_ <= 0.0 || alpha_ > 1.0) {
        throw std::invalid_argument(
            "SelectFdr: alpha must be in the range (0, 1]"
        );
    }
}

void SelectFdr::fit(
    const Matrix& X,
    const Matrix& y)
{
    if (X.rows == 0 || X.cols == 0) {
        throw std::invalid_argument(
            "SelectFdr: input data must not be empty"
        );
    }

    if (y.rows == 0 || y.cols == 0) {
        throw std::invalid_argument(
            "SelectFdr: target data must not be empty"
        );
    }

    if (X.rows != y.rows) {
        throw std::invalid_argument(
            "SelectFdr: X and y must have the same number of rows"
        );
    }

    if (y.cols != 1) {
        throw std::invalid_argument(
            "SelectFdr: y must contain exactly one target column"
        );
    }

    selected_features_.clear();

    std::vector<double> p_values;
    p_values.reserve(X.cols);

    for (std::size_t feature = 0;
         feature < X.cols;
         ++feature) {

        Matrix feature_data(X.rows, 1);

        for (std::size_t row = 0;
             row < X.rows;
             ++row) {

            feature_data(row, 0) =
                X(row, feature);
        }

        const auto result =
            f_regression(feature_data, y);

        p_values.push_back(result.p_values[0]);
    }

    std::vector<std::size_t> indices(X.cols);

    for (std::size_t feature = 0;
         feature < X.cols;
         ++feature) {

        indices[feature] = feature;
    }

    std::stable_sort(
        indices.begin(),
        indices.end(),
        [&p_values](std::size_t lhs, std::size_t rhs) {
            return p_values[lhs] < p_values[rhs];
        }
    );

    const std::size_t number_of_features = X.cols;

    std::size_t largest_rank = 0;

    for (std::size_t rank = 1;
         rank <= number_of_features;
         ++rank) {

        const std::size_t feature =
            indices[rank - 1];

        const double threshold =
            (static_cast<double>(rank) /
             static_cast<double>(number_of_features)) *
            alpha_;

        if (p_values[feature] <= threshold) {
            largest_rank = rank;
        }
    }

    if (largest_rank > 0) {
        for (std::size_t rank = 0;
             rank < largest_rank;
             ++rank) {

            selected_features_.push_back(
                indices[rank]
            );
        }

        std::sort(
            selected_features_.begin(),
            selected_features_.end()
        );
    }

    fitted_ = true;
}

Matrix SelectFdr::transform(
    const Matrix& X) const
{
    if (!fitted_) {
        throw std::logic_error(
            "SelectFdr::transform: "
            "selector has not been fitted"
        );
    }

    if (X.rows == 0 || X.cols == 0) {
        throw std::invalid_argument(
            "SelectFdr::transform: input data must not be empty"
        );
    }

    for (const std::size_t feature : selected_features_) {
        if (feature >= X.cols) {
            throw std::invalid_argument(
                "SelectFdr::transform: "
                "input column count does not match fitted data"
            );
        }
    }

    Matrix result(
        X.rows,
        selected_features_.size()
    );

    for (std::size_t row = 0;
         row < X.rows;
         ++row) {

        for (std::size_t output_column = 0;
             output_column < selected_features_.size();
             ++output_column) {

            const std::size_t input_column =
                selected_features_[output_column];

            result(row, output_column) =
                X(row, input_column);
        }
    }

    return result;
}

Matrix SelectFdr::fit_transform(
    const Matrix& X,
    const Matrix& y)
{
    fit(X, y);
    return transform(X);
}

const std::vector<std::size_t>&
SelectFdr::selected_features() const noexcept
{
    return selected_features_;
}

} // namespace ml
