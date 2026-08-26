#include "ml/feature_selection/select_percentile.hpp"

#include "ml/feature_selection/f_regression.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace ml {

SelectPercentile::SelectPercentile(double percentile)
    : percentile_(percentile)
{
    if (!std::isfinite(percentile_)) {
        throw std::invalid_argument(
            "SelectPercentile: percentile must be finite"
        );
    }

    if (percentile_ <= 0.0 || percentile_ > 100.0) {
        throw std::invalid_argument(
            "SelectPercentile: percentile must be in the range (0, 100]"
        );
    }
}

void SelectPercentile::fit(
    const Matrix& X,
    const Matrix& y)
{
    if (X.rows == 0 || X.cols == 0) {
        throw std::invalid_argument(
            "SelectPercentile: X must not be empty"
        );
    }

    if (y.rows == 0 || y.cols == 0) {
        throw std::invalid_argument(
            "SelectPercentile: y must not be empty"
        );
    }

    if (y.cols != 1) {
        throw std::invalid_argument(
            "SelectPercentile: y must contain exactly one column"
        );
    }

    if (X.rows != y.rows) {
        throw std::invalid_argument(
            "SelectPercentile: X and y must have the same number of rows"
        );
    }

    const FRegressionResult result =
        f_regression(X, y);

    scores_ = result.scores;
    selected_features_.clear();

    const std::size_t feature_count = X.cols;

    const std::size_t selected_count =
        static_cast<std::size_t>(
            std::ceil(
                (percentile_ / 100.0) *
                static_cast<double>(feature_count)
            )
        );

    std::vector<std::size_t> indices(feature_count);

    for (std::size_t feature = 0;
         feature < feature_count;
         ++feature)
    {
        indices[feature] = feature;
    }

    std::stable_sort(
        indices.begin(),
        indices.end(),
        [this](std::size_t lhs, std::size_t rhs) {
            return scores_[lhs] > scores_[rhs];
        }
    );

    selected_features_.assign(
        indices.begin(),
        indices.begin() + selected_count
    );

    std::sort(
        selected_features_.begin(),
        selected_features_.end()
    );

    fitted_ = true;
}

Matrix SelectPercentile::transform(
    const Matrix& X) const
{
    if (!fitted_) {
        throw std::logic_error(
            "SelectPercentile::transform: "
            "selector has not been fitted"
        );
    }

    if (X.rows == 0 || X.cols == 0) {
        throw std::invalid_argument(
            "SelectPercentile::transform: "
            "X must not be empty"
        );
    }

    if (X.cols != scores_.size()) {
        throw std::invalid_argument(
            "SelectPercentile::transform: "
            "input column count does not match fitted data"
        );
    }

    Matrix result(
        X.rows,
        selected_features_.size()
    );

    for (std::size_t row = 0;
         row < X.rows;
         ++row)
    {
        for (std::size_t output_column = 0;
             output_column < selected_features_.size();
             ++output_column)
        {
            const std::size_t input_column =
                selected_features_[output_column];

            result(row, output_column) =
                X(row, input_column);
        }
    }

    return result;
}

Matrix SelectPercentile::fit_transform(
    const Matrix& X,
    const Matrix& y)
{
    fit(X, y);

    return transform(X);
}

const std::vector<double>&
SelectPercentile::scores() const
{
    if (!fitted_) {
        throw std::logic_error(
            "SelectPercentile::scores: "
            "selector has not been fitted"
        );
    }

    return scores_;
}

const std::vector<std::size_t>&
SelectPercentile::selected_features() const
{
    if (!fitted_) {
        throw std::logic_error(
            "SelectPercentile::selected_features: "
            "selector has not been fitted"
        );
    }

    return selected_features_;
}

} // namespace ml