#include "ml/feature_selection/select_fpr.hpp"

#include "ml/feature_selection/f_regression.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace ml {

SelectFpr::SelectFpr(double alpha)
    : alpha_(alpha)
{
    if (!std::isfinite(alpha_)) {
        throw std::invalid_argument(
            "SelectFpr: alpha must be finite"
        );
    }

    if (alpha_ <= 0.0 || alpha_ > 1.0) {
        throw std::invalid_argument(
            "SelectFpr: alpha must be in the range (0, 1]"
        );
    }
}

void SelectFpr::fit(
    const Matrix& X,
    const Matrix& y)
{
    if (X.rows == 0 || X.cols == 0) {
        throw std::invalid_argument(
            "SelectFpr: input data must not be empty"
        );
    }

    if (y.rows == 0 || y.cols == 0) {
        throw std::invalid_argument(
            "SelectFpr: target data must not be empty"
        );
    }

    if (X.rows != y.rows) {
        throw std::invalid_argument(
            "SelectFpr: X and y must have the same number of rows"
        );
    }

    if (y.cols != 1) {
        throw std::invalid_argument(
            "SelectFpr: y must contain exactly one target column"
        );
    }

    selected_features_.clear();

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

        if (result.p_values[0] < alpha_) {
            selected_features_.push_back(feature);
        }
    }

    fitted_ = true;
}

Matrix SelectFpr::transform(
    const Matrix& X) const
{
    if (!fitted_) {
        throw std::logic_error(
            "SelectFpr::transform: "
            "selector has not been fitted"
        );
    }

    if (X.rows == 0 || X.cols == 0) {
        throw std::invalid_argument(
            "SelectFpr::transform: input data must not be empty"
        );
    }

    for (const std::size_t feature : selected_features_) {
        if (feature >= X.cols) {
            throw std::invalid_argument(
                "SelectFpr::transform: "
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

Matrix SelectFpr::fit_transform(
    const Matrix& X,
    const Matrix& y)
{
    fit(X, y);
    return transform(X);
}

const std::vector<std::size_t>&
SelectFpr::selected_features() const noexcept
{
    return selected_features_;
}

} // namespace ml