#include "ml/feature_selection/generic_univariate_selector.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ml {

GenericUnivariateSelect::GenericUnivariateSelect(
    UnivariateScoreFunction score_function,
    SelectionMode mode,
    double param)
    : score_function_(std::move(score_function)),
      mode_(mode),
      param_(param)
{
    if (!score_function_) {
        throw std::invalid_argument(
            "GenericUnivariateSelect: score function must be valid"
        );
    }

    if (!std::isfinite(param_)) {
        throw std::invalid_argument(
            "GenericUnivariateSelect: parameter must be finite"
        );
    }

    if (param_ < 0.0) {
        throw std::invalid_argument(
            "GenericUnivariateSelect: parameter must not be negative"
        );
    }

    if (mode_ == SelectionMode::Percentile &&
        param_ > 100.0) {
        throw std::invalid_argument(
            "GenericUnivariateSelect: percentile must be between 0 and 100"
        );
    }
}

void GenericUnivariateSelect::validate_input(
    const Matrix& data) const
{
    if (data.rows == 0 || data.cols == 0) {
        throw std::invalid_argument(
            "GenericUnivariateSelect: input data must not be empty"
        );
    }
}

void GenericUnivariateSelect::fit(
    const Matrix& data)
{
    validate_input(data);

    scores_.clear();
    selected_features_.clear();

    scores_.reserve(data.cols);

    for (std::size_t feature = 0;
         feature < data.cols;
         ++feature) {
        scores_.push_back(
            score_function_(data, feature)
        );
    }

    std::vector<std::size_t> indices(data.cols);

    for (std::size_t feature = 0;
         feature < data.cols;
         ++feature) {
        indices[feature] = feature;
    }

    std::sort(
        indices.begin(),
        indices.end(),
        [this](std::size_t lhs, std::size_t rhs) {
            return scores_[lhs] > scores_[rhs];
        }
    );

    std::size_t count = 0;

    switch (mode_) {
    case SelectionMode::KBest:
        count = static_cast<std::size_t>(param_);
        break;

    case SelectionMode::Percentile:
        count = static_cast<std::size_t>(
            std::ceil(
                static_cast<double>(data.cols) *
                param_ /
                100.0
            )
        );
        break;

    case SelectionMode::Fpr:
    case SelectionMode::Fdr:
    case SelectionMode::Fwe:
        /*
         * These selection modes require p-values and additional
         * statistical information. The generic selector currently
         * uses param_ as a fraction of features to retain.
         */
        count = static_cast<std::size_t>(
            std::ceil(
                static_cast<double>(data.cols) *
                param_
            )
        );
        break;
    }

    count = std::min(count, data.cols);

    selected_features_.reserve(count);

    for (std::size_t i = 0; i < count; ++i) {
        selected_features_.push_back(indices[i]);
    }

    /*
     * Keep feature indices in their original column order.
     */
    std::sort(
        selected_features_.begin(),
        selected_features_.end()
    );

    fitted_ = true;
}

Matrix GenericUnivariateSelect::transform(
    const Matrix& data) const
{
    if (!fitted_) {
        throw std::logic_error(
            "GenericUnivariateSelect: selector has not been fitted"
        );
    }

    validate_input(data);

    if (data.cols != scores_.size()) {
        throw std::invalid_argument(
            "GenericUnivariateSelect: input feature count does not match fitted data"
        );
    }

    Matrix result(
        data.rows,
        selected_features_.size()
    );

    for (std::size_t row = 0;
         row < data.rows;
         ++row) {

        for (std::size_t column = 0;
             column < selected_features_.size();
             ++column) {

            result(
                row,
                column
            ) = data(
                row,
                selected_features_[column]
            );
        }
    }

    return result;
}

Matrix GenericUnivariateSelect::fit_transform(
    const Matrix& data)
{
    fit(data);
    return transform(data);
}

const std::vector<double>&
GenericUnivariateSelect::scores() const
{
    return scores_;
}

const std::vector<std::size_t>&
GenericUnivariateSelect::selected_features() const
{
    return selected_features_;
}

} // namespace ml
