#include "ml/feature_selection/select_from_model.hpp"

#include <stdexcept>

namespace ml {

SelectFromModel::SelectFromModel(
    double threshold
)
    : threshold_(threshold)
{
}

void SelectFromModel::fit(
    const std::vector<double>& scores
)
{
    if (scores.empty()) {
        throw std::invalid_argument(
            "SelectFromModel: scores must not be empty"
        );
    }

    selected_features_.clear();

    for (std::size_t feature = 0;
         feature < scores.size();
         ++feature) {

        if (scores[feature] >= threshold_) {
            selected_features_.push_back(
                feature
            );
        }
    }

    fitted_ = true;
}

Matrix SelectFromModel::transform(
    const Matrix& data
) const
{
        throw std::logic_error(
            "SelectFromModel::transform: "
            "selector has not been fitted"
        );
    }

    if (data.rows == 0 || data.cols == 0) {
        throw std::invalid_argument(
            "SelectFromModel::transform: "
            "input data must not be empty"
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

            const std::size_t feature =
                selected_features_[column];

            result(row, column) =
                data(row, feature);
        }
    }

    return result;
}

Matrix SelectFromModel::fit_transform(
    const Matrix& data,
    const std::vector<double>& scores
)
{
    fit(scores);

    if (data.cols != scores.size()) {
        throw std::invalid_argument(
            "SelectFromModel::fit_transform: "
            "number of scores must match input columns"
        );
    }

    return transform(data);
}

const std::vector<std::size_t>&
SelectFromModel::selected_features() const
{
        throw std::logic_error(
            "SelectFromModel::selected_features: "
            "selector has not been fitted"
        );
    }

    return selected_features_;
}

} // namespace ml
