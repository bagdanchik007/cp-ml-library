#pragma once

#include "ml/core/matrix/matrix.hpp"
#include "ml/model_selection/kfold.hpp"

#include <functional>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace ml
{

inline Matrix select_rows(
    const Matrix& matrix,
    const std::vector<size_t>& indices
)
{
    Matrix result(indices.size(), matrix.cols);

    for (size_t row = 0; row < indices.size(); ++row)
    {
        if (indices[row] >= matrix.rows)
        {
            throw std::out_of_range(
                "select_rows: index out of range"
            );
        }

        result.set_row(
            row,
            matrix.row(indices[row])
        );
    }

    return result;
}


inline std::vector<double> select_rows(
    const std::vector<double>& values,
    const std::vector<size_t>& indices
)
{
    std::vector<double> result;
    result.reserve(indices.size());

    for (size_t index : indices)
    {
        if (index >= values.size())
        {
            throw std::out_of_range(
                "select_rows: index out of range"
            );
        }

        result.push_back(values[index]);
    }

    return result;
}


inline std::vector<double> prediction_to_vector(
    const std::vector<double>& predictions
)
{
    return predictions;
}


inline std::vector<double> prediction_to_vector(
    const Matrix& predictions
)
{
    if (predictions.cols != 1)
    {
        throw std::invalid_argument(
            "prediction_to_vector: "
            "prediction matrix must have exactly one column"
        );
    }

    std::vector<double> result;
    result.reserve(predictions.rows);

    for (size_t row = 0; row < predictions.rows; ++row)
    {
        result.push_back(
            predictions(row, 0)
        );
    }

    return result;
}


template <typename EstimatorFactory, typename Scorer>
std::vector<double> cross_validate(
    EstimatorFactory make_estimator,
    const Matrix& features,
    const std::vector<double>& targets,
    const KFold& splitter,
    Scorer scorer
)
{
    if (features.rows != targets.size())
    {
        throw std::invalid_argument(
            "cross_validate: "
            "feature and target row counts differ"
        );
    }

    std::vector<double> scores;

    for (const auto& fold : splitter.split(features.rows))
    {
        auto estimator = make_estimator();

        const Matrix x_train =
            select_rows(
                features,
                fold.train
            );

        const Matrix x_test =
            select_rows(
                features,
                fold.test
            );

        const auto y_train =
            select_rows(
                targets,
                fold.train
            );

        const auto y_test =
            select_rows(
                targets,
                fold.test
            );

        estimator.fit(
            x_train,
            y_train
        );

        const auto raw_predictions =
            estimator.predict(x_test);

        const auto predictions =
            prediction_to_vector(
                raw_predictions
            );

        scores.push_back(
            scorer(
                y_test,
                predictions
            )
        );
    }

    return scores;
}


inline double mean_score(
    const std::vector<double>& scores
)
{
    if (scores.empty())
    {
        throw std::invalid_argument(
            "mean_score: scores must not be empty"
        );
    }

    return std::accumulate(
        scores.begin(),
        scores.end(),
        0.0
    ) / static_cast<double>(scores.size());
}

} // namespace ml