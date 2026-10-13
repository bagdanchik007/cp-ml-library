
#pragma once

#include "ml/core/matrix/matrix.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace ml {

struct PermutationImportanceResult {
    double baseline_score = 0.0;
    std::vector<double> importances;
    std::vector<double> standard_deviations;
};

/**
 * @brief Measures feature importance by randomly permuting feature columns.
 *
 * The estimator must already be fitted. Higher scorer values must indicate
 * better model performance.
 */
template <typename EstimatorType, typename Scorer>
PermutationImportanceResult permutation_importance(
    const EstimatorType& estimator,
    const Matrix& features,
    const std::vector<double>& targets,
    Scorer scorer,
    std::size_t n_repeats = 5,
    unsigned seed = 42
) {
    if (features.rows == 0 || features.cols == 0) {
        throw std::invalid_argument(
            "permutation_importance: features must not be empty"
        );
    }

    if (targets.size() != features.rows) {
        throw std::invalid_argument(
            "permutation_importance: target count must match feature rows"
        );
    }

    if (n_repeats == 0) {
        throw std::invalid_argument(
            "permutation_importance: n_repeats must be at least 1"
        );
    }

    const auto predict_values = [&estimator, &targets](
        const Matrix& input
    ) {
        const Matrix predictions = estimator.predict(input);

        if (predictions.rows != targets.size() ||
            predictions.cols != 1) {
            throw std::invalid_argument(
                "permutation_importance: predict must return one value per sample"
            );
        }

        return predictions.column(0);
    };

    const auto baseline_predictions = predict_values(features);

    PermutationImportanceResult result;
    result.baseline_score = scorer(targets, baseline_predictions);
    result.importances.resize(features.cols, 0.0);
    result.standard_deviations.resize(features.cols, 0.0);

    std::mt19937 generator(seed);

    for (std::size_t feature = 0; feature < features.cols; ++feature) {
        std::vector<double> decreases;
        decreases.reserve(n_repeats);

        const std::vector<double> original_column =
            features.column(feature);

        for (std::size_t repeat = 0; repeat < n_repeats; ++repeat) {
            Matrix permuted_features = features;
            std::vector<double> shuffled_column = original_column;

            std::shuffle(
                shuffled_column.begin(),
                shuffled_column.end(),
                generator
            );

            permuted_features.set_column(feature, shuffled_column);

            const auto predictions = predict_values(permuted_features);
            const double permuted_score = scorer(targets, predictions);

            decreases.push_back(
                result.baseline_score - permuted_score
            );
        }

        const double mean = std::accumulate(
            decreases.begin(),
            decreases.end(),
            0.0
        ) / static_cast<double>(n_repeats);

        result.importances[feature] = mean;

        double squared_deviation = 0.0;

        for (double decrease : decreases) {
            const double difference = decrease - mean;
            squared_deviation += difference * difference;
        }

        result.standard_deviations[feature] = std::sqrt(
            squared_deviation / static_cast<double>(n_repeats)
        );
    }

    return result;
}

} // namespace ml
