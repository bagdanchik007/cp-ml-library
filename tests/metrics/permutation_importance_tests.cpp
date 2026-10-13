
#include "ml/metrics/model_evaluation.hpp"
#include "ml/metrics/permutation_importance.hpp"

#include <cassert>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace {

struct FirstFeatureEstimator {
    ml::Matrix predict(const ml::Matrix& X) const {
        ml::Matrix predictions(X.rows, 1);

        for (std::size_t i = 0; i < X.rows; ++i) {
            predictions(i, 0) = X(i, 0);
        }

        return predictions;
    }
};

auto negative_mse_scorer() {
    return [](const std::vector<double>& actual,
              const std::vector<double>& predicted) {
        return -ml::mean_squared_error(actual, predicted);
    };
}

} // namespace

int main() {
    const ml::Matrix X{
        {0.0, 5.0},
        {1.0, 1.0},
        {2.0, 4.0},
        {3.0, 2.0},
        {4.0, 6.0},
        {5.0, 3.0}
    };

    const std::vector<double> y{0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
    const FirstFeatureEstimator estimator;
    const auto scorer = negative_mse_scorer();

    const auto result = ml::permutation_importance(
        estimator, X, y, scorer, 10, 42
    );

    assert(std::abs(result.baseline_score) < 1e-12);
    assert(result.importances.size() == 2);
    assert(result.standard_deviations.size() == 2);
    assert(result.importances[0] > 0.0);
    assert(std::abs(result.importances[1]) < 1e-12);
    assert(std::abs(result.standard_deviations[1]) < 1e-12);

    const auto repeated = ml::permutation_importance(
        estimator, X, y, scorer, 10, 42
    );

    assert(result.importances == repeated.importances);
    assert(result.standard_deviations == repeated.standard_deviations);

    bool invalid_targets_thrown = false;

    try {
        (void)ml::permutation_importance(
            estimator, X, std::vector<double>{1.0},
            scorer, 5, 42
        );
    } catch (const std::invalid_argument&) {
        invalid_targets_thrown = true;
    }

    assert(invalid_targets_thrown);

    bool invalid_repeats_thrown = false;

    try {
        (void)ml::permutation_importance(
            estimator, X, y, scorer, 0, 42
        );
    } catch (const std::invalid_argument&) {
        invalid_repeats_thrown = true;
    }

    assert(invalid_repeats_thrown);
}
