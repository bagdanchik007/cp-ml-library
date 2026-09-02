#include "ml/metrics/model_evaluation.hpp"

#include <cassert>
#include <cmath>

int main() {
    const std::vector<double> actual{100.0, 200.0, 300.0};
    const std::vector<double> predicted{110.0, 190.0, 330.0};

    // Errors: 10%, 5%, 10% -> mean 8.333...%
    assert(std::abs(ml::mean_absolute_percentage_error(actual, predicted) - (0.25 / 3.0)) < 1e-9);

    assert(std::abs(ml::explained_variance_score(actual, actual) - 1.0) < 1e-12);

    // A constant offset does not hurt explained variance (unlike r2_score).
    std::vector<double> shifted(actual.size());
    for (size_t i = 0; i < actual.size(); ++i) shifted[i] = actual[i] + 5.0;
    assert(std::abs(ml::explained_variance_score(actual, shifted) - 1.0) < 1e-9);
}
