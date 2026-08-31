#include "ml/metrics/model_evaluation.hpp"

#include <cassert>
#include <cmath>

int main() {
    const std::vector<double> actual{1.0, 2.0, 3.0};
    const std::vector<double> predicted{1.0, 1.0, 5.0};
    assert(std::abs(ml::mean_squared_error(actual, predicted) - 5.0 / 3.0) < 1e-12);
    assert(std::abs(ml::mean_absolute_error(actual, predicted) - 1.0) < 1e-12);
    assert(ml::r2_score(actual, actual) == 1.0);
    assert(ml::accuracy_score({1, 0, 1, 1}, {1, 1, 1, 0}) == 0.5);
    assert(std::abs(ml::root_mean_squared_error(actual, predicted) - std::sqrt(5.0 / 3.0)) < 1e-12);
}
