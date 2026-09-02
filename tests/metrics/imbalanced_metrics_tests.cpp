#include "ml/metrics/model_evaluation.hpp"

#include <cassert>
#include <cmath>

int main() {
    // Imbalanced: 8 negatives, 2 positives. Model predicts all negative.
    const std::vector<int> actual{0, 0, 0, 0, 0, 0, 0, 0, 1, 1};
    const std::vector<int> all_negative{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    // Plain accuracy would be 0.8, but balanced accuracy sees the failure on class 1.
    assert(std::abs(ml::balanced_accuracy_score(actual, all_negative) - 0.5) < 1e-12);
    assert(ml::balanced_accuracy_score(actual, actual) == 1.0);

    // Matthews correlation coefficient: perfect predictions score 1.0.
    assert(std::abs(ml::matthews_corrcoef(actual, actual) - 1.0) < 1e-12);

    // Predicting the opposite label everywhere scores -1.0.
    const std::vector<int> inverted{1, 1, 1, 1, 1, 1, 1, 1, 0, 0};
    assert(std::abs(ml::matthews_corrcoef(actual, inverted) - (-1.0)) < 1e-12);

    // A model that always predicts negative has zero correlation on this data.
    assert(std::abs(ml::matthews_corrcoef(actual, all_negative) - 0.0) < 1e-12);
}
