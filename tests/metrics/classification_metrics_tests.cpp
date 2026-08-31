#include "ml/metrics/model_evaluation.hpp"

#include <cassert>
#include <cmath>

int main() {
    const std::vector<int> actual{1, 0, 1, 1, 0, 1};
    const std::vector<int> predicted{1, 0, 0, 1, 1, 1};

    // true positives = 3, false positives = 1, false negatives = 1
    assert(std::abs(ml::precision_score(actual, predicted) - 0.75) < 1e-12);
    assert(std::abs(ml::recall_score(actual, predicted) - 0.75) < 1e-12);
    assert(std::abs(ml::f1_score(actual, predicted) - 0.75) < 1e-12);

    const auto confusion = ml::confusion_matrix(actual, predicted, 2);
    assert(confusion(0, 0) == 1.0);
    assert(confusion(0, 1) == 1.0);
    assert(confusion(1, 0) == 1.0);
    assert(confusion(1, 1) == 3.0);

    // Perfect predictions score 1.0 everywhere.
    assert(ml::precision_score(actual, actual) == 1.0);
    assert(ml::recall_score(actual, actual) == 1.0);
    assert(ml::f1_score(actual, actual) == 1.0);
}
