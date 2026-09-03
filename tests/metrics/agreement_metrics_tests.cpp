#include "ml/metrics/model_evaluation.hpp"

#include <cassert>
#include <cmath>

int main() {
    const std::vector<int> actual{1, 0, 1, 1, 0, 1, 0, 0};
    const std::vector<int> predicted{1, 0, 0, 1, 0, 1, 1, 0};

    assert(std::abs(ml::cohen_kappa_score(actual, actual) - 1.0) < 1e-12);
    assert(ml::cohen_kappa_score(actual, predicted) < 1.0);
    assert(ml::cohen_kappa_score(actual, predicted) > 0.0);

    // intersection = 3 (indices 0,3,5), union = 5 (0,2,3,5,6)
    assert(std::abs(ml::jaccard_score(actual, predicted) - 3.0 / 5.0) < 1e-12);
    assert(ml::jaccard_score(actual, actual) == 1.0);

    // 2 mismatches (index 2 and 6) out of 8 samples
    assert(std::abs(ml::hamming_loss(actual, predicted) - 2.0 / 8.0) < 1e-12);
    assert(ml::hamming_loss(actual, actual) == 0.0);
}
