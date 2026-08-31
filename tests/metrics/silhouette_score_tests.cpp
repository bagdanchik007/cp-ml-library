#include "ml/metrics/model_evaluation.hpp"

#include <cassert>

int main() {
    // Two well-separated, dense clusters should score close to 1.
    const ml::Matrix X{
        {0.0, 0.0}, {0.1, 0.1}, {0.0, 0.1},
        {10.0, 10.0}, {10.1, 9.9}, {9.9, 10.1}
    };
    const std::vector<int> labels{0, 0, 0, 1, 1, 1};

    const double score = ml::silhouette_score(X, labels);
    assert(score > 0.9);
    assert(score <= 1.0);
}
