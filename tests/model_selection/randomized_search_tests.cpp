#include "ml/algorithms/ridge_regression.hpp"
#include "ml/metrics/model_evaluation.hpp"
#include "ml/model_selection/randomized_search.hpp"

#include <cassert>

int main() {
    // y = 2*x + 1, noise-free.
    const ml::Matrix X{{0.0}, {1.0}, {2.0}, {3.0}, {4.0}, {5.0}, {6.0}, {7.0}};
    const std::vector<double> y{1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0};

    const std::vector<double> alpha_candidates{100.0, 10.0, 1.0, 0.1, 0.01};

    auto factory = [](double alpha) { return ml::RidgeRegression(alpha); };
    auto scorer = [](const std::vector<double>& actual, const std::vector<double>& predicted) {
        return -ml::mean_squared_error(actual, predicted); // higher (less negative) is better
    };

    ml::RandomizedSearch search(alpha_candidates, factory, ml::KFold(4), scorer, /*n_iterations=*/5, /*seed=*/7);
    search.fit(X, y);

    // With noise-free data, small alpha (less regularization) should win.
    assert(search.best_parameter() <= 1.0);
    assert(search.best_score() > -1.0);

    // Trying more iterations than candidates should not throw or loop forever.
    ml::RandomizedSearch exhaustive_search(alpha_candidates, factory, ml::KFold(4), scorer, /*n_iterations=*/100, 7);
    exhaustive_search.fit(X, y);
    assert(exhaustive_search.best_parameter() <= 1.0);
}
