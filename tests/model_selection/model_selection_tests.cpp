#include "ml/algorithms/linear_regression.hpp"
#include "ml/metrics/model_evaluation.hpp"
#include "ml/model_selection/cross_validation.hpp"
#include "ml/model_selection/grid_search.hpp"

#include <cassert>
#include <cmath>

int main() {
    ml::KFold folds(3, false);
    const auto partitions = folds.split(6);
    assert(partitions.size() == 3 && partitions[0].test.size() == 2 && partitions[0].train.size() == 4);

    const ml::Matrix x{{0.0}, {1.0}, {2.0}, {3.0}, {4.0}, {5.0}};
    const std::vector<double> y{1.0, 3.0, 5.0, 7.0, 9.0, 11.0};
    const auto scorer = [](const std::vector<double>& actual, const std::vector<double>& predicted) { return -ml::mean_squared_error(actual, predicted); };
    const auto scores = ml::cross_validate([] { return ml::LinearRegression{}; }, x, y, folds, scorer);
    assert(scores.size() == 3);
    auto factory = [](double rate) { (void)rate; return ml::LinearRegression{}; };
    ml::GridSearch<double, decltype(factory), decltype(scorer)> search({0.01, 0.1}, factory, folds, scorer);
    search.fit(x, y);
    assert(search.best_parameter() == 0.01 || search.best_parameter() == 0.1);
}
