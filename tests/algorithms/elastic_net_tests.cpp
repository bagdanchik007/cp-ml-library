#include "ml/algorithms/elastic_net.hpp"

#include <cassert>
#include <cmath>

int main() {
    // y = 3*x1 + 2, x2 is irrelevant noise.
    const ml::Matrix X{
        {0.0, 1.0}, {1.0, -1.0}, {2.0, 2.0}, {3.0, -2.0}, {4.0, 0.5}
    };
    const std::vector<double> y{2.0, 5.0, 8.0, 11.0, 14.0};

    ml::ElasticNet model(0.01, 0.5);
    assert(!model.is_fitted());
    model.fit(X, y, 5000);
    assert(model.is_fitted());

    assert(std::abs(model.weights()[0] - 3.0) < 0.3);
    assert(std::abs(model.weights()[1]) < 0.4);
    assert(std::abs(model.bias() - 2.0) < 0.3);

    // l1_ratio == 1.0 behaves like pure Lasso; l1_ratio == 0.0 like pure Ridge.
    ml::ElasticNet pure_lasso(0.01, 1.0);
    pure_lasso.fit(X, y, 5000);
    assert(std::abs(pure_lasso.weights()[0] - 3.0) < 0.3);

    bool threw = false;
    try {
        ml::ElasticNet invalid(1.0, 1.5);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}
