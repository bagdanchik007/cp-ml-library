#include "ml/feature_selection/chi2.hpp"

#include <cassert>
#include <cmath>
#include <stdexcept>

namespace {

void test_basic_chi2()
{
    const ml::Matrix X{
        {0.0, 1.0},
        {0.0, 2.0},
        {1.0, 1.0},
        {1.0, 2.0}
    };

    const ml::Matrix y{
        {0.0},
        {0.0},
        {1.0},
        {1.0}
    };

    const ml::Chi2Result result =
        ml::chi2(X, y);

    assert(result.scores.size() == 2);
    assert(result.p_values.size() == 2);

    assert(std::isfinite(result.scores[0]));
    assert(std::isfinite(result.scores[1]));

    assert(result.scores[0] >= 0.0);
    assert(result.scores[1] >= 0.0);

    assert(result.scores[0] > result.scores[1]);
}

void test_independent_feature()
{
    const ml::Matrix X{
        {0.0},
        {0.0},
        {1.0},
        {1.0}
    };

    const ml::Matrix y{
        {0.0},
        {1.0},
        {0.0},
        {1.0}
    };

    const ml::Chi2Result result =
        ml::chi2(X, y);

    assert(result.scores.size() == 1);
    assert(std::abs(result.scores[0]) < 1e-12);
    assert(result.p_values[0] == 1.0);
}

void test_empty_X()
{
    bool threw = false;

    try {
        ml::chi2(
            ml::Matrix{},
            ml::Matrix{}
        );
    }
    catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

void test_invalid_target_columns()
{
    const ml::Matrix X{
        {1.0},
        {2.0}
    };

    const ml::Matrix y{
        {0.0, 1.0},
        {1.0, 0.0}
    };

    bool threw = false;

    try {
        ml::chi2(X, y);
    }
    catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

void test_mismatched_rows()
{
    const ml::Matrix X{
        {1.0},
        {2.0},
        {3.0}
    };

    const ml::Matrix y{
        {0.0},
        {1.0}
    };

    bool threw = false;

    try {
        ml::chi2(X, y);
    }
    catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

} // namespace

int main()
{
    test_basic_chi2();
    test_independent_feature();
    test_empty_X();
    test_invalid_target_columns();
    test_mismatched_rows();

    return 0;
}
