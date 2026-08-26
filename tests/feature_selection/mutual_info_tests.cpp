#include "ml/feature_selection/mutual_info.hpp"

#include <cassert>
#include <cmath>
#include <stdexcept>

namespace {

void test_basic_mutual_information()
{
    const ml::Matrix X{
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };

    const ml::Matrix y{
        {0.0},
        {0.0},
        {1.0},
        {1.0}
    };

    const ml::MutualInfoResult result =
        ml::mutual_info(X, y);

    assert(result.scores.size() == 2);

    assert(result.scores[0] > 0.0);
    assert(std::isfinite(result.scores[0]));

    assert(result.scores[1] < 1e-12);
}

void test_independent_feature()
{
    const ml::Matrix X{
        {0.0},
        {1.0},
        {0.0},
        {1.0}
    };

    const ml::Matrix y{
        {0.0},
        {0.0},
        {1.0},
        {1.0}
    };

    const ml::MutualInfoResult result =
        ml::mutual_info(X, y);

    assert(result.scores.size() == 1);
    assert(std::abs(result.scores[0]) < 1e-12);
}

void test_empty_input()
{
    bool threw = false;

    try {
        ml::mutual_info(
            ml::Matrix{},
            ml::Matrix{}
        );
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
        {2.0}
    };

    const ml::Matrix y{
        {1.0}
    };

    bool threw = false;

    try {
        ml::mutual_info(X, y);
    }
    catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

void test_invalid_target_shape()
{
    const ml::Matrix X{
        {1.0},
        {2.0}
    };

    const ml::Matrix y{
        {1.0, 2.0},
        {3.0, 4.0}
    };

    bool threw = false;

    try {
        ml::mutual_info(X, y);
    }
    catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

} // namespace

int main()
{
    test_basic_mutual_information();
    test_independent_feature();
    test_empty_input();
    test_mismatched_rows();
    test_invalid_target_shape();

    return 0;
}
