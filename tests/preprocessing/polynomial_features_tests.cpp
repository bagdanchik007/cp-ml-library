#include "ml/preprocessing/polynomial_features.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace ml;

namespace {

void test_degree_two()
{
    PolynomialFeatures transformer(2);

    const Matrix data{
        {1.0, 2.0}
    };

    const Matrix result =
        transformer.fit_transform(
            data
        );

    assert(result.rows == 1);

    assert(result.cols == 6);

    assert(
        std::abs(
            result(0, 0) - 1.0
        ) < 1e-9
    );

    assert(
        std::abs(
            result(0, 1) - 1.0
        ) < 1e-9
    );

    assert(
        std::abs(
            result(0, 2) - 2.0
        ) < 1e-9
    );

    assert(
        std::abs(
            result(0, 3) - 1.0
        ) < 1e-9
    );

    assert(
        std::abs(
            result(0, 4) - 2.0
        ) < 1e-9
    );

    assert(
        std::abs(
            result(0, 5) - 4.0
        ) < 1e-9
    );
}

void test_without_bias()
{
    PolynomialFeatures transformer(
        2,
        false
    );

    const Matrix data{
        {2.0}
    };

    const Matrix result =
        transformer.fit_transform(
            data
        );

    assert(result.rows == 1);

    assert(result.cols == 2);

    assert(
        std::abs(
            result(0, 0) - 2.0
        ) < 1e-9
    );

    assert(
        std::abs(
            result(0, 1) - 4.0
        ) < 1e-9
    );
}

void test_multiple_rows()
{
    PolynomialFeatures transformer(2);

    const Matrix data{
        {1.0, 2.0},
        {3.0, 4.0}
    };

    const Matrix result =
        transformer.fit_transform(
            data
        );

    assert(result.rows == 2);

    assert(result.cols == 6);

    assert(
        std::abs(
            result(1, 0) - 1.0
        ) < 1e-9
    );

    assert(
        std::abs(
            result(1, 1) - 3.0
        ) < 1e-9
    );

    assert(
        std::abs(
            result(1, 2) - 4.0
        ) < 1e-9
    );
}

void test_is_fitted()
{
    PolynomialFeatures transformer;

    assert(
        !transformer.is_fitted()
    );

    transformer.fit(
        Matrix{
            {1.0}
        }
    );

    assert(
        transformer.is_fitted()
    );
}

}

int main()
{
    std::cout
        << "Running PolynomialFeatures tests...\n";

    test_degree_two();

    test_without_bias();

    test_multiple_rows();

    test_is_fitted();

    std::cout
        << "All PolynomialFeatures tests passed!\n";

    return 0;
}