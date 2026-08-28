#include "ml/ml.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace ml;

void test_linear_regression() {
    std::cout << "[TEST] Linear Regression ... ";

    // y = 3*x + 2
    Matrix X = {
        {1.0},
        {2.0},
        {3.0},
        {4.0},
        {5.0}
    };

    std::vector<double> y = {
        5.0,
        8.0,
        11.0,
        14.0,
        17.0
    };

    LinearRegression lr;

    lr.fit(X, y, 0.05, 3000, false);

    assert(lr.is_fitted());

    assert(
        std::abs(lr.weights()[0] - 3.0) < 0.15
    );

    assert(
        std::abs(lr.bias() - 2.0) < 0.15
    );

    const double mse = lr.score(X, y);

    assert(mse < 0.05);

    std::cout
        << "OK (weights≈" << lr.weights()[0]
        << ", bias≈" << lr.bias()
        << ", mse=" << mse
        << ")\n";
}

void test_linear_regression_prediction() {
    std::cout << "[TEST] Linear Regression prediction ... ";

    // y = 2*x + 1
    Matrix X = {
        {1.0},
        {2.0},
        {3.0},
        {4.0}
    };

    std::vector<double> y = {
        3.0,
        5.0,
        7.0,
        9.0
    };

    LinearRegression lr;

    lr.fit(X, y, 0.05, 3000);

    Matrix prediction_input = {
        {5.0},
        {6.0}
    };

    const auto predictions =
        lr.predict(prediction_input);

    assert(predictions.size() == 2);

    assert(
        std::abs(predictions[0] - 11.0) < 0.2
    );

    assert(
        std::abs(predictions[1] - 13.0) < 0.2
    );

    std::cout << "OK\n";
}

void test_linear_regression_unfitted_prediction() {
    std::cout << "[TEST] Unfitted prediction validation ... ";

    LinearRegression lr;

    Matrix X = {
        {1.0},
        {2.0}
    };

    bool thrown = false;

    try {
        [[maybe_unused]] const auto predictions =
            lr.predict(X);
    } catch (const std::runtime_error&) {
        thrown = true;
    }

    assert(thrown);

    std::cout << "OK\n";
}

void test_linear_regression_dimension_validation() {
    std::cout << "[TEST] Linear Regression dimension validation ... ";

    Matrix X = {
        {1.0},
        {2.0},
        {3.0}
    };

    std::vector<double> y = {
        3.0,
        5.0,
        7.0
    };

    LinearRegression lr;

    lr.fit(X, y, 0.05, 1000);

    Matrix invalid_X = {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    bool prediction_thrown = false;

    try {
        [[maybe_unused]] const auto predictions =
            lr.predict(invalid_X);
    } catch (const std::invalid_argument&) {
        prediction_thrown = true;
    }

    assert(prediction_thrown);

    std::cout << "OK\n";
}

void test_linear_regression_training_validation() {
    std::cout << "[TEST] Linear Regression training validation ... ";

    Matrix X = {
        {1.0},
        {2.0},
        {3.0}
    };

    std::vector<double> y = {
        3.0,
        5.0,
        7.0
    };

    {
        LinearRegression lr;

        bool thrown = false;

        try {
            lr.fit(X, y, 0.0, 1000);
        } catch (const std::invalid_argument&) {
            thrown = true;
        }

        assert(thrown);
    }

    {
        LinearRegression lr;

        bool thrown = false;

        try {
            lr.fit(X, y, -0.01, 1000);
        } catch (const std::invalid_argument&) {
            thrown = true;
        }

        assert(thrown);
    }

    {
        LinearRegression lr;

        bool thrown = false;

        try {
            lr.fit(X, y, 0.01, 0);
        } catch (const std::invalid_argument&) {
            thrown = true;
        }

        assert(thrown);
    }

    {
        LinearRegression lr;

        bool thrown = false;

        try {
            lr.fit(X, y, 0.01, -100);
        } catch (const std::invalid_argument&) {
            thrown = true;
        }

        assert(thrown);
    }

    std::cout << "OK\n";
}

void test_linear_regression_sample_validation() {
    std::cout << "[TEST] Linear Regression sample validation ... ";

    Matrix X = {
        {1.0},
        {2.0},
        {3.0}
    };

    std::vector<double> invalid_y = {
        3.0,
        5.0
    };

    LinearRegression lr;

    bool thrown = false;

    try {
        lr.fit(X, invalid_y, 0.01, 1000);
    } catch (const std::invalid_argument&) {
        thrown = true;
    }

    assert(thrown);

    std::cout << "OK\n";
}

void test_linear_regression_empty_data() {
    std::cout << "[TEST] Linear Regression empty data validation ... ";

    LinearRegression lr;

    Matrix empty_matrix;

    std::vector<double> empty_y;

    bool thrown = false;

    try {
        lr.fit(empty_matrix, empty_y, 0.01, 1000);
    } catch (const std::invalid_argument&) {
        thrown = true;
    }

    assert(thrown);

    std::cout << "OK\n";
}

int main() {
    std::cout << "Running linear regression tests...\n\n";

    test_linear_regression();
    test_linear_regression_prediction();
    test_linear_regression_unfitted_prediction();
    test_linear_regression_dimension_validation();
    test_linear_regression_training_validation();
    test_linear_regression_sample_validation();
    test_linear_regression_empty_data();

    std::cout
        << "\nAll linear regression tests passed!\n";

    return 0;
}