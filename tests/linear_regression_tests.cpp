#include "ml/ml.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

using namespace ml;

void test_linear_regression() {
    std::cout << "[TEST] Linear Regression ... ";

    // y = 3*x + 2
    Matrix X = {{1.0}, {2.0}, {3.0}, {4.0}, {5.0}};
    std::vector<double> y = {5.0, 8.0, 11.0, 14.0, 17.0};

    LinearRegression lr;
    lr.fit(X, y, 0.05, 3000, false);

    assert(lr.is_fitted());
    assert(std::abs(lr.weights()[0] - 3.0) < 0.15);
    assert(std::abs(lr.bias() - 2.0) < 0.15);

    double mse = lr.score(X, y);

    assert(mse < 0.05);

    std::cout << "OK (weights≈" << lr.weights()[0]
              << ", bias≈" << lr.bias()
              << ", mse=" << mse << ")\n";
}

int main() {
    test_linear_regression();

    std::cout << "\nAll linear regression tests passed!\n";
    return 0;
}