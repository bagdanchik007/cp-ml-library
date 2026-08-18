#include "ml/ml.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

using namespace ml;

void test_matrix() {
    std::cout << "[TEST] Matrix utilities ... ";

    Matrix m = {{1.0, 2.0}, {3.0, 4.0}};

    assert(m.rows == 2 && m.cols == 2);
    assert(m(0, 1) == 2.0);

    auto t = m.transpose();

    assert(t.rows == 2 && t.cols == 2);
    assert(t(1, 0) == 2.0);

    std::vector<double> v = {1.0, 1.0};
    auto res = m * v;

    assert(std::abs(res[0] - 3.0) < 1e-9);
    assert(std::abs(res[1] - 7.0) < 1e-9);

    std::cout << "OK\n";
}

int main() {
    test_matrix();

    std::cout << "\nAll matrix tests passed!\n";
    return 0;
}