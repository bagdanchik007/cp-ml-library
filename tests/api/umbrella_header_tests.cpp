#include "ml/ml.hpp"

#include <cassert>
#include <cmath>

namespace {

bool almost_equal(double a, double b, double epsilon = 1e-9) {
    return std::abs(a - b) < epsilon;
}

void test_matrix_api() {
    ml::Matrix matrix{
        {1.0, 2.0},
        {3.0, 4.0}
    };

    assert(matrix.rows == 2);
    assert(matrix.cols == 2);
    assert(almost_equal(matrix(0, 0), 1.0));
    assert(almost_equal(matrix(1, 1), 4.0));
}

void test_standard_scaler_api() {
    ml::Matrix data{
        {1.0, 2.0},
        {3.0, 4.0},
        {5.0, 6.0}
    };

    ml::StandardScaler scaler;

    const ml::Matrix transformed =
        scaler.fit_transform(data);

    assert(almost_equal(transformed(1, 0), 0.0));
    assert(almost_equal(transformed(1, 1), 0.0));
}

} // namespace

int main() {
    test_matrix_api();
    test_standard_scaler_api();

    return 0;
}
