#include "ml/preprocessing/function_transformer.hpp"

#include <cassert>
#include <cmath>

int main() {
    const ml::Matrix data{{1.0, 4.0}, {9.0, 16.0}};

    ml::FunctionTransformer sqrt_transformer(
        [](double x) { return std::sqrt(x); },
        [](double x) { return x * x; }
    );

    const auto transformed = sqrt_transformer.transform(data);
    assert(transformed(0, 0) == 1.0);
    assert(transformed(0, 1) == 2.0);
    assert(transformed(1, 0) == 3.0);
    assert(transformed(1, 1) == 4.0);
    assert(sqrt_transformer.has_inverse());

    const auto restored = sqrt_transformer.inverse_transform(transformed);
    for (size_t row = 0; row < data.rows; ++row) {
        for (size_t column = 0; column < data.cols; ++column) {
            assert(std::abs(restored(row, column) - data(row, column)) < 1e-9);
        }
    }

    ml::FunctionTransformer no_inverse([](double x) { return x + 1.0; });
    assert(!no_inverse.has_inverse());
    bool threw = false;
    try {
        no_inverse.inverse_transform(data);
    } catch (const std::logic_error&) {
        threw = true;
    }
    assert(threw);
}
