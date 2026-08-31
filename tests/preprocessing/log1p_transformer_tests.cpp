#include "ml/preprocessing/log1p_transformer.hpp"

#include <cassert>
#include <cmath>

int main() {
    const ml::Matrix data{{0.0, 1.0}, {std::exp(1.0) - 1.0, 9.0}};

    ml::Log1pTransformer transformer;
    const auto transformed = transformer.transform(data);

    assert(std::abs(transformed(0, 0) - 0.0) < 1e-12);
    assert(std::abs(transformed(0, 1) - std::log(2.0)) < 1e-12);
    assert(std::abs(transformed(1, 0) - 1.0) < 1e-9);
    assert(std::abs(transformed(1, 1) - std::log(10.0)) < 1e-12);

    const auto restored = transformer.inverse_transform(transformed);
    for (size_t row = 0; row < data.rows; ++row) {
        for (size_t column = 0; column < data.cols; ++column) {
            assert(std::abs(restored(row, column) - data(row, column)) < 1e-9);
        }
    }

    bool threw = false;
    try {
        transformer.transform(ml::Matrix{{-2.0}});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}
