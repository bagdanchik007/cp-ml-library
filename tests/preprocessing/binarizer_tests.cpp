#include "ml/preprocessing/binarizer.hpp"

#include <cassert>
#include <stdexcept>

namespace {

void test_default_threshold() {
    ml::Matrix data{
        {-1.0, 0.0, 1.0},
        {2.5, -3.0, 0.5}
    };

    ml::Binarizer binarizer;

    const ml::Matrix result =
        binarizer.transform(data);

    assert(result.rows == 2);
    assert(result.cols == 3);

    assert(result(0, 0) == 0.0);
    assert(result(0, 1) == 0.0);
    assert(result(0, 2) == 1.0);

    assert(result(1, 0) == 1.0);
    assert(result(1, 1) == 0.0);
    assert(result(1, 2) == 1.0);
}

void test_custom_threshold() {
    ml::Matrix data{
        {2.0, 5.0, 8.0}
    };

    ml::Binarizer binarizer(5.0);

    const ml::Matrix result =
        binarizer.transform(data);

    assert(result.rows == 1);
    assert(result.cols == 3);

    assert(result(0, 0) == 0.0);
    assert(result(0, 1) == 0.0);
    assert(result(0, 2) == 1.0);
}

void test_fit_transform() {
    ml::Matrix data{
        {-2.0},
        {3.0}
    };

    ml::Binarizer binarizer;

    const ml::Matrix result =
        binarizer.fit_transform(data);

    assert(result.rows == 2);
    assert(result.cols == 1);

    assert(result(0, 0) == 0.0);
    assert(result(1, 0) == 1.0);
}

void test_empty_input() {
    ml::Binarizer binarizer;

    bool threw = false;

    try {
        binarizer.transform(ml::Matrix{});
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

} // namespace

int main() {
    test_default_threshold();
    test_custom_threshold();
    test_fit_transform();
    test_empty_input();

    return 0;
}