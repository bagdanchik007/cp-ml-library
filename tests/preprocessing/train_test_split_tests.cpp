#include "ml/preprocessing/train_test_split.hpp"

#include <cassert>
#include <stdexcept>

namespace {

void test_default_split() {
    ml::Matrix data{
        {1.0, 2.0},
        {3.0, 4.0},
        {5.0, 6.0},
        {7.0, 8.0},
        {9.0, 10.0}
    };

    const ml::TrainTestSplit result =
        ml::train_test_split(data);

    assert(result.x_train.rows == 4);
    assert(result.x_train.cols == 2);

    assert(result.x_test.rows == 1);
    assert(result.x_test.cols == 2);

    assert(result.x_train(0, 0) == 1.0);
    assert(result.x_train(3, 1) == 8.0);

    assert(result.x_test(0, 0) == 9.0);
    assert(result.x_test(0, 1) == 10.0);
}

void test_custom_split() {
    ml::Matrix data{
        {1.0},
        {2.0},
        {3.0},
        {4.0}
    };

    const ml::TrainTestSplit result =
        ml::train_test_split(data, 0.5);

    assert(result.x_train.rows == 2);
    assert(result.x_test.rows == 2);

    assert(result.x_train(0, 0) == 1.0);
    assert(result.x_train(1, 0) == 2.0);

    assert(result.x_test(0, 0) == 3.0);
    assert(result.x_test(1, 0) == 4.0);
}

void test_invalid_test_size() {
    ml::Matrix data{
        {1.0, 2.0},
        {3.0, 4.0}
    };

    bool threw = false;

    try {
        ml::train_test_split(data, 0.0);
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

void test_empty_input() {
    bool threw = false;

    try {
        ml::train_test_split(ml::Matrix{});
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

} // namespace

int main() {
    test_default_split();
    test_custom_split();
    test_invalid_test_size();
    test_empty_input();

    return 0;
}