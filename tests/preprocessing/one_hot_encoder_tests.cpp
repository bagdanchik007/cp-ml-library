#include "ml/preprocessing/one_hot_encoder.hpp"

#include <cassert>
#include <stdexcept>
#include <vector>

namespace {

void test_fit_transform() {
    const std::vector<int> labels{
        0,
        1,
        2,
        1
    };

    ml::OneHotEncoder encoder;

    const ml::Matrix result =
        encoder.fit_transform(labels);

    assert(result.rows == 4);
    assert(result.cols == 3);

    assert(result(0, 0) == 1.0);
    assert(result(0, 1) == 0.0);
    assert(result(0, 2) == 0.0);

    assert(result(1, 0) == 0.0);
    assert(result(1, 1) == 1.0);
    assert(result(1, 2) == 0.0);

    assert(result(2, 0) == 0.0);
    assert(result(2, 1) == 0.0);
    assert(result(2, 2) == 1.0);

    assert(result(3, 0) == 0.0);
    assert(result(3, 1) == 1.0);
    assert(result(3, 2) == 0.0);
}

void test_sorted_categories() {
    const std::vector<int> labels{
        30,
        10,
        20
    };

    ml::OneHotEncoder encoder;

    const ml::Matrix result =
        encoder.fit_transform(labels);

    assert(result.rows == 3);
    assert(result.cols == 3);

    assert(result(0, 2) == 1.0);
    assert(result(1, 0) == 1.0);
    assert(result(2, 1) == 1.0);
}

void test_transform_after_fit() {
    ml::OneHotEncoder encoder;

    encoder.fit({
        1,
        2,
        3
    });

    const ml::Matrix result =
        encoder.transform({
            3,
            1,
            2
        });

    assert(result.rows == 3);
    assert(result.cols == 3);

    assert(result(0, 2) == 1.0);
    assert(result(1, 0) == 1.0);
    assert(result(2, 1) == 1.0);
}

void test_transform_before_fit() {
    ml::OneHotEncoder encoder;

    bool threw = false;

    try {
        encoder.transform({
            1,
            2
        });
    } catch (const std::logic_error&) {
        threw = true;
    }

    assert(threw);
}

void test_unknown_category() {
    ml::OneHotEncoder encoder;

    encoder.fit({
        1,
        2,
        3
    });

    bool threw = false;

    try {
        encoder.transform({
            1,
            4
        });
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

void test_empty_input() {
    ml::OneHotEncoder encoder;

    bool threw = false;

    try {
        encoder.fit({});
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}
void test_categories_are_sorted() {
    ml::OneHotEncoder encoder;

    const std::vector<int> labels{
        30,
        10,
        20
    };

    const ml::Matrix result =
        encoder.fit_transform(labels);

    assert(result.rows == 3);
    assert(result.cols == 3);

    // Categories: 10, 20, 30
    assert(result(0, 2) == 1.0);
    assert(result(1, 0) == 1.0);
    assert(result(2, 1) == 1.0);
}

void test_unknown_label() {
    ml::OneHotEncoder encoder;

    encoder.fit({
        10,
        20,
        30
    });

    bool threw = false;

    try {
        encoder.transform({
            40
        });
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}


void test_empty_fit() {
    ml::OneHotEncoder encoder;

    bool threw = false;

    try {
        encoder.fit({});
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

} // namespace

int main() {
    test_fit_transform();
    test_sorted_categories();
    test_transform_after_fit();
    test_transform_before_fit();
    test_unknown_category();
    test_empty_input();
    test_empty_fit();
    return 0;
}
