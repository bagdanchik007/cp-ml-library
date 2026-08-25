#include "ml/preprocessing/label_encoder.hpp"

#include <cassert>
#include <stdexcept>
#include <vector>

namespace {

void test_inverse_transform() {
    ml::LabelEncoder encoder;

    const std::vector<int> labels{
        10,
        20,
        30,
        10,
        20
    };

    const std::vector<int> encoded =
        encoder.fit_transform(labels);

    const std::vector<int> decoded =
        encoder.inverse_transform(encoded);

    assert(decoded.size() == labels.size());

    for (size_t i = 0; i < labels.size(); ++i) {
        assert(decoded[i] == labels[i]);
    }
}

void test_inverse_transform_known_values() {
    ml::LabelEncoder encoder;

    encoder.fit({
        100,
        200,
        300
    });

    const std::vector<int> encoded{
        2,
        0,
        1
    };

    const std::vector<int> decoded =
        encoder.inverse_transform(encoded);

    assert(decoded[0] == 300);
    assert(decoded[1] == 100);
    assert(decoded[2] == 200);
}

void test_invalid_index() {
    ml::LabelEncoder encoder;

    encoder.fit({
        10,
        20,
        30
    });

    bool threw = false;

    try {
        encoder.inverse_transform({
            0,
            3
        });
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

void test_before_fit() {
    ml::LabelEncoder encoder;

    bool threw = false;

    try {
        encoder.inverse_transform({
            0
        });
    } catch (const std::logic_error&) {
        threw = true;
    }

    assert(threw);
}

} // namespace

int main() {
    test_inverse_transform();
    test_inverse_transform_known_values();
    test_invalid_index();
    test_before_fit();

    return 0;
}