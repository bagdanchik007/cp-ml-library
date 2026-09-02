#include "ml/preprocessing/ordinal_encoder.hpp"

#include <cassert>

int main() {
    const std::vector<std::vector<std::string>> columns{
        {"red", "green", "blue", "green"},
        {"small", "large", "small", "medium"}
    };

    ml::OrdinalEncoder encoder;
    assert(!encoder.is_fitted());
    const auto encoded = encoder.fit_transform(columns);
    assert(encoder.is_fitted());

    assert(encoded.size() == 2);
    assert(encoded[0][0] == 0); // red -> first seen -> 0
    assert(encoded[0][1] == 1); // green -> second seen -> 1
    assert(encoded[0][2] == 2); // blue -> third seen -> 2
    assert(encoded[0][3] == 1); // green repeats -> 1

    assert(encoder.n_categories(0) == 3);
    assert(encoder.n_categories(1) == 3);

    bool threw = false;
    try {
        encoder.transform({{"purple"}, {"small"}});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}
