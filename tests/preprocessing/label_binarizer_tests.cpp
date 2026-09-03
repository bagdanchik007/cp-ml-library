#include "ml/preprocessing/label_binarizer.hpp"

#include <cassert>

int main() {
    const std::vector<int> y{2, 0, 1, 0, 2};

    ml::LabelBinarizer binarizer;
    assert(!binarizer.is_fitted());
    const auto one_hot = binarizer.fit_transform(y);
    assert(binarizer.is_fitted());

    assert(one_hot.rows == 5);
    assert(one_hot.cols == 3);
    assert(binarizer.classes() == std::vector<int>({0, 1, 2}));

    // First row corresponds to label 2 -> column index 2 is set.
    assert(one_hot(0, 2) == 1.0);
    assert(one_hot(0, 0) == 0.0);
    assert(one_hot(0, 1) == 0.0);

    const auto recovered = binarizer.inverse_transform(one_hot);
    assert(recovered == y);

    bool threw = false;
    try {
        binarizer.transform({5});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}
