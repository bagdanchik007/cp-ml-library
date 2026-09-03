#include "ml/model_selection/shuffle_split.hpp"

#include <cassert>
#include <set>

int main() {
    ml::ShuffleSplit splitter(5, 0.2, 7);
    const auto folds = splitter.split(10);
    assert(folds.size() == 5);

    for (const auto& fold : folds) {
        assert(fold.test.size() == 2);
        assert(fold.train.size() == 8);

        // No overlap between train and test within a single split.
        std::set<size_t> combined(fold.train.begin(), fold.train.end());
        for (size_t index : fold.test) assert(combined.insert(index).second);
        assert(combined.size() == 10);
    }

    bool threw_zero = false;
    try {
        ml::ShuffleSplit invalid(5, 0.0);
    } catch (const std::invalid_argument&) {
        threw_zero = true;
    }
    assert(threw_zero);

    bool threw_too_small = false;
    try {
        ml::ShuffleSplit(5, 0.05).split(10); // rounds down to 0 test samples
    } catch (const std::invalid_argument&) {
        threw_too_small = true;
    }
    assert(threw_too_small);
}
