#include "ml/model_selection/repeated_kfold.hpp"

#include <cassert>
#include <set>

int main() {
    ml::RepeatedKFold splitter(5, 3, 7);
    const auto folds = splitter.split(10);

    // 5 splits * 3 repeats = 15 folds total.
    assert(folds.size() == 15);

    for (const auto& fold : folds) {
        assert(fold.train.size() + fold.test.size() == 10);
        std::set<size_t> combined(fold.train.begin(), fold.train.end());
        for (size_t index : fold.test) assert(combined.insert(index).second);
        assert(combined.size() == 10);
    }

    // Different repeats should not always produce identical test folds
    // (i.e. the shuffling actually changes something across repeats).
    bool any_repeat_differs = false;
    for (size_t fold = 0; fold < 5; ++fold) {
        if (folds[fold].test != folds[fold + 5].test) any_repeat_differs = true;
    }
    assert(any_repeat_differs);

    bool threw = false;
    try {
        ml::RepeatedKFold invalid(5, 0);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}
