#include "ml/model_selection/stratified_kfold.hpp"

#include <cassert>
#include <set>

int main() {
    const std::vector<int> labels{0, 0, 0, 0, 0, 0, 1, 1, 1, 1};

    ml::StratifiedKFold splitter(5, true, 7);
    const auto folds = splitter.split(labels);
    assert(folds.size() == 5);

    for (const auto& fold : folds) {
        assert(fold.train.size() + fold.test.size() == labels.size());

        // Every sample appears in either train or test, never both.
        std::set<size_t> combined(fold.train.begin(), fold.train.end());
        for (size_t index : fold.test) assert(combined.insert(index).second);
        assert(combined.size() == labels.size());
    }

    // All test folds together cover the whole dataset exactly once.
    std::set<size_t> all_test_indices;
    for (const auto& fold : folds) {
        for (size_t index : fold.test) assert(all_test_indices.insert(index).second);
    }
    assert(all_test_indices.size() == labels.size());
}
