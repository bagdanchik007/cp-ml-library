#include "ml/model_selection/leave_one_out.hpp"

#include <cassert>
#include <set>

int main() {
    ml::LeaveOneOut loo;
    const auto folds = loo.split(5);
    assert(folds.size() == 5);

    std::set<size_t> seen_test_indices;
    for (size_t fold = 0; fold < folds.size(); ++fold) {
        assert(folds[fold].test.size() == 1);
        assert(folds[fold].train.size() == 4);
        seen_test_indices.insert(folds[fold].test[0]);

        // The held-out sample never appears in its own training set.
        for (size_t train_index : folds[fold].train) assert(train_index != folds[fold].test[0]);
    }
    assert(seen_test_indices.size() == 5);

    bool threw = false;
    try {
        loo.split(1);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}
