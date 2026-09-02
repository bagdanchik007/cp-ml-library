#pragma once

#include "ml/model_selection/kfold.hpp"

#include <numeric>
#include <stdexcept>
#include <vector>

namespace ml {

/**
 * @brief Leave-One-Out cross-validation splitter.
 *
 * Equivalent to KFold with n_splits == sample_count: each fold holds
 * exactly one sample for testing and all remaining samples for training.
 * Useful for small datasets where every sample counts, though it is
 * expensive for large datasets since it produces one fold per sample.
 */
class LeaveOneOut {
public:
    LeaveOneOut() = default;

    std::vector<FoldIndices> split(size_t sample_count) const {
        if (sample_count < 2) throw std::invalid_argument("LeaveOneOut::split: need at least 2 samples");

        std::vector<size_t> indices(sample_count);
        std::iota(indices.begin(), indices.end(), 0);

        std::vector<FoldIndices> result;
        result.reserve(sample_count);
        for (size_t test_index = 0; test_index < sample_count; ++test_index) {
            FoldIndices item;
            item.test = {test_index};
            item.train.reserve(sample_count - 1);
            for (size_t i = 0; i < sample_count; ++i) {
                if (i != test_index) item.train.push_back(i);
            }
            result.push_back(std::move(item));
        }
        return result;
    }
};

} // namespace ml
