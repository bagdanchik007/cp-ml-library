#pragma once

#include "ml/model_selection/kfold.hpp"

#include <algorithm>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace ml {

/**
 * @brief Repeats KFold cross-validation multiple times with different
 * random shuffles, producing n_repeats * n_splits folds in total.
 *
 * Averaging scores over the repeats reduces the variance in the
 * cross-validation estimate compared to a single KFold pass, at the
 * cost of proportionally more compute.
 */
class RepeatedKFold {
public:
    RepeatedKFold(size_t n_splits = 5, size_t n_repeats = 10, unsigned seed = 42)
        : n_splits_(n_splits), n_repeats_(n_repeats), seed_(seed) {
        if (n_splits_ < 2) throw std::invalid_argument("RepeatedKFold: n_splits must be at least 2");
        if (n_repeats_ == 0) throw std::invalid_argument("RepeatedKFold: n_repeats must be at least 1");
    }

    std::vector<FoldIndices> split(size_t sample_count) const {
        if (sample_count < n_splits_) throw std::invalid_argument("RepeatedKFold::split: fewer samples than folds");

        std::vector<FoldIndices> result;
        result.reserve(n_splits_ * n_repeats_);

        for (size_t repeat = 0; repeat < n_repeats_; ++repeat) {
            std::vector<size_t> indices(sample_count);
            std::iota(indices.begin(), indices.end(), 0);
            std::mt19937 generator(seed_ + static_cast<unsigned>(repeat));
            std::shuffle(indices.begin(), indices.end(), generator);

            for (size_t fold = 0; fold < n_splits_; ++fold) {
                FoldIndices item;
                for (size_t i = 0; i < indices.size(); ++i) {
                    if (i % n_splits_ == fold) item.test.push_back(indices[i]);
                    else item.train.push_back(indices[i]);
                }
                result.push_back(std::move(item));
            }
        }
        return result;
    }

private:
    size_t n_splits_;
    size_t n_repeats_;
    unsigned seed_;
};

} // namespace ml
