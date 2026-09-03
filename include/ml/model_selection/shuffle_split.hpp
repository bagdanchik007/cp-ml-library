#pragma once

#include "ml/model_selection/kfold.hpp"

#include <algorithm>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace ml {

/**
 * @brief Generates independent random train/test splits.
 *
 * Unlike KFold, splits are not guaranteed to partition the dataset:
 * the same sample can appear in the test set of multiple splits, and
 * some samples may never be selected for testing at all. Useful when
 * the exact fold structure of KFold isn't needed and a fixed test
 * proportion with repeated random sampling is preferred.
 */
class ShuffleSplit {
public:
    ShuffleSplit(size_t n_splits = 10, double test_size = 0.2, unsigned seed = 42)
        : n_splits_(n_splits), test_size_(test_size), seed_(seed) {
        if (n_splits_ == 0) throw std::invalid_argument("ShuffleSplit: n_splits must be at least 1");
        if (test_size_ <= 0.0 || test_size_ >= 1.0) throw std::invalid_argument("ShuffleSplit: test_size must be within (0, 1)");
    }

    std::vector<FoldIndices> split(size_t sample_count) const {
        const auto test_count = static_cast<size_t>(test_size_ * static_cast<double>(sample_count));
        if (test_count == 0 || test_count >= sample_count) {
            throw std::invalid_argument("ShuffleSplit::split: test_size produces an empty train or test set for this sample count");
        }

        std::vector<FoldIndices> result;
        result.reserve(n_splits_);

        for (size_t split_index = 0; split_index < n_splits_; ++split_index) {
            std::vector<size_t> indices(sample_count);
            std::iota(indices.begin(), indices.end(), 0);
            std::mt19937 generator(seed_ + static_cast<unsigned>(split_index));
            std::shuffle(indices.begin(), indices.end(), generator);

            FoldIndices item;
            item.test.assign(indices.begin(), indices.begin() + static_cast<std::ptrdiff_t>(test_count));
            item.train.assign(indices.begin() + static_cast<std::ptrdiff_t>(test_count), indices.end());
            result.push_back(std::move(item));
        }
        return result;
    }

private:
    size_t n_splits_;
    double test_size_;
    unsigned seed_;
};

} // namespace ml
