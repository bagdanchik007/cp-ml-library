#pragma once

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ml {

struct FoldIndices { std::vector<size_t> train; std::vector<size_t> test; };

class KFold {
public:
    explicit KFold(size_t n_splits = 5, bool shuffle = true, unsigned seed = 42)
        : n_splits_(n_splits), shuffle_(shuffle), seed_(seed) {
        if (n_splits_ < 2) throw std::invalid_argument("KFold: n_splits must be at least 2");
    }

    std::vector<FoldIndices> split(size_t sample_count) const {
        if (sample_count < n_splits_) throw std::invalid_argument("KFold::split: fewer samples than folds");
        std::vector<size_t> indices(sample_count); std::iota(indices.begin(), indices.end(), 0);
        if (shuffle_) { std::mt19937 generator(seed_); std::shuffle(indices.begin(), indices.end(), generator); }
        std::vector<FoldIndices> result; result.reserve(n_splits_);
        const size_t base = sample_count / n_splits_, remainder = sample_count % n_splits_;
        size_t begin = 0;
        for (size_t fold = 0; fold < n_splits_; ++fold) {
            const size_t size = base + (fold < remainder ? 1 : 0), end = begin + size;
            FoldIndices item; item.test.insert(item.test.end(), indices.begin() + static_cast<std::ptrdiff_t>(begin), indices.begin() + static_cast<std::ptrdiff_t>(end));
            item.train.reserve(sample_count - size);
            item.train.insert(item.train.end(), indices.begin(), indices.begin() + static_cast<std::ptrdiff_t>(begin));
            item.train.insert(item.train.end(), indices.begin() + static_cast<std::ptrdiff_t>(end), indices.end());
            result.push_back(std::move(item)); begin = end;
        }
        return result;
    }

private:
    size_t n_splits_; bool shuffle_; unsigned seed_;
};

} // namespace ml
