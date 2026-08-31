#pragma once

#include "ml/model_selection/kfold.hpp"

#include <algorithm>
#include <map>
#include <random>
#include <stdexcept>
#include <vector>

namespace ml {

/**
 * @brief K-Fold cross-validation splitter that preserves class proportions.
 *
 * Samples are grouped by label, shuffled within each group, and then
 * distributed round-robin across the folds so each fold's class ratio
 * approximates the ratio of the full dataset.
 */
class StratifiedKFold {
public:
    explicit StratifiedKFold(size_t n_splits = 5, bool shuffle = true, unsigned seed = 42)
        : n_splits_(n_splits), shuffle_(shuffle), seed_(seed) {
        if (n_splits_ < 2) throw std::invalid_argument("StratifiedKFold: n_splits must be at least 2");
    }

    std::vector<FoldIndices> split(const std::vector<int>& labels) const {
        if (labels.size() < n_splits_) throw std::invalid_argument("StratifiedKFold::split: fewer samples than folds");

        std::map<int, std::vector<size_t>> groups;
        for (size_t i = 0; i < labels.size(); ++i) groups[labels[i]].push_back(i);

        if (shuffle_) {
            std::mt19937 generator(seed_);
            for (auto& [label, indices] : groups) std::shuffle(indices.begin(), indices.end(), generator);
        }

        std::vector<std::vector<size_t>> fold_test_indices(n_splits_);
        for (const auto& [label, indices] : groups) {
            for (size_t i = 0; i < indices.size(); ++i) {
                fold_test_indices[i % n_splits_].push_back(indices[i]);
            }
        }

        std::vector<FoldIndices> result;
        result.reserve(n_splits_);
        for (size_t fold = 0; fold < n_splits_; ++fold) {
            FoldIndices item;
            item.test = fold_test_indices[fold];
            std::sort(item.test.begin(), item.test.end());
            for (size_t other = 0; other < n_splits_; ++other) {
                if (other == fold) continue;
                item.train.insert(item.train.end(), fold_test_indices[other].begin(), fold_test_indices[other].end());
            }
            std::sort(item.train.begin(), item.train.end());
            result.push_back(std::move(item));
        }
        return result;
    }

private:
    size_t n_splits_;
    bool shuffle_;
    unsigned seed_;
};

} // namespace ml
