#pragma once

#include "ml/algorithms/decision_tree.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace ml {

/** Bootstrap aggregation of randomized CART classifiers. */
class RandomForestClassifier {
public:
    explicit RandomForestClassifier(size_t n_estimators = 100, size_t max_depth = 10,
                                    size_t min_samples_split = 2, unsigned seed = 42)
        : n_estimators_(n_estimators), max_depth_(max_depth), min_samples_split_(min_samples_split), seed_(seed) {
        if (n_estimators_ == 0) throw std::invalid_argument("RandomForestClassifier: n_estimators must be positive");
    }

    void fit(const Matrix& features, const std::vector<int>& targets) {
        if (features.rows == 0 || features.cols == 0 || targets.size() != features.rows)
            throw std::invalid_argument("RandomForestClassifier::fit: invalid training data");
        trees_.clear(); feature_count_ = features.cols;
        const size_t sampled_features = std::max<size_t>(1, static_cast<size_t>(std::sqrt(static_cast<double>(features.cols))));
        std::mt19937 generator(seed_);
        std::uniform_int_distribution<size_t> row_dist(0, features.rows - 1);
        for (size_t tree_index = 0; tree_index < n_estimators_; ++tree_index) {
            Matrix bootstrap(features.rows, features.cols); std::vector<int> labels(features.rows);
            for (size_t row = 0; row < features.rows; ++row) { const size_t selected = row_dist(generator); bootstrap.set_row(row, features.row(selected)); labels[row] = targets[selected]; }
            auto tree = std::make_unique<DecisionTreeClassifier>(max_depth_, min_samples_split_, sampled_features, seed_ + static_cast<unsigned>(tree_index));
            tree->fit(bootstrap, labels); trees_.push_back(std::move(tree));
        }
        fitted_ = true;
    }

    std::vector<int> predict(const Matrix& features) const {
        if (!fitted_) throw std::logic_error("RandomForestClassifier::predict: model has not been fitted");
        if (features.cols != feature_count_) throw std::invalid_argument("RandomForestClassifier::predict: feature count mismatch");
        std::vector<std::vector<int>> all_predictions; all_predictions.reserve(trees_.size());
        for (const auto& tree : trees_) all_predictions.push_back(tree->predict(features));
        std::vector<int> result(features.rows);
        for (size_t row = 0; row < features.rows; ++row) { std::unordered_map<int, size_t> votes; for (const auto& prediction : all_predictions) ++votes[prediction[row]]; int best = 0; size_t count = 0; for (const auto& vote : votes) if (vote.second > count || (vote.second == count && vote.first < best)) { best = vote.first; count = vote.second; } result[row] = best; }
        return result;
    }

    bool is_fitted() const noexcept { return fitted_; }

private:
    size_t n_estimators_, max_depth_, min_samples_split_, feature_count_ = 0;
    unsigned seed_;
    std::vector<std::unique_ptr<DecisionTreeClassifier>> trees_;
    bool fitted_ = false;
};

} // namespace ml
