#pragma once

#include "ml/core/estimators/classifier.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ml {

/**
 * @brief CART-style decision tree classifier using Gini impurity.
 */
class DecisionTreeClassifier : public Classifier {
public:
    explicit DecisionTreeClassifier(
        size_t max_depth = 10,
        size_t min_samples_split = 2,
        size_t max_features = 0,
        unsigned seed = 42
    )
        : max_depth_(max_depth),
          min_samples_split_(min_samples_split),
          max_features_(max_features),
          generator_(seed) {
        if (min_samples_split_ < 2) {
            throw std::invalid_argument(
                "DecisionTreeClassifier: min_samples_split must be at least 2"
            );
        }
    }

    /**
     * @brief Train the decision tree.
     */
    void fit(
        const Matrix& features,
        const Matrix& targets
    ) override {
        if (features.rows == 0 ||
            features.cols == 0 ||
            targets.rows != features.rows ||
            targets.cols != 1) {
            throw std::invalid_argument(
                "DecisionTreeClassifier::fit: invalid training data"
            );
        }

        std::vector<int> labels(targets.rows);

        for (size_t i = 0; i < targets.rows; ++i) {
            labels[i] = static_cast<int>(targets(i, 0));
        }

        fit(features, labels);
    }

    /**
     * @brief Train the decision tree using integer class labels.
     *
     * This overload preserves the original API.
     */
    void fit(
        const Matrix& features,
        const std::vector<int>& targets
    ) {
        if (features.rows == 0 ||
            features.cols == 0 ||
            targets.size() != features.rows) {
            throw std::invalid_argument(
                "DecisionTreeClassifier::fit: invalid training data"
            );
        }

        feature_count_ = features.cols;

        std::vector<size_t> samples(features.rows);
        std::iota(samples.begin(), samples.end(), 0);

        root_ = build(
            features,
            targets,
            samples,
            0
        );

        fitted_ = true;
    }

    /**
     * @brief Predict class labels through the Classifier interface.
     */
    Matrix predict(const Matrix& features) const override {
        const auto labels = predict_labels(features);

        Matrix result(labels.size(), 1);

        for (size_t i = 0; i < labels.size(); ++i) {
            result(i, 0) = static_cast<double>(labels[i]);
        }

        return result;
    }

    /**
     * @brief Predict integer class labels.
     *
     * Preserves the original integer-based API.
     */
    std::vector<int> predict_labels(
        const Matrix& features
    ) const {
        if (!fitted_) {
            throw std::logic_error(
                "DecisionTreeClassifier::predict: "
                "model has not been fitted"
            );
        }

        if (features.cols != feature_count_) {
            throw std::invalid_argument(
                "DecisionTreeClassifier::predict: "
                "feature count mismatch"
            );
        }

        std::vector<int> result(features.rows);

        for (size_t row = 0; row < features.rows; ++row) {
            const Node* node = root_.get();

            while (!node->leaf) {
                node =
                    features(row, node->feature) <= node->threshold
                    ? node->left.get()
                    : node->right.get();
            }

            result[row] = node->label;
        }

        return result;
    }

    /**
     * @brief Return class probabilities.
     *
     * The current tree stores one class label per leaf, so
     * predictions are represented as a one-hot probability matrix.
     */
    Matrix predict_proba(
        const Matrix& features
    ) const override {
        const auto labels = predict_labels(features);

        if (labels.empty()) {
            return Matrix();
        }

        int max_label = labels[0];

        for (const int label : labels) {
            max_label = std::max(max_label, label);
        }

        if (max_label < 0) {
            throw std::runtime_error(
                "DecisionTreeClassifier: invalid class label"
            );
        }

        Matrix probabilities(
            labels.size(),
            static_cast<size_t>(max_label + 1)
        );

        for (size_t row = 0; row < labels.size(); ++row) {
            probabilities(
                row,
                static_cast<size_t>(labels[row])
            ) = 1.0;
        }

        return probabilities;
    }

    bool is_fitted() const noexcept {
        return fitted_;
    }

private:
    struct Node {
        bool leaf = true;
        int label = 0;
        size_t feature = 0;
        double threshold = 0.0;

        std::unique_ptr<Node> left;
        std::unique_ptr<Node> right;
    };

    static int majority_label(
        const std::vector<int>& targets,
        const std::vector<size_t>& samples
    ) {
        std::unordered_map<int, size_t> counts;

        for (size_t index : samples) {
            ++counts[targets[index]];
        }

        int best = targets[samples.front()];
        size_t best_count = 0;

        for (const auto& item : counts) {
            if (item.second > best_count ||
                (item.second == best_count &&
                 item.first < best)) {
                best = item.first;
                best_count = item.second;
            }
        }

        return best;
    }

    static double gini(
        const std::vector<int>& targets,
        const std::vector<size_t>& samples
    ) {
        if (samples.empty()) {
            return 0.0;
        }

        std::unordered_map<int, size_t> counts;

        for (size_t index : samples) {
            ++counts[targets[index]];
        }

        double impurity = 1.0;

        for (const auto& item : counts) {
            const double p =
                static_cast<double>(item.second) /
                static_cast<double>(samples.size());

            impurity -= p * p;
        }

        return impurity;
    }

    std::unique_ptr<Node> build(
        const Matrix& x,
        const std::vector<int>& y,
        const std::vector<size_t>& samples,
        size_t depth
    ) {
        auto node = std::make_unique<Node>();

        node->label =
            majority_label(y, samples);

        if (depth >= max_depth_ ||
            samples.size() < min_samples_split_ ||
            gini(y, samples) == 0.0) {
            return node;
        }

        std::vector<size_t> candidates(x.cols);

        std::iota(
            candidates.begin(),
            candidates.end(),
            0
        );

        if (max_features_ != 0 &&
            max_features_ < candidates.size()) {
            std::shuffle(
                candidates.begin(),
                candidates.end(),
                generator_
            );

            candidates.resize(max_features_);
        }

        double best_score =
            std::numeric_limits<double>::infinity();

        size_t best_feature = 0;
        double best_threshold = 0.0;

        std::vector<size_t> best_left;
        std::vector<size_t> best_right;

        for (size_t feature : candidates) {
            std::vector<double> values;
            values.reserve(samples.size());

            for (size_t row : samples) {
                values.push_back(x(row, feature));
            }

            std::sort(
                values.begin(),
                values.end()
            );

            values.erase(
                std::unique(
                    values.begin(),
                    values.end()
                ),
                values.end()
            );

            for (size_t i = 1; i < values.size(); ++i) {
                const double threshold =
                    (values[i - 1] + values[i]) / 2.0;

                std::vector<size_t> left;
                std::vector<size_t> right;

                for (size_t row : samples) {
                    if (x(row, feature) <= threshold) {
                        left.push_back(row);
                    } else {
                        right.push_back(row);
                    }
                }

                if (left.empty() || right.empty()) {
                    continue;
                }

                const double score =
                    (
                        left.size() * gini(y, left) +
                        right.size() * gini(y, right)
                    ) /
                    static_cast<double>(samples.size());

                if (score < best_score) {
                    best_score = score;
                    best_feature = feature;
                    best_threshold = threshold;
                    best_left = std::move(left);
                    best_right = std::move(right);
                }
            }
        }

        if (best_left.empty() || best_right.empty()) {
            return node;
        }

        node->leaf = false;
        node->feature = best_feature;
        node->threshold = best_threshold;

        node->left =
            build(
                x,
                y,
                best_left,
                depth + 1
            );

        node->right =
            build(
                x,
                y,
                best_right,
                depth + 1
            );

        return node;
    }

    size_t max_depth_;
    size_t min_samples_split_;
    size_t max_features_;
    size_t feature_count_ = 0;

    std::mt19937 generator_;

    std::unique_ptr<Node> root_;

    bool fitted_ = false;
};

} // namespace ml