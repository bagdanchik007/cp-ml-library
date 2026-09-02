#pragma once

#include <algorithm>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace ml {

/**
 * @brief Baseline classifier that ignores input features entirely.
 *
 * Useful as a sanity check: any real model should clearly outperform
 * this. Supports three strategies:
 *   - "most_frequent": always predicts the most common training label
 *   - "stratified": predicts randomly, respecting training class frequencies
 *   - "uniform": predicts uniformly at random among observed classes
 */
class DummyClassifier {
public:
    explicit DummyClassifier(std::string strategy = "most_frequent", unsigned seed = 42)
        : strategy_(std::move(strategy)), generator_(seed) {
        if (strategy_ != "most_frequent" && strategy_ != "stratified" && strategy_ != "uniform") {
            throw std::invalid_argument("DummyClassifier: strategy must be 'most_frequent', 'stratified' or 'uniform'");
        }
    }

    void fit(const std::vector<int>& y) {
        if (y.empty()) throw std::invalid_argument("DummyClassifier::fit: y must not be empty");

        std::map<int, size_t> counts;
        for (int label : y) ++counts[label];

        classes_.clear();
        weights_.clear();
        for (const auto& [label, count] : counts) {
            classes_.push_back(label);
            weights_.push_back(static_cast<double>(count));
        }

        most_frequent_ = classes_.front();
        size_t best_count = 0;
        for (const auto& [label, count] : counts) {
            if (count > best_count) {
                best_count = count;
                most_frequent_ = label;
            }
        }
        fitted_ = true;
    }

    std::vector<int> predict(size_t n_samples) {
        if (!fitted_) throw std::runtime_error("DummyClassifier::predict: model is not fitted");

        std::vector<int> predictions;
        predictions.reserve(n_samples);

        if (strategy_ == "most_frequent") {
            predictions.assign(n_samples, most_frequent_);
        } else if (strategy_ == "uniform") {
            std::uniform_int_distribution<size_t> distribution(0, classes_.size() - 1);
            for (size_t i = 0; i < n_samples; ++i) predictions.push_back(classes_[distribution(generator_)]);
        } else { // stratified
            std::discrete_distribution<size_t> distribution(weights_.begin(), weights_.end());
            for (size_t i = 0; i < n_samples; ++i) predictions.push_back(classes_[distribution(generator_)]);
        }
        return predictions;
    }

    bool is_fitted() const { return fitted_; }

private:
    std::string strategy_;
    std::mt19937 generator_;
    std::vector<int> classes_;
    std::vector<double> weights_;
    int most_frequent_ = 0;
    bool fitted_ = false;
};

} // namespace ml
