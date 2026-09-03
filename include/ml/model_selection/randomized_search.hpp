#pragma once

#include "ml/model_selection/cross_validation.hpp"

#include <algorithm>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ml {

/**
 * @brief Randomized hyperparameter search: evaluates a fixed number of
 * randomly sampled candidates via cross-validation, instead of the
 * exhaustive grid covered by GridSearch.
 *
 * Useful when the candidate space is large, since it lets you trade
 * search thoroughness for compute budget by choosing n_iterations.
 */
template <typename Parameter, typename EstimatorFactory, typename Scorer>
class RandomizedSearch {
public:
    RandomizedSearch(
        std::vector<Parameter> candidates,
        EstimatorFactory factory,
        KFold splitter,
        Scorer scorer,
        size_t n_iterations = 10,
        unsigned seed = 42,
        bool greater_is_better = true
    )
        : candidates_(std::move(candidates)),
          factory_(std::move(factory)),
          splitter_(std::move(splitter)),
          scorer_(std::move(scorer)),
          n_iterations_(n_iterations),
          seed_(seed),
          greater_is_better_(greater_is_better) {
        if (candidates_.empty()) throw std::invalid_argument("RandomizedSearch: candidates must not be empty");
        if (n_iterations_ == 0) throw std::invalid_argument("RandomizedSearch: n_iterations must be at least 1");
    }

    void fit(const Matrix& features, const std::vector<double>& targets) {
        best_score_ = greater_is_better_
            ? -std::numeric_limits<double>::infinity()
            : std::numeric_limits<double>::infinity();

        std::mt19937 generator(seed_);
        std::uniform_int_distribution<size_t> distribution(0, candidates_.size() - 1);

        const size_t trials = std::min(n_iterations_, candidates_.size());
        std::vector<size_t> tried_indices;
        tried_indices.reserve(trials);

        for (size_t trial = 0; trial < trials; ++trial) {
            size_t index;
            do {
                index = distribution(generator);
            } while (std::find(tried_indices.begin(), tried_indices.end(), index) != tried_indices.end());
            tried_indices.push_back(index);

            const auto& candidate = candidates_[index];
            const auto scores = cross_validate([&] { return factory_(candidate); }, features, targets, splitter_, scorer_);
            const double score = mean_score(scores);

            if ((greater_is_better_ && score > best_score_) || (!greater_is_better_ && score < best_score_)) {
                best_score_ = score;
                best_parameter_ = candidate;
            }
        }
        fitted_ = true;
    }

    const Parameter& best_parameter() const {
        if (!fitted_) throw std::logic_error("RandomizedSearch: fit must be called first");
        return best_parameter_;
    }

    double best_score() const {
        if (!fitted_) throw std::logic_error("RandomizedSearch: fit must be called first");
        return best_score_;
    }

private:
    std::vector<Parameter> candidates_;
    EstimatorFactory factory_;
    KFold splitter_;
    Scorer scorer_;
    size_t n_iterations_;
    unsigned seed_;
    bool greater_is_better_;
    Parameter best_parameter_{};
    double best_score_{};
    bool fitted_ = false;
};

} // namespace ml
