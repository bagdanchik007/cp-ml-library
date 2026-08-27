#pragma once

#include "ml/model_selection/cross_validation.hpp"

#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ml {

/** Generic grid search; factory maps a candidate parameter to a new estimator. */
template <typename Parameter, typename EstimatorFactory, typename Scorer>
class GridSearch {
public:
    GridSearch(std::vector<Parameter> candidates, EstimatorFactory factory, KFold splitter, Scorer scorer, bool greater_is_better = true)
        : candidates_(std::move(candidates)), factory_(std::move(factory)), splitter_(std::move(splitter)), scorer_(std::move(scorer)), greater_is_better_(greater_is_better) {
        if (candidates_.empty()) throw std::invalid_argument("GridSearch: candidates must not be empty");
    }
    void fit(const Matrix& features, const std::vector<double>& targets) {
        best_score_ = greater_is_better_ ? -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::infinity();
        for (const auto& candidate : candidates_) {
            const auto scores = cross_validate([&] { return factory_(candidate); }, features, targets, splitter_, scorer_);
            const double score = mean_score(scores);
            if ((greater_is_better_ && score > best_score_) || (!greater_is_better_ && score < best_score_)) { best_score_ = score; best_parameter_ = candidate; }
        }
        fitted_ = true;
    }
    const Parameter& best_parameter() const { if (!fitted_) throw std::logic_error("GridSearch: fit must be called first"); return best_parameter_; }
    double best_score() const { if (!fitted_) throw std::logic_error("GridSearch: fit must be called first"); return best_score_; }
private:
    std::vector<Parameter> candidates_; EstimatorFactory factory_; KFold splitter_; Scorer scorer_; bool greater_is_better_;
    Parameter best_parameter_{}; double best_score_{}; bool fitted_ = false;
};

} // namespace ml
