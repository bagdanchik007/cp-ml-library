#pragma once

#include "ml/core/matrix/matrix.hpp"

#include <cstddef>
#include <functional>
#include <vector>

namespace ml {

using UnivariateScoreFunction =
    std::function<double(
        const Matrix& data,
        std::size_t feature
    )>;

enum class SelectionMode {
    Percentile,
    KBest,
    Fpr,
    Fdr,
    Fwe
};

class GenericUnivariateSelect {
public:
    GenericUnivariateSelect(
        UnivariateScoreFunction score_function,
        SelectionMode mode,
        double param
    );

    void fit(const Matrix& data);

    Matrix transform(const Matrix& data) const;

    Matrix fit_transform(const Matrix& data);

    const std::vector<double>& scores() const;

    const std::vector<std::size_t>& selected_features() const;

private:
    void validate_input(const Matrix& data) const;

    UnivariateScoreFunction score_function_;
    SelectionMode mode_;
    double param_;

    std::vector<double> scores_;
    std::vector<std::size_t> selected_features_;

    bool fitted_ = false;
};

} // namespace ml
