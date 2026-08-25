#pragma once

#include "ml/core/matrix/matrix.hpp"

#include <cstddef>
#include <functional>
#include <vector>

namespace ml {

using FeatureScoreFunction =
    std::function<double(
        const Matrix& data,
        size_t feature
    )>;

class SelectKBest {
public:
    SelectKBest(
        size_t k,
        FeatureScoreFunction score_function
    );

    void fit(
        const Matrix& data
    );

    Matrix transform(
        const Matrix& data
    ) const;

    Matrix fit_transform(
        const Matrix& data
    );

    const std::vector<double>& scores() const;

    const std::vector<size_t>& selected_features() const;

private:
    size_t k_;
    FeatureScoreFunction score_function_;

    std::vector<double> scores_;
    std::vector<size_t> selected_features_;

    bool fitted_ = false;
};

} // namespace ml