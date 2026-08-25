#pragma once

#include "ml/core/matrix/matrix.hpp"

#include <cstddef>
#include <functional>
#include <vector>

namespace ml {

class GenericUnivariateSelector {
public:
    using ScoreFunction =
        std::function<std::vector<double>(
            const Matrix& X,
            const Matrix& y
        )>;

    explicit GenericUnivariateSelector(
        ScoreFunction score_function
    );

    void fit(
        const Matrix& X,
        const Matrix& y
    );

    Matrix transform(
        const Matrix& X
    ) const;

    Matrix fit_transform(
        const Matrix& X,
        const Matrix& y
    );

    const std::vector<std::size_t>&
    selected_features() const noexcept;

protected:
    ScoreFunction score_function_;
    std::vector<std::size_t> selected_features_;
};

} // namespace ml