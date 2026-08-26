#pragma once

#include "ml/core/matrix/matrix.hpp"

#include <cstddef>
#include <vector>

namespace ml {

class SelectFromModel {
public:
    explicit SelectFromModel(
        double threshold
    );

    void fit(
        const std::vector<double>& scores
    );

    Matrix transform(
        const Matrix& data
    ) const;

    Matrix fit_transform(
        const Matrix& data,
        const std::vector<double>& scores
    );

    const std::vector<std::size_t>&
    selected_features() const;

private:
    double threshold_;

    std::vector<std::size_t>
        selected_features_;

    bool fitted_ = false;
};

} // namespace ml
