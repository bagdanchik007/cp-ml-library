#pragma once

#include "ml/core/matrix/matrix.hpp"

#include <cstddef>
#include <vector>

namespace ml {

class OneHotEncoder {
public:
    OneHotEncoder() = default;

    void fit(
        const std::vector<int>& labels
    );

    Matrix transform(
        const std::vector<int>& labels
    ) const;

    Matrix fit_transform(
        const std::vector<int>& labels
    );

private:
    std::vector<int> categories_;
    bool fitted_ = false;
};

} // namespace ml