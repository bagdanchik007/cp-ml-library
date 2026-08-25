#pragma once

#include "ml/core/matrix/matrix.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
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

inline void OneHotEncoder::fit(
    const std::vector<int>& labels
) {
    if (labels.empty()) {
        throw std::invalid_argument(
            "OneHotEncoder::fit: input labels must not be empty"
        );
    }

    categories_ = labels;

    std::sort(
        categories_.begin(),
        categories_.end()
    );

    categories_.erase(
        std::unique(
            categories_.begin(),
            categories_.end()
        ),
        categories_.end()
    );

    fitted_ = true;
}

inline Matrix OneHotEncoder::transform(
    const std::vector<int>& labels
) const {
    if (!fitted_) {
        throw std::logic_error(
            "OneHotEncoder::transform: encoder has not been fitted"
        );
    }

    Matrix result(
        labels.size(),
        categories_.size()
    );

    for (size_t row = 0; row < labels.size(); ++row) {
        const auto it = std::lower_bound(
            categories_.begin(),
            categories_.end(),
            labels[row]
        );

        if (it == categories_.end() || *it != labels[row]) {
            throw std::invalid_argument(
                "OneHotEncoder::transform: unknown label"
            );
        }

        const size_t column =
            static_cast<size_t>(
                std::distance(
                    categories_.begin(),
                    it
                )
            );

        result(row, column) = 1.0;
    }

    return result;
}

inline Matrix OneHotEncoder::fit_transform(
    const std::vector<int>& labels
) {
    fit(labels);

    return transform(labels);
}

} // namespace ml