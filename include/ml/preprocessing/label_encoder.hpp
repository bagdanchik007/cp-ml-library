#pragma once

#include <cstddef>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace ml {

class LabelEncoder {
public:
    LabelEncoder() = default;

    void fit(
        const std::vector<int>& labels
    );

    std::vector<int> transform(
        const std::vector<int>& labels
    ) const;

    std::vector<int> fit_transform(
        const std::vector<int>& labels
    );

    std::vector<int> inverse_transform(
        const std::vector<int>& encoded
    ) const;

private:
    std::vector<int> classes_;
    std::unordered_map<int, std::size_t> label_to_index_;
    bool fitted_ = false;
};

inline void LabelEncoder::fit(
    const std::vector<int>& labels
) {
    if (labels.empty()) {
        throw std::invalid_argument(
            "LabelEncoder::fit: input labels must not be empty"
        );
    }

    classes_.clear();
    label_to_index_.clear();

    for (const int label : labels) {
        if (label_to_index_.find(label) ==
            label_to_index_.end()) {
            const std::size_t index = classes_.size();

            classes_.push_back(label);
            label_to_index_[label] = index;
        }
    }

    fitted_ = true;
}

inline std::vector<int> LabelEncoder::transform(
    const std::vector<int>& labels
) const {
    if (!fitted_) {
        throw std::logic_error(
            "LabelEncoder::transform: encoder has not been fitted"
        );
    }

    std::vector<int> result;
    result.reserve(labels.size());

    for (const int label : labels) {
        const auto it = label_to_index_.find(label);

        if (it == label_to_index_.end()) {
            throw std::invalid_argument(
                "LabelEncoder::transform: unknown label"
            );
        }

        result.push_back(
            static_cast<int>(it->second)
        );
    }

    return result;
}

inline std::vector<int> LabelEncoder::fit_transform(
    const std::vector<int>& labels
) {
    fit(labels);

    return transform(labels);
}

inline std::vector<int> LabelEncoder::inverse_transform(
    const std::vector<int>& encoded
) const {
    if (!fitted_) {
        throw std::logic_error(
            "LabelEncoder::inverse_transform: encoder has not been fitted"
        );
    }

    std::vector<int> result;
    result.reserve(encoded.size());

    for (const int index : encoded) {
        if (index < 0 ||
            static_cast<std::size_t>(index) >= classes_.size()) {
            throw std::invalid_argument(
                "LabelEncoder::inverse_transform: invalid encoded label"
            );
        }

        result.push_back(
            classes_[static_cast<std::size_t>(index)]
        );
    }

    return result;
}

} // namespace ml