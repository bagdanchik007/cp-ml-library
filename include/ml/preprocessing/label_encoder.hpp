#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace ml {

class LabelEncoder {
public:
    LabelEncoder() = default;

    void fit(
        const std::vector<std::string>& labels
    );

    std::vector<size_t> transform(
        const std::vector<std::string>& labels
    ) const;

    std::vector<size_t> fit_transform(
        const std::vector<std::string>& labels
    );

    std::vector<std::string> inverse_transform(
        const std::vector<size_t>& encoded_labels
    ) const;

private:
    std::unordered_map<std::string, size_t> label_to_index_;
    std::vector<std::string> index_to_label_;
    bool fitted_ = false;
};

inline void LabelEncoder::fit(
    const std::vector<std::string>& labels
) {
    if (labels.empty()) {
        throw std::invalid_argument(
            "LabelEncoder::fit: labels must not be empty"
        );
    }

    label_to_index_.clear();
    index_to_label_.clear();

    for (const std::string& label : labels) {
        if (label_to_index_.find(label) ==
            label_to_index_.end()) {

            const size_t index =
                index_to_label_.size();

            label_to_index_[label] = index;
            index_to_label_.push_back(label);
        }
    }

    fitted_ = true;
}

inline std::vector<size_t> LabelEncoder::transform(
    const std::vector<std::string>& labels
) const {
    if (!fitted_) {
        throw std::logic_error(
            "LabelEncoder::transform: encoder has not been fitted"
        );
    }

    std::vector<size_t> result;
    result.reserve(labels.size());

    for (const std::string& label : labels) {
        const auto iterator =
            label_to_index_.find(label);

        if (iterator == label_to_index_.end()) {
            throw std::invalid_argument(
                "LabelEncoder::transform: unknown label"
            );
        }

        result.push_back(iterator->second);
    }

    return result;
}

inline std::vector<size_t> LabelEncoder::fit_transform(
    const std::vector<std::string>& labels
) {
    fit(labels);

    return transform(labels);
}

inline std::vector<std::string>
LabelEncoder::inverse_transform(
    const std::vector<size_t>& encoded_labels
) const {
    if (!fitted_) {
        throw std::logic_error(
            "LabelEncoder::inverse_transform: encoder has not been fitted"
        );
    }

    std::vector<std::string> result;
    result.reserve(encoded_labels.size());

    for (const size_t index : encoded_labels) {
        if (index >= index_to_label_.size()) {
            throw std::invalid_argument(
                "LabelEncoder::inverse_transform: invalid encoded label"
            );
        }

        result.push_back(
            index_to_label_[index]
        );
    }

    return result;
}

} // namespace ml