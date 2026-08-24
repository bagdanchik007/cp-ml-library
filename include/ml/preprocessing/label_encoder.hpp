#pragma once

#include <cstddef>
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

} // namespace ml