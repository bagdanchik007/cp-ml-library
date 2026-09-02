#pragma once

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace ml {

/**
 * @brief Encodes string-valued categorical columns as integer codes.
 *
 * Unlike LabelEncoder (single column of int labels), OrdinalEncoder
 * operates on a table of string columns, fitting an independent
 * category mapping per column, matching the common tabular preprocessing
 * pattern of encoding several categorical features at once.
 */
class OrdinalEncoder {
public:
    OrdinalEncoder() = default;

    void fit(const std::vector<std::vector<std::string>>& columns) {
        if (columns.empty()) throw std::invalid_argument("OrdinalEncoder::fit: input must have at least one column");

        category_to_code_.assign(columns.size(), {});
        code_to_category_.assign(columns.size(), {});

        for (size_t column = 0; column < columns.size(); ++column) {
            for (const auto& value : columns[column]) {
                if (category_to_code_[column].find(value) == category_to_code_[column].end()) {
                    const int code = static_cast<int>(code_to_category_[column].size());
                    category_to_code_[column][value] = code;
                    code_to_category_[column].push_back(value);
                }
            }
        }
        fitted_ = true;
    }

    std::vector<std::vector<int>> transform(const std::vector<std::vector<std::string>>& columns) const {
        if (!fitted_) throw std::logic_error("OrdinalEncoder::transform: encoder has not been fitted");
        if (columns.size() != category_to_code_.size()) {
            throw std::invalid_argument("OrdinalEncoder::transform: column count mismatch");
        }

        std::vector<std::vector<int>> result(columns.size());
        for (size_t column = 0; column < columns.size(); ++column) {
            result[column].reserve(columns[column].size());
            for (const auto& value : columns[column]) {
                const auto it = category_to_code_[column].find(value);
                if (it == category_to_code_[column].end()) {
                    throw std::invalid_argument("OrdinalEncoder::transform: unknown category encountered");
                }
                result[column].push_back(it->second);
            }
        }
        return result;
    }

    std::vector<std::vector<int>> fit_transform(const std::vector<std::vector<std::string>>& columns) {
        fit(columns);
        return transform(columns);
    }

    size_t n_categories(size_t column) const { return code_to_category_.at(column).size(); }
    bool is_fitted() const { return fitted_; }

private:
    std::vector<std::unordered_map<std::string, int>> category_to_code_;
    std::vector<std::vector<std::string>> code_to_category_;
    bool fitted_ = false;
};

} // namespace ml
