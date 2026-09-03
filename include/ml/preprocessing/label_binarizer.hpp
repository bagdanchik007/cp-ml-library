#pragma once

#include "ml/core/matrix/matrix.hpp"

#include <algorithm>
#include <set>
#include <stdexcept>
#include <vector>

namespace ml {

/**
 * @brief Converts a vector of integer class labels into a one-hot encoded
 * Matrix, e.g. for use as targets in a softmax/cross-entropy training loop.
 */
class LabelBinarizer {
public:
    LabelBinarizer() = default;

    void fit(const std::vector<int>& y) {
        if (y.empty()) throw std::invalid_argument("LabelBinarizer::fit: y must not be empty");
        const std::set<int> unique_labels(y.begin(), y.end());
        classes_.assign(unique_labels.begin(), unique_labels.end());
        fitted_ = true;
    }

    Matrix transform(const std::vector<int>& y) const {
        if (!fitted_) throw std::logic_error("LabelBinarizer::transform: binarizer has not been fitted");

        Matrix result(y.size(), classes_.size(), 0.0);
        for (size_t i = 0; i < y.size(); ++i) {
            const auto it = std::find(classes_.begin(), classes_.end(), y[i]);
            if (it == classes_.end()) throw std::invalid_argument("LabelBinarizer::transform: unknown class label encountered");
            result(i, static_cast<size_t>(std::distance(classes_.begin(), it))) = 1.0;
        }
        return result;
    }

    Matrix fit_transform(const std::vector<int>& y) {
        fit(y);
        return transform(y);
    }

    std::vector<int> inverse_transform(const Matrix& one_hot) const {
        if (!fitted_) throw std::logic_error("LabelBinarizer::inverse_transform: binarizer has not been fitted");
        if (one_hot.cols != classes_.size()) throw std::invalid_argument("LabelBinarizer::inverse_transform: column count mismatch");

        std::vector<int> labels;
        labels.reserve(one_hot.rows);
        for (size_t i = 0; i < one_hot.rows; ++i) {
            size_t best_column = 0;
            double best_value = one_hot(i, 0);
            for (size_t j = 1; j < one_hot.cols; ++j) {
                if (one_hot(i, j) > best_value) {
                    best_value = one_hot(i, j);
                    best_column = j;
                }
            }
            labels.push_back(classes_[best_column]);
        }
        return labels;
    }

    const std::vector<int>& classes() const { return classes_; }
    bool is_fitted() const { return fitted_; }

private:
    std::vector<int> classes_;
    bool fitted_ = false;
};

} // namespace ml
