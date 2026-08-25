#pragma once

#include "ml/core/matrix/matrix.hpp"

#include <utility>
#include <algorithm>
#include <cstddef>
#include <functional>
#include <stdexcept>
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
    )
        : k_(k),
          score_function_(std::move(score_function)) {

        if (k_ == 0) {
            throw std::invalid_argument(
                "SelectKBest: k must be greater than zero"
            );
        }

        if (!score_function_) {
            throw std::invalid_argument(
                "SelectKBest: score function must be valid"
            );
        }
    }

    void fit(
        const Matrix& data
    ) {
        validate_input(data);

        if (k_ > data.cols) {
            throw std::invalid_argument(
                "SelectKBest: k cannot exceed the number of features"
            );
        }

        scores_.clear();
        selected_features_.clear();

        scores_.reserve(data.cols);

        for (size_t feature = 0;
             feature < data.cols;
             ++feature) {

            const double score =
                score_function_(data, feature);

            scores_.push_back(score);
        }

        std::vector<size_t> indices(data.cols);

        for (size_t feature = 0;
             feature < data.cols;
             ++feature) {

            indices[feature] = feature;
        }

        std::stable_sort(
            indices.begin(),
            indices.end(),
            [this](size_t lhs, size_t rhs) {
                return scores_[lhs] > scores_[rhs];
            }
        );

        selected_features_.assign(
            indices.begin(),
            indices.begin() + k_
        );

        std::sort(
            selected_features_.begin(),
            selected_features_.end()
        );

        fitted_ = true;
    }

    Matrix transform(
        const Matrix& data
    ) const {
        if (!fitted_) {
            throw std::logic_error(
                "SelectKBest::transform: "
                "selector has not been fitted"
            );
        }

        validate_input(data);

        if (data.cols != scores_.size()) {
            throw std::invalid_argument(
                "SelectKBest::transform: "
                "input column count does not match fitted data"
            );
        }

        Matrix result(
            data.rows,
            selected_features_.size()
        );

        for (size_t row = 0;
             row < data.rows;
             ++row) {

            for (size_t output_column = 0;
                 output_column < selected_features_.size();
                 ++output_column) {

                const size_t input_column =
                    selected_features_[output_column];

                result(row, output_column) =
                    data(row, input_column);
            }
        }

        return result;
    }

    Matrix fit_transform(
        const Matrix& data
    ) {
        fit(data);
        return transform(data);
    }

    const std::vector<double>& scores() const {
        if (!fitted_) {
            throw std::logic_error(
                "SelectKBest::scores: "
                "selector has not been fitted"
            );
        }

        return scores_;
    }

    const std::vector<size_t>& selected_features() const {
        if (!fitted_) {
            throw std::logic_error(
                "SelectKBest::selected_features: "
                "selector has not been fitted"
            );
        }

        return selected_features_;
    }

private:
    static void validate_input(
        const Matrix& data
    ) {
        if (data.rows == 0 || data.cols == 0) {
            throw std::invalid_argument(
                "SelectKBest: input data must not be empty"
            );
        }
    }

private:
    size_t k_;
    FeatureScoreFunction score_function_;

    std::vector<double> scores_;
    std::vector<size_t> selected_features_;

    bool fitted_ = false;
};

} // namespace ml