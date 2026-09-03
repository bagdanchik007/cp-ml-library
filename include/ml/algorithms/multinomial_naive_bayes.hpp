#pragma once

#include "ml/core/matrix/matrix.hpp"

#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <vector>

namespace ml {

/**
 * @brief Multinomial Naive Bayes classifier, suited for count-based
 * features (e.g. word frequencies in text classification), unlike
 * GaussianNaiveBayes which assumes continuous, normally distributed features.
 *
 * Uses Laplace (add-one) smoothing to avoid zero probabilities for
 * features unseen in a given class during training.
 */
class MultinomialNaiveBayes {
public:
    explicit MultinomialNaiveBayes(double alpha = 1.0) : alpha_(alpha) {
        if (alpha_ < 0.0) throw std::invalid_argument("MultinomialNaiveBayes: alpha must be non-negative");
    }

    void fit(const Matrix& X, const std::vector<int>& y) {
        if (X.rows != y.size() || X.rows == 0) {
            throw std::invalid_argument("MultinomialNaiveBayes::fit: X and y must be non-empty with matching row counts");
        }

        const size_t n_features = X.cols;
        std::map<int, double> class_totals;
        std::map<int, std::vector<double>> feature_counts;
        std::map<int, double> feature_count_sum;

        for (size_t i = 0; i < X.rows; ++i) {
            const int label = y[i];
            class_totals[label] += 1.0;
            if (feature_counts.find(label) == feature_counts.end()) {
                feature_counts[label] = std::vector<double>(n_features, 0.0);
                feature_count_sum[label] = 0.0;
            }
            for (size_t j = 0; j < n_features; ++j) {
                if (X(i, j) < 0.0) throw std::invalid_argument("MultinomialNaiveBayes::fit: features must be non-negative counts");
                feature_counts[label][j] += X(i, j);
                feature_count_sum[label] += X(i, j);
            }
        }

        const auto n_samples = static_cast<double>(X.rows);
        log_prior_.clear();
        log_likelihood_.clear();

        for (const auto& [label, total] : class_totals) {
            log_prior_[label] = std::log(total / n_samples);

            std::vector<double> likelihoods(n_features, 0.0);
            const double denominator = feature_count_sum[label] + alpha_ * static_cast<double>(n_features);
            for (size_t j = 0; j < n_features; ++j) {
                likelihoods[j] = std::log((feature_counts[label][j] + alpha_) / denominator);
            }
            log_likelihood_[label] = std::move(likelihoods);
        }
        fitted_ = true;
    }

    std::vector<int> predict(const Matrix& X) const {
        if (!fitted_) throw std::runtime_error("MultinomialNaiveBayes::predict: model is not fitted");

        std::vector<int> predictions;
        predictions.reserve(X.rows);
        for (size_t i = 0; i < X.rows; ++i) predictions.push_back(predict_single(X.row(i)));
        return predictions;
    }

    bool is_fitted() const { return fitted_; }

private:
    double alpha_;
    std::map<int, double> log_prior_;
    std::map<int, std::vector<double>> log_likelihood_;
    bool fitted_ = false;

    int predict_single(const std::vector<double>& sample) const {
        int best_label = log_prior_.begin()->first;
        double best_score = -std::numeric_limits<double>::infinity();

        for (const auto& [label, prior] : log_prior_) {
            double score = prior;
            const auto& likelihoods = log_likelihood_.at(label);
            for (size_t j = 0; j < sample.size(); ++j) score += sample[j] * likelihoods[j];

            if (score > best_score) {
                best_score = score;
                best_label = label;
            }
        }
        return best_label;
    }
};

} // namespace ml
