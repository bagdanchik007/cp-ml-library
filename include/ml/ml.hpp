#pragma once

/**
 * @file ml.hpp
 * @brief Main umbrella header for the Cpp ML Library.
 */

#include "core/matrix/matrix.hpp"
#include "core/matrix/matrix_utility.hpp"

#include "core/estimators/estimator.hpp"
#include "core/estimators/regressor.hpp"
#include "core/estimators/classifier.hpp"

#include "version.hpp"

#include "algorithms/linear_regression.hpp"
#include "algorithms/kmeans.hpp"
#include "algorithms/decision_tree.hpp"
#include "algorithms/random_forest.hpp"
#include "algorithms/dbscan.hpp"
#include "algorithms/agglomerative_clustering.hpp"
#include "algorithms/knn_classifier.hpp"
#include "algorithms/weighted_knn_classifier.hpp"
#include "algorithms/gaussian_naive_bayes.hpp"
#include "algorithms/multinomial_naive_bayes.hpp"
#include "algorithms/logistic_regression.hpp"
#include "algorithms/ridge_regression.hpp"
#include "algorithms/perceptron.hpp"
#include "algorithms/softmax_regression.hpp"
#include "algorithms/lasso_regression.hpp"
#include "algorithms/elastic_net.hpp"
#include "algorithms/knn_regressor.hpp"
#include "algorithms/dummy_classifier.hpp"
#include "algorithms/dummy_regressor.hpp"

#include "data/dataset.hpp"
#include "data/csv_reader.hpp"

#include "metrics/model_evaluation.hpp"

#include "model_selection/kfold.hpp"
#include "model_selection/stratified_kfold.hpp"
#include "model_selection/leave_one_out.hpp"
#include "model_selection/repeated_kfold.hpp"
#include "model_selection/shuffle_split.hpp"
#include "model_selection/randomized_search.hpp"
#include "model_selection/cross_validation.hpp"
#include "model_selection/grid_search.hpp"

#include "pipeline/pipeline.hpp"

#include "preprocessing/standard_scaler.hpp"
#include "preprocessing/min_max_scaler.hpp"
#include "preprocessing/normalizer.hpp"
#include "preprocessing/robust_scaler.hpp"
#include "preprocessing/one_hot_encoder.hpp"
#include "preprocessing/simple_imputer.hpp"
#include "preprocessing/train_test_split.hpp"
#include "preprocessing/k_bins_discretizer.hpp"
#include "preprocessing/log1p_transformer.hpp"
#include "preprocessing/ordinal_encoder.hpp"
#include "preprocessing/label_binarizer.hpp"
#include "preprocessing/function_transformer.hpp"
