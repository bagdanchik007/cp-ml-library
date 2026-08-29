#pragma once

/**
 * @file ml.hpp
 * @brief Main umbrella header for the Cpp ML Library.
 */

#include "core/matrix/matrix.hpp"
#include "core/matrix/matrix_utility.hpp"
#include "core/estimators/estimator.hpp"
#include "core/estimators/classifier.hpp"
#include "version.hpp"

#include "algorithms/linear_regression.hpp"
#include "algorithms/kmeans.hpp"
#include "algorithms/decision_tree.hpp"
#include "algorithms/random_forest.hpp"
#include "algorithms/dbscan.hpp"

#include "data/dataset.hpp"
#include "data/csv_reader.hpp"

#include "metrics/model_evaluation.hpp"

#include "model_selection/kfold.hpp"
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
