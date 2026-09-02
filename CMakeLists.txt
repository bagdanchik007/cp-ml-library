cmake_minimum_required(VERSION 3.16)

project(
    CppMLLibrary
    VERSION 1.1.0
    LANGUAGES CXX
)

# -------------------------------------------------
# C++ standard
# -------------------------------------------------

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ML_BUILD_BENCHMARKS "Build local benchmark executables" OFF)

# -------------------------------------------------
# Library
# -------------------------------------------------

add_library(
    ml_library
    src/algorithms/linear_regression.cpp
    src/algorithms/kmeans.cpp
    src/feature_selection/f_regression.cpp
    src/feature_selection/select_percentile.cpp
    src/feature_selection/select_fpr.cpp
    src/feature_selection/select_fdr.cpp
    src/feature_selection/chi2.cpp
    src/feature_selection/generic_univariate_selector.cpp
    src/feature_selection/mutual_info.cpp
    src/feature_selection/select_from_model.cpp
)

target_include_directories(
    ml_library
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# -------------------------------------------------
# Demo executable
# -------------------------------------------------

add_executable(
    ml_demo
    examples/demo.cpp
    examples/linear_regression_demo.cpp
    examples/kmeans_demo.cpp
    examples/preprocessing/preprocessing_demo.cpp
    examples/feature_selection/feature_selection_demo.cpp
    examples/new_algorithms_demo.cpp
    examples/more_algorithms_demo.cpp
)

target_link_libraries(
    ml_demo
    PRIVATE
        ml_library
)

# -------------------------------------------------
# Testing
# -------------------------------------------------

enable_testing()

# -------------------------------------------------
# Helper function for test targets
# -------------------------------------------------

function(add_ml_test TARGET_NAME SOURCE_FILE)

    add_executable(
        ${TARGET_NAME}
        ${SOURCE_FILE}
    )

    target_link_libraries(
        ${TARGET_NAME}
        PRIVATE
            ml_library
    )

    # Keep assert-based checks active in Release test builds
    target_compile_options(
        ${TARGET_NAME}
        PRIVATE
            $<$<CXX_COMPILER_ID:MSVC>:/UNDEBUG>
            $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-UNDEBUG>
    )

    add_test(
        NAME ${TARGET_NAME}
        COMMAND ${TARGET_NAME}
    )

endfunction()

# -------------------------------------------------
# Matrix tests
# -------------------------------------------------

add_ml_test(
    matrix_tests
    tests/core/matrix/matrix_tests.cpp
)

add_ml_test(
    matrix_arithmetic_tests
    tests/core/matrix/matrix_arithmetic_tests.cpp
)

add_ml_test(
    matrix_utility_tests
    tests/core/matrix/matrix_utility_tests.cpp
)

add_ml_test(
    matrix_validation_tests
    tests/core/matrix/matrix_validation_tests.cpp
)

add_ml_test(
    matrix_transformation_tests
    tests/core/matrix/matrix_transformation_tests.cpp
)

add_ml_test(
    matrix_statistics_tests
    tests/core/matrix/matrix_statistics_tests.cpp
)

add_ml_test(
    matrix_norm_tests
    tests/core/matrix/matrix_norm_tests.cpp
)

add_ml_test(
    matrix_column_tests
    tests/core/matrix/matrix_column_tests.cpp
)

add_ml_test(
    matrix_row_tests
    tests/core/matrix/matrix_row_tests.cpp
)

add_ml_test(
    matrix_comparison_tests
    tests/core/matrix/matrix_comparison_tests.cpp
)

add_ml_test(
    matrix_scalar_tests
    tests/core/matrix/matrix_scalar_tests.cpp
)

add_ml_test(
    matrix_linear_algebra_tests
    tests/core/matrix/matrix_linear_algebra_tests.cpp
)

add_ml_test(
    matrix_decomposition_tests
    tests/core/matrix/matrix_decomposition_tests.cpp
)

add_ml_test(
    matrix_solve_tests
    tests/core/matrix/matrix_solve_tests.cpp
)

# -------------------------------------------------
# Estimator tests
# -------------------------------------------------

add_ml_test(
    estimator_tests
    tests/core/estimators/estimator_tests.cpp
)

add_ml_test(
    classifier_tests
    tests/core/estimators/classifier_tests.cpp
)

add_ml_test(
    regressor_tests
    tests/core/estimators/regressor_tests.cpp
)
# -------------------------------------------------
# Algorithms tests
# -------------------------------------------------

add_ml_test(
    linear_regression_tests
    tests/algorithms/linear_regression_tests.cpp
)

add_ml_test(
    kmeans_tests
    tests/algorithms/kmeans_tests.cpp
)

add_ml_test(
    tree_ensemble_tests
    tests/algorithms/tree_ensemble_tests.cpp
)

add_ml_test(
    dbscan_tests
    tests/algorithms/dbscan_tests.cpp
)

add_ml_test(
    knn_classifier_tests
    tests/algorithms/knn_classifier_tests.cpp
)

add_ml_test(
    gaussian_naive_bayes_tests
    tests/algorithms/gaussian_naive_bayes_tests.cpp
)

add_ml_test(
    logistic_regression_tests
    tests/algorithms/logistic_regression_tests.cpp
)

add_ml_test(
    ridge_regression_tests
    tests/algorithms/ridge_regression_tests.cpp
)

add_ml_test(
    perceptron_tests
    tests/algorithms/perceptron_tests.cpp
)

add_ml_test(
    softmax_regression_tests
    tests/algorithms/softmax_regression_tests.cpp
)

add_ml_test(
    lasso_regression_tests
    tests/algorithms/lasso_regression_tests.cpp
)

add_ml_test(
    knn_regressor_tests
    tests/algorithms/knn_regressor_tests.cpp
)

add_ml_test(
    dummy_classifier_tests
    tests/algorithms/dummy_classifier_tests.cpp
)

add_ml_test(
    dummy_regressor_tests
    tests/algorithms/dummy_regressor_tests.cpp
)

# -------------------------------------------------
# Preprocessing tests
# -------------------------------------------------

add_ml_test(
    normalizer_tests
    tests/preprocessing/normalizer_tests.cpp
)

add_ml_test(
    standard_scaler_tests
    tests/preprocessing/standard_scaler_tests.cpp
)

add_ml_test(
    min_max_scaler_tests
    tests/preprocessing/min_max_scaler_tests.cpp
)

add_ml_test(
    robust_scaler_tests
    tests/preprocessing/robust_scaler_tests.cpp
)

add_ml_test(
    one_hot_encoder_tests
    tests/preprocessing/one_hot_encoder_tests.cpp
)

add_ml_test(
    train_test_split_tests
    tests/preprocessing/train_test_split_tests.cpp
)

add_ml_test(
    k_bins_discretizer_tests
    tests/preprocessing/k_bins_discretizer_tests.cpp
)

add_ml_test(
    log1p_transformer_tests
    tests/preprocessing/log1p_transformer_tests.cpp
)

add_ml_test(
    ordinal_encoder_tests
    tests/preprocessing/ordinal_encoder_tests.cpp
)

add_ml_test(
    label_encoder_tests
    tests/preprocessing/label_encoder_tests.cpp
)

add_ml_test(
    polynomial_features_tests
    tests/preprocessing/polynomial_features_tests.cpp
)

add_ml_test(
    simple_imputer_tests
    tests/preprocessing/simple_imputer_tests.cpp
)

add_ml_test(
    binarizer_tests
    tests/preprocessing/binarizer_tests.cpp
)

add_ml_test(
    max_abs_scaler_tests
    tests/preprocessing/max_abs_scaler_tests.cpp
)

# -------------------------------------------------
# Feature selection tests
# -------------------------------------------------

add_ml_test(
    variance_threshold_tests
    tests/feature_selection/variance_threshold_tests.cpp
)

add_ml_test(
    select_k_best_tests
    tests/feature_selection/select_k_best_tests.cpp
)

add_ml_test(
    f_regression_tests
    tests/feature_selection/f_regression_tests.cpp
)

add_ml_test(
    select_percentile_tests
    tests/feature_selection/select_percentile_tests.cpp
)

add_ml_test(
    select_fpr_tests
    tests/feature_selection/select_fpr_tests.cpp
)

add_ml_test(
    select_fdr_tests
    tests/feature_selection/select_fdr_tests.cpp
)

add_ml_test(
    mutual_info_tests
    tests/feature_selection/mutual_info_tests.cpp
)

add_ml_test(
    chi2_tests
    tests/feature_selection/chi2_tests.cpp
)

add_ml_test(
    generic_univariate_selector_tests
    tests/feature_selection/generic_univariate_selector_tests.cpp
)

# -------------------------------------------------
# API tests
# -------------------------------------------------

add_ml_test(
    umbrella_header_tests
    tests/api/umbrella_header_tests.cpp
)

# -------------------------------------------------
# Data and metrics tests
# -------------------------------------------------

add_ml_test(
    dataset_tests
    tests/data/dataset_tests.cpp
)

add_ml_test(
    model_evaluation_tests
    tests/metrics/model_evaluation_tests.cpp
)

add_ml_test(
    classification_metrics_tests
    tests/metrics/classification_metrics_tests.cpp
)

add_ml_test(
    silhouette_score_tests
    tests/metrics/silhouette_score_tests.cpp
)

add_ml_test(
    imbalanced_metrics_tests
    tests/metrics/imbalanced_metrics_tests.cpp
)

add_ml_test(
    regression_extra_metrics_tests
    tests/metrics/regression_extra_metrics_tests.cpp
)

add_ml_test(
    model_selection_tests
    tests/model_selection/model_selection_tests.cpp
)

add_ml_test(
    stratified_kfold_tests
    tests/model_selection/stratified_kfold_tests.cpp
)

add_ml_test(
    leave_one_out_tests
    tests/model_selection/leave_one_out_tests.cpp
)

add_ml_test(
    pipeline_tests
    tests/pipeline/pipeline_tests.cpp
)

add_ml_test(
    train_test_integration_tests
    tests/integration/train_test_integration_tests.cpp
)

# -------------------------------------------------
# Compiler warnings
# -------------------------------------------------

if(MSVC)

    set(
        ML_WARNING_FLAGS
        /W4
    )

else()

    set(
        ML_WARNING_FLAGS
        -Wall
        -Wextra
        -Wpedantic
    )

endif()

# -------------------------------------------------
# Apply warnings to all project targets
# -------------------------------------------------

set(
    ML_TARGETS

    # Library
    ml_library

    # Demo
    ml_demo

    # Estimator tests
    classifier_tests

    # Matrix tests
    matrix_tests
    matrix_arithmetic_tests
    matrix_utility_tests
    matrix_validation_tests
    matrix_transformation_tests
    matrix_statistics_tests
    matrix_norm_tests
    matrix_column_tests
    matrix_row_tests
    matrix_comparison_tests
    matrix_scalar_tests
    matrix_linear_algebra_tests
    matrix_decomposition_tests
    matrix_solve_tests

    # Algorithm tests
    linear_regression_tests
    kmeans_tests
    tree_ensemble_tests
    dbscan_tests

    # Preprocessing tests
    normalizer_tests
    standard_scaler_tests
    min_max_scaler_tests
    robust_scaler_tests
    one_hot_encoder_tests
    train_test_split_tests
    label_encoder_tests
    polynomial_features_tests
    simple_imputer_tests
    binarizer_tests
    max_abs_scaler_tests

    # Feature selection tests
    variance_threshold_tests
    select_k_best_tests
    f_regression_tests
    select_percentile_tests
    select_fpr_tests
    select_fdr_tests
    mutual_info_tests
    chi2_tests
    generic_univariate_selector_tests

    # API tests
    umbrella_header_tests

    # Data and metrics tests
    dataset_tests
    model_evaluation_tests
    model_selection_tests
    pipeline_tests
    train_test_integration_tests
)

foreach(TARGET_NAME IN LISTS ML_TARGETS)

    target_compile_options(
        ${TARGET_NAME}
        PRIVATE
            ${ML_WARNING_FLAGS}
    )

endforeach()

# -------------------------------------------------
# Install rules
# -------------------------------------------------

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(
    TARGETS ml_library
    EXPORT CppMLLibraryTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(
    DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(
    EXPORT CppMLLibraryTargets
    FILE CppMLLibraryTargets.cmake
    NAMESPACE CppMLLibrary::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CppMLLibrary
)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/CppMLLibraryConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/CppMLLibraryConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CppMLLibrary
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/CppMLLibraryConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/CppMLLibraryConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/CppMLLibraryConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CppMLLibrary
)

if(ML_BUILD_BENCHMARKS)

    add_executable(
        matrix_benchmark
        benchmarks/matrix_benchmark.cpp
    )

    add_executable(
        algorithm_benchmark
        benchmarks/algorithm_benchmark.cpp
    )

    target_link_libraries(
        matrix_benchmark
        PRIVATE
            ml_library
    )

    target_link_libraries(
        algorithm_benchmark
        PRIVATE
            ml_library
    )

    target_compile_options(
        matrix_benchmark
        PRIVATE
            ${ML_WARNING_FLAGS}
    )

    target_compile_options(
        algorithm_benchmark
        PRIVATE
            ${ML_WARNING_FLAGS}
    )

endif()