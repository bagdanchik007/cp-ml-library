cmake_minimum_required(VERSION 3.16)

project(
    CppMLLibrary
    VERSION 1.0.0
    LANGUAGES CXX
)

# -------------------------------------------------
# C++ standard
# -------------------------------------------------

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# -------------------------------------------------
# Library
# -------------------------------------------------

add_library(
    ml_library
    src/algorithms/linear_regression.cpp
    src/algorithms/kmeans.cpp
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

# -------------------------------------------------
# Matrix arithmetic tests
# -------------------------------------------------

add_ml_test(
    matrix_arithmetic_tests
    tests/core/matrix/matrix_arithmetic_tests.cpp
)

# -------------------------------------------------
# Matrix utility tests
# -------------------------------------------------

add_ml_test(
    matrix_utility_tests
    tests/core/matrix/matrix_utility_tests.cpp
)

# -------------------------------------------------
# Matrix validation tests
# -------------------------------------------------

add_ml_test(
    matrix_validation_tests
    tests/core/matrix/matrix_validation_tests.cpp
)

# -------------------------------------------------
# Matrix transformation tests
# -------------------------------------------------

add_ml_test(
    matrix_transformation_tests
    tests/core/matrix/matrix_transformation_tests.cpp
)

# -------------------------------------------------
# Matrix statistics tests
# -------------------------------------------------

add_ml_test(
    matrix_statistics_tests
    tests/core/matrix/matrix_statistics_tests.cpp
)

# -------------------------------------------------
# Matrix norm tests
# -------------------------------------------------

add_ml_test(
    matrix_norm_tests
    tests/core/matrix/matrix_norm_tests.cpp
)

# -------------------------------------------------
# Matrix column tests
# -------------------------------------------------

add_ml_test(
    matrix_column_tests
    tests/core/matrix/matrix_column_tests.cpp
)

# -------------------------------------------------
# Matrix row tests
# -------------------------------------------------

add_ml_test(
    matrix_row_tests
    tests/core/matrix/matrix_row_tests.cpp
)

# -------------------------------------------------
# Matrix comparison tests
# -------------------------------------------------

add_ml_test(
    matrix_comparison_tests
    tests/core/matrix/matrix_comparison_tests.cpp
)

# -------------------------------------------------
# Matrix scalar tests
# -------------------------------------------------

add_ml_test(
    matrix_scalar_tests
    tests/core/matrix/matrix_scalar_tests.cpp
)

# -------------------------------------------------
# Matrix linear algebra tests
# -------------------------------------------------

add_ml_test(
    matrix_linear_algebra_tests
    tests/core/matrix/matrix_linear_algebra_tests.cpp
)

# -------------------------------------------------
# Matrix decomposition tests
# -------------------------------------------------

add_ml_test(
    matrix_decomposition_tests
    tests/core/matrix/matrix_decomposition_tests.cpp
)

# -------------------------------------------------
# Matrix solve tests
# -------------------------------------------------

add_ml_test(
    matrix_solve_tests
    tests/core/matrix/matrix_solve_tests.cpp
)

# -------------------------------------------------
# Linear Regression tests
# -------------------------------------------------

add_ml_test(
    linear_regression_tests
    tests/algorithms/linear_regression_tests.cpp
)

# -------------------------------------------------
# K-Means tests
# -------------------------------------------------

add_ml_test(
    kmeans_tests
    tests/algorithms/kmeans_tests.cpp
)

# -------------------------------------------------
# StandardScaler tests
# -------------------------------------------------

add_ml_test(
    standard_scaler_tests
    tests/preprocessing/standard_scaler_tests.cpp
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
    ml_library
    ml_demo

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

    linear_regression_tests
    kmeans_tests
    standard_scaler_tests
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

install(
    TARGETS ml_library
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(
    DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)