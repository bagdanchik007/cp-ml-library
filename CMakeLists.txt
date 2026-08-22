cmake_minimum_required(VERSION 3.16)

project(
    CppMLLibrary
    VERSION 1.0.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# -------------------------------------------------
# Library
# -------------------------------------------------

add_library(ml_library
    src/algorithms/linear_regression.cpp
    src/algorithms/kmeans.cpp

    # Matrix core
    include/ml/core/matrix/matrix.hpp
    include/ml/core/matrix/matrix_utility.hpp

    # Matrix operations
    include/ml/core/matrix/operations/matrix_construction.hpp
    include/ml/core/matrix/operations/matrix_access.hpp
    include/ml/core/matrix/operations/matrix_statistics.hpp
    include/ml/core/matrix/operations/matrix_linear_algebra.hpp
    include/ml/core/matrix/operations/matrix_arithmetic.hpp
    include/ml/core/matrix/operations/matrix_transformations.hpp
    include/ml/core/matrix/operations/matrix_scalar.hpp
    include/ml/core/matrix/operations/matrix_factory.hpp
    include/ml/core/matrix/operations/matrix_io.hpp
    include/ml/core/matrix/operations/matrix_utility.hpp
    include/ml/core/matrix/operations/matrix_comparison.hpp
    include/ml/core/matrix/operations/matrix_slicing.hpp
    include/ml/core/matrix/operations/matrix_decomposition.hpp
    # Matrix internals
    include/ml/core/matrix/detail/matrix_validation.hpp
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
# Tests
# -------------------------------------------------

enable_testing()

# -------------------------------------------------
# Matrix tests
# -------------------------------------------------

add_executable(
    matrix_tests
    tests/core/matrix/matrix_tests.cpp
)

target_link_libraries(
    matrix_tests
    PRIVATE
        ml_library
)

add_test(
    NAME matrix_tests
    COMMAND matrix_tests
)

# -------------------------------------------------
# Matrix arithmetic tests
# -------------------------------------------------

add_executable(
    matrix_arithmetic_tests
    tests/core/matrix/matrix_arithmetic_tests.cpp
)

target_link_libraries(
    matrix_arithmetic_tests
    PRIVATE
        ml_library
)

add_test(
    NAME matrix_arithmetic_tests
    COMMAND matrix_arithmetic_tests
)

# -------------------------------------------------
# Matrix utility tests
# -------------------------------------------------

add_executable(
    matrix_utility_tests
    tests/core/matrix/matrix_utility_tests.cpp
)

target_link_libraries(
    matrix_utility_tests
    PRIVATE
        ml_library
)

add_test(
    NAME matrix_utility_tests
    COMMAND matrix_utility_tests
)

# -------------------------------------------------
# Matrix validation tests
# -------------------------------------------------

add_executable(
    matrix_validation_tests
    tests/core/matrix/matrix_validation_tests.cpp
)

target_link_libraries(
    matrix_validation_tests
    PRIVATE
        ml_library
)

add_test(
    NAME matrix_validation_tests
    COMMAND matrix_validation_tests
)

# -------------------------------------------------
# Matrix transformation tests
# -------------------------------------------------

add_executable(
    matrix_transformation_tests
    tests/core/matrix/matrix_transformation_tests.cpp
)

target_link_libraries(
    matrix_transformation_tests
    PRIVATE
        ml_library
)

add_test(
    NAME matrix_transformation_tests
    COMMAND matrix_transformation_tests
)

# -------------------------------------------------
# Matrix statistics tests
# -------------------------------------------------

add_executable(
    matrix_statistics_tests
    tests/core/matrix/matrix_statistics_tests.cpp
)

target_link_libraries(
    matrix_statistics_tests
    PRIVATE
        ml_library
)

add_test(
    NAME matrix_statistics_tests
    COMMAND matrix_statistics_tests
)

# -------------------------------------------------
# Matrix norm tests
# -------------------------------------------------

add_executable(
    matrix_norm_tests
    tests/core/matrix/matrix_norm_tests.cpp
)

target_link_libraries(
    matrix_norm_tests
    PRIVATE
        ml_library
)

add_test(
    NAME matrix_norm_tests
    COMMAND matrix_norm_tests
)

# -------------------------------------------------
# Matrix column tests
# -------------------------------------------------

add_executable(
    matrix_column_tests
    tests/core/matrix/matrix_column_tests.cpp
)

target_link_libraries(
    matrix_column_tests
    PRIVATE
        ml_library
)

add_test(
    NAME matrix_column_tests
    COMMAND matrix_column_tests
)

# -------------------------------------------------
# Matrix row tests
# -------------------------------------------------

add_executable(
    matrix_row_tests
    tests/core/matrix/matrix_row_tests.cpp
)

target_link_libraries(
    matrix_row_tests
    PRIVATE
        ml_library
)

add_test(
    NAME matrix_row_tests
    COMMAND matrix_row_tests
)

# -------------------------------------------------
# Matrix comparison tests
# -------------------------------------------------

add_executable(
    matrix_comparison_tests
    tests/core/matrix/matrix_comparison_tests.cpp
)

target_link_libraries(
    matrix_comparison_tests
    PRIVATE
        ml_library
)

add_test(
    NAME matrix_comparison_tests
    COMMAND matrix_comparison_tests
)

# -------------------------------------------------
# Matrix scalar tests
# -------------------------------------------------

add_executable(
    matrix_scalar_tests
    tests/core/matrix/matrix_scalar_tests.cpp
)

target_link_libraries(
    matrix_scalar_tests
    PRIVATE
        ml_library
)

add_test(
    NAME matrix_scalar_tests
    COMMAND matrix_scalar_tests
)

# -------------------------------------------------
# Matrix linear algebra tests
# -------------------------------------------------

add_executable(
    matrix_linear_algebra_tests
    tests/core/matrix/matrix_linear_algebra_tests.cpp
)

target_link_libraries(
    matrix_linear_algebra_tests
    PRIVATE
        ml_library
)

add_test(
    NAME matrix_linear_algebra_tests
    COMMAND matrix_linear_algebra_tests
)

# -------------------------------------------------
# Linear Regression tests
# -------------------------------------------------

add_executable(
    linear_regression_tests
    tests/algorithms/linear_regression_tests.cpp
)

target_link_libraries(
    linear_regression_tests
    PRIVATE
        ml_library
)

add_test(
    NAME linear_regression_tests
    COMMAND linear_regression_tests
)

# -------------------------------------------------
# K-Means tests
# -------------------------------------------------

add_executable(
    kmeans_tests
    tests/algorithms/kmeans_tests.cpp
)

target_link_libraries(
    kmeans_tests
    PRIVATE
        ml_library
)

add_test(
    NAME kmeans_tests
    COMMAND kmeans_tests
)

# -------------------------------------------------
# Compiler warnings
# -------------------------------------------------

set(ML_WARNING_FLAGS
    -Wall
    -Wextra
    -Wpedantic
)

if(MSVC)

    target_compile_options(
        ml_library
        PRIVATE
            /W4
    )

    target_compile_options(
        ml_demo
        PRIVATE
            /W4
    )

    target_compile_options(
        matrix_tests
        PRIVATE
            /W4
    )

    target_compile_options(
        matrix_arithmetic_tests
        PRIVATE
            /W4
    )

    target_compile_options(
        matrix_utility_tests
        PRIVATE
            /W4
    )

    target_compile_options(
        matrix_validation_tests
        PRIVATE
            /W4
    )

    target_compile_options(
        matrix_transformation_tests
        PRIVATE
            /W4
    )

    target_compile_options(
        matrix_statistics_tests
        PRIVATE
            /W4
    )

    target_compile_options(
        linear_regression_tests
        PRIVATE
            /W4
    )

    target_compile_options(
        kmeans_tests
        PRIVATE
            /W4
    )

else()

    target_compile_options(
        ml_library
        PRIVATE
            ${ML_WARNING_FLAGS}
    )

    target_compile_options(
        ml_demo
        PRIVATE
            ${ML_WARNING_FLAGS}
    )

    target_compile_options(
        matrix_tests
        PRIVATE
            ${ML_WARNING_FLAGS}
    )

    target_compile_options(
        matrix_arithmetic_tests
        PRIVATE
            ${ML_WARNING_FLAGS}
    )

    target_compile_options(
        matrix_utility_tests
        PRIVATE
            ${ML_WARNING_FLAGS}
    )

    target_compile_options(
        matrix_validation_tests
        PRIVATE
            ${ML_WARNING_FLAGS}
    )

    target_compile_options(
        matrix_transformation_tests
        PRIVATE
            ${ML_WARNING_FLAGS}
    )

    target_compile_options(
        matrix_statistics_tests
        PRIVATE
            ${ML_WARNING_FLAGS}
    )

    target_compile_options(
        linear_regression_tests
        PRIVATE
            ${ML_WARNING_FLAGS}
    )

    target_compile_options(
        kmeans_tests
        PRIVATE
            ${ML_WARNING_FLAGS}
    )

endif()

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

