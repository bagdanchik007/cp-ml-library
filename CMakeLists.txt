cmake_minimum_required(VERSION 3.16)
project(CppMLLibrary VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# -------------------------------------------------
# Library
# -------------------------------------------------
add_library(ml_library
    src/linear_regression.cpp
    src/kmeans.cpp
)

target_include_directories(ml_library
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# -------------------------------------------------
# Demo executable
# -------------------------------------------------
add_executable(ml_demo examples/demo.cpp)
target_link_libraries(ml_demo PRIVATE ml_library)

# -------------------------------------------------
# Matrix tests
# -------------------------------------------------
enable_testing()

add_executable(matrix_tests tests/matrix_tests.cpp)
target_link_libraries(matrix_tests PRIVATE ml_library)
add_test(NAME matrix_tests COMMAND matrix_tests)

# -------------------------------------------------
# Compiler warnings
# -------------------------------------------------
if(MSVC)
    target_compile_options(ml_library PRIVATE /W4)
    target_compile_options(ml_demo PRIVATE /W4)
    target_compile_options(matrix_tests PRIVATE /W4)
else()
    target_compile_options(ml_library PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(ml_demo PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(matrix_tests PRIVATE -Wall -Wextra -Wpedantic)
endif()

# -------------------------------------------------
# Install rules
# -------------------------------------------------
include(GNUInstallDirs)

install(TARGETS ml_library
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})