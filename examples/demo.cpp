#include <iostream>

namespace ml::examples {

void run_linear_regression_demo();
void run_kmeans_demo();
void run_preprocessing_demo();
void run_feature_selection_demo();

}

int main()
{
    std::cout << "=== Cpp ML Library Demo ===\n\n";

    std::cout << "--- Algorithms ---\n\n";

    ml::examples::run_linear_regression_demo();
    ml::examples::run_kmeans_demo();

    std::cout << "\n--- Preprocessing ---\n\n";

    ml::examples::run_preprocessing_demo();

    std::cout << "\n--- Feature Selection ---\n\n";

    ml::examples::run_feature_selection_demo();

    std::cout << "\nDemo finished successfully.\n";

    return 0;
}