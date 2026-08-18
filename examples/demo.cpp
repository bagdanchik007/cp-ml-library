#include <iostream>

namespace ml::examples {

void run_linear_regression_demo();
void run_kmeans_demo();

}

int main() {
    std::cout << "=== Cpp ML Library Demo ===\n\n";

    ml::examples::run_linear_regression_demo();
    ml::examples::run_kmeans_demo();

    std::cout << "Demo finished successfully.\n";

    return 0;
}