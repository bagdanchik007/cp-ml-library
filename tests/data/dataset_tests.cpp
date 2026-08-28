#include "ml/data/dataset.hpp"
#include "ml/data/csv_reader.hpp"

#include <cassert>
#include <cstdio>
#include <fstream>
#include <stdexcept>

int main() {
    ml::Dataset dataset(ml::Matrix{{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}, {7.0, 8.0}}, {0.0, 1.0, 0.0, 1.0});
    assert(dataset.rows() == 4);
    assert(dataset.columns() == 2);
    const auto split = ml::train_test_split(dataset, 0.25, false);
    assert(split.train.rows() == 3);
    assert(split.test.rows() == 1);
    assert(split.test.features()(0, 0) == 7.0);

    bool mismatched = false;
    try { ml::Dataset invalid(ml::Matrix(2, 1), {1.0}); } catch (const std::invalid_argument&) { mismatched = true; }
    assert(mismatched);

    const char* path = "cp_ml_library_dataset_test.csv";
    { std::ofstream output(path); output << "a,b,target\n1,2,0\n3,4,1\n"; }
    const auto csv = ml::read_csv(path);
    std::remove(path);
    assert(csv.rows() == 2 && csv.columns() == 2);
    assert(csv.features()(1, 1) == 4.0 && csv.targets()[1] == 1.0);
}
