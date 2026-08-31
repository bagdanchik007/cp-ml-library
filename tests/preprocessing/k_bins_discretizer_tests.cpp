#include "ml/preprocessing/k_bins_discretizer.hpp"

#include <cassert>

int main() {
    const ml::Matrix data{{0.0}, {2.5}, {5.0}, {7.5}, {10.0}};

    ml::KBinsDiscretizer discretizer(5);
    const auto bins = discretizer.fit_transform(data);

    assert(bins(0, 0) == 0.0);
    assert(bins(4, 0) == 4.0);
    assert(bins(2, 0) == 2.0);

    // Out-of-range values are clamped to the valid bin range.
    const auto out_of_range = discretizer.transform(ml::Matrix{{-5.0}, {50.0}});
    assert(out_of_range(0, 0) == 0.0);
    assert(out_of_range(1, 0) == 4.0);
}
