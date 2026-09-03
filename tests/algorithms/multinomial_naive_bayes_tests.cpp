#include "ml/algorithms/multinomial_naive_bayes.hpp"

#include <cassert>

int main() {
    // Toy "word count" features: [count_word_sport, count_word_finance]
    const ml::Matrix X{
        {5.0, 0.0}, {4.0, 1.0}, {6.0, 0.0}, // class 0: "sport" documents
        {0.0, 5.0}, {1.0, 4.0}, {0.0, 6.0}  // class 1: "finance" documents
    };
    const std::vector<int> y{0, 0, 0, 1, 1, 1};

    ml::MultinomialNaiveBayes model(1.0);
    assert(!model.is_fitted());
    model.fit(X, y);
    assert(model.is_fitted());

    const ml::Matrix query{{5.0, 0.0}, {0.0, 5.0}};
    const auto predictions = model.predict(query);
    assert(predictions.size() == 2);
    assert(predictions[0] == 0);
    assert(predictions[1] == 1);

    bool threw = false;
    try {
        ml::MultinomialNaiveBayes invalid(-1.0);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}
