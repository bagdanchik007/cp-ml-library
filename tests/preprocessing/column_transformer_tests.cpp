#include "ml/preprocessing/column_transformer.hpp"
#include "ml/preprocessing/standard_scaler.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace
{

    bool approximately_equal(
        double left,
        double right,
        double epsilon = 1e-9)
    {
        return std::abs(left - right) < epsilon;
    }

    void test_selected_columns_are_transformed()
    {
        const ml::Matrix data{
            {1.0, 10.0, 100.0},
            {2.0, 20.0, 200.0},
            {3.0, 30.0, 300.0}};

        ml::ColumnTransformer transformer;

        transformer.add_transformer(
            ml::StandardScaler{},
            {0, 2});

        const ml::Matrix result =
            transformer.fit_transform(data);

        assert(result.rows == 3);
        assert(result.cols == 3);

        // Nicht ausgewählte Spalte bleibt unverändert.
        assert(result(0, 1) == 10.0);
        assert(result(1, 1) == 20.0);
        assert(result(2, 1) == 30.0);

        // Ausgewählte Spalten haben Mittelwert 0.
        assert(
            approximately_equal(
                result(0, 0) +
                    result(1, 0) +
                    result(2, 0),
                0.0));

        assert(
            approximately_equal(
                result(0, 2) +
                    result(1, 2) +
                    result(2, 2),
                0.0));
    }

    void test_transform_after_fit()
    {
        const ml::Matrix training_data{
            {1.0, 10.0},
            {2.0, 20.0},
            {3.0, 30.0}};

        const ml::Matrix test_data{
            {4.0, 40.0}};

        ml::ColumnTransformer transformer;

        transformer.add_transformer(
            ml::StandardScaler{},
            {0});

        transformer.fit(training_data);

        assert(transformer.is_fitted());

        const ml::Matrix result =
            transformer.transform(test_data);

        assert(result.rows == 1);
        assert(result.cols == 2);

        // Spalte 1 wurde nicht transformiert.
        assert(result(0, 1) == 40.0);

        // Spalte 0 wurde mit den Trainingsdaten skaliert.
        assert(result(0, 0) > 0.0);
    }

    void test_multiple_transformers()
    {
        const ml::Matrix data{
            {1.0, 10.0},
            {2.0, 20.0},
            {3.0, 30.0}};

        ml::ColumnTransformer transformer;

        transformer.add_transformer(
            ml::StandardScaler{},
            {0});

        transformer.add_transformer(
            ml::StandardScaler{},
            {1});

        const ml::Matrix result =
            transformer.fit_transform(data);

        assert(result.rows == 3);
        assert(result.cols == 2);

        assert(
            approximately_equal(
                result(0, 0) +
                    result(1, 0) +
                    result(2, 0),
                0.0));

        assert(
            approximately_equal(
                result(0, 1) +
                    result(1, 1) +
                    result(2, 1),
                0.0));
    }

    void test_transform_before_fit_throws()
    {
        const ml::Matrix data{
            {1.0},
            {2.0}};

        ml::ColumnTransformer transformer;

        transformer.add_transformer(
            ml::StandardScaler{},
            {0});

        bool threw = false;

        try
        {

            transformer.transform(data);
        }
        catch (const std::logic_error &)
        {

            threw = true;
        }

        assert(threw);
    }

    void test_invalid_column_throws()
    {
        const ml::Matrix data{
            {1.0, 2.0},
            {3.0, 4.0}};

        ml::ColumnTransformer transformer;

        transformer.add_transformer(
            ml::StandardScaler{},
            {2});

        bool threw = false;

        try
        {

            transformer.fit(data);
        }
        catch (const std::out_of_range &)
        {

            threw = true;
        }

        assert(threw);
    }

    void test_empty_columns_throw()
    {
        ml::ColumnTransformer transformer;

        bool threw = false;

        try
        {

            transformer.add_transformer(
                ml::StandardScaler{},
                {});
        }
        catch (const std::invalid_argument &)
        {

            threw = true;
        }

        assert(threw);
    }

} // namespace

int main()
{
    std::cout
        << "Running ColumnTransformer tests...\n";

    test_selected_columns_are_transformed();

    test_transform_after_fit();

    test_multiple_transformers();

    test_transform_before_fit_throws();

    test_invalid_column_throws();

    test_empty_columns_throw();

    std::cout
        << "All ColumnTransformer tests passed!\n";

    return 0;
}