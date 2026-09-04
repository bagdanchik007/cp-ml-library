#pragma once

#include "ml/core/matrix/matrix.hpp"

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ml {

class ColumnTransformerInterface {

public:

    virtual ~ColumnTransformerInterface() = default;

    virtual void fit(
        const Matrix& data
    ) = 0;

    virtual Matrix transform(
        const Matrix& data
    ) const = 0;

};


template <typename Transformer>

class ColumnTransformerAdapter
    : public ColumnTransformerInterface {

public:

    explicit ColumnTransformerAdapter(
        Transformer transformer
    )
        : transformer_(
            std::move(transformer)
        ) {}

    void fit(
        const Matrix& data
    ) override
    {
        transformer_.fit(data);
    }

    Matrix transform(
        const Matrix& data
    ) const override
    {
        return transformer_.transform(data);
    }

private:

    Transformer transformer_;

};


class ColumnTransformer {

public:

    ColumnTransformer() = default;


    template <typename Transformer>

    void add_transformer(
        Transformer transformer,
        std::vector<size_t> columns
    )
    {
        if (columns.empty()) {

            throw std::invalid_argument(
                "ColumnTransformer::add_transformer: "
                "columns must not be empty"
            );
        }

        TransformerEntry entry;

        entry.transformer =
            std::make_unique<
                ColumnTransformerAdapter<Transformer>
            >(
                std::move(transformer)
            );

        entry.columns =
            std::move(columns);

        transformers_.push_back(
            std::move(entry)
        );
    }


    void fit(
        const Matrix& data
    )
    {
        validate_columns(data);

        for (auto& entry : transformers_) {

            const Matrix selected =
                select_columns(
                    data,
                    entry.columns
                );

            entry.transformer->fit(
                selected
            );
        }

        fitted_ = true;
    }


    Matrix transform(
        const Matrix& data
    ) const
    {
        if (!fitted_) {

            throw std::logic_error(
                "ColumnTransformer::transform: "
                "transformer has not been fitted"
            );
        }

        validate_columns(data);

        Matrix result = data;

        for (const auto& entry : transformers_) {

            const Matrix selected =
                select_columns(
                    data,
                    entry.columns
                );

            const Matrix transformed =
                entry.transformer->transform(
                    selected
                );

            if (
                transformed.rows != data.rows ||
                transformed.cols != entry.columns.size()
            ) {

                throw std::runtime_error(
                    "ColumnTransformer::transform: "
                    "transformed data has invalid shape"
                );
            }

            set_columns(
                result,
                transformed,
                entry.columns
            );
        }

        return result;
    }


    Matrix fit_transform(
        const Matrix& data
    )
    {
        fit(data);

        return transform(data);
    }


    bool is_fitted() const noexcept
    {
        return fitted_;
    }


private:

    struct TransformerEntry {

        std::unique_ptr<
            ColumnTransformerInterface
        > transformer;

        std::vector<size_t> columns;

    };


    static Matrix select_columns(
        const Matrix& data,
        const std::vector<size_t>& columns
    )
    {
        Matrix result(
            data.rows,
            columns.size()
        );

        for (
            size_t row = 0;
            row < data.rows;
            ++row
        ) {

            for (
                size_t column = 0;
                column < columns.size();
                ++column
            ) {

                result(row, column) =
                    data(
                        row,
                        columns[column]
                    );
            }
        }

        return result;
    }


    static void set_columns(
        Matrix& destination,
        const Matrix& source,
        const std::vector<size_t>& columns
    )
    {
        for (
            size_t row = 0;
            row < destination.rows;
            ++row
        ) {

            for (
                size_t column = 0;
                column < columns.size();
                ++column
            ) {

                destination(
                    row,
                    columns[column]
                ) =
                    source(
                        row,
                        column
                    );
            }
        }
    }


    void validate_columns(
        const Matrix& data
    ) const
    {
        if (
            data.rows == 0 ||
            data.cols == 0
        ) {

            throw std::invalid_argument(
                "ColumnTransformer: "
                "input data must not be empty"
            );
        }

        for (
            const auto& entry :
            transformers_
        ) {

            for (
                size_t column :
                entry.columns
            ) {

                if (
                    column >= data.cols
                ) {

                    throw std::out_of_range(
                        "ColumnTransformer: "
                        "column index out of range"
                    );
                }
            }
        }
    }


    std::vector<TransformerEntry>
        transformers_;

    bool fitted_ = false;

};

} // namespace ml