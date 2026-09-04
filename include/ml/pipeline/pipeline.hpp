#pragma once

#include "ml/core/matrix/matrix.hpp"

#include <stdexcept>
#include <tuple>
#include <utility>

namespace ml {

/**
 * Type-safe supervised pipeline for transformers exposing
 * fit_transform/transform and an estimator exposing fit/predict.
 */
template <typename Estimator, typename... Transformers>
class Pipeline {

public:

    Pipeline(
        Estimator estimator,
        Transformers... transformers
    )
        : estimator_(std::move(estimator)),
          transformers_(std::move(transformers)...)
    {
    }

    void fit(
        const Matrix& features,
        const std::vector<double>& targets
    )
    {
        estimator_.fit(
            fit_transform(features),
            targets
        );

        fitted_ = true;
    }

    Matrix predict(
        const Matrix& features
    ) const
    {
        if (!fitted_) {
            throw std::logic_error(
                "Pipeline::predict: "
                "pipeline has not been fitted"
            );
        }

        return estimator_.predict(
            transform(features)
        );
    }

    Estimator& estimator() noexcept
    {
        return estimator_;
    }

    const Estimator& estimator() const noexcept
    {
        return estimator_;
    }

    bool is_fitted() const noexcept
    {
        return fitted_;
    }

private:

    template <size_t Index = 0>
    Matrix fit_transform(
        Matrix data
    )
    {
        if constexpr (
            Index < sizeof...(Transformers)
        ) {
            return fit_transform<Index + 1>(
                std::get<Index>(
                    transformers_
                ).fit_transform(data)
            );
        } else {
            return data;
        }
    }

    template <size_t Index = 0>
    Matrix transform(
        Matrix data
    ) const
    {
        if constexpr (
            Index < sizeof...(Transformers)
        ) {
            return transform<Index + 1>(
                std::get<Index>(
                    transformers_
                ).transform(data)
            );
        } else {
            return data;
        }
    }

private:

    Estimator estimator_;

    std::tuple<Transformers...> transformers_;

    bool fitted_ = false;

};


/**
 * Create a pipeline from an estimator and transformers.
 */
template <
    typename Estimator,
    typename... Transformers
>
Pipeline<Estimator, Transformers...> make_pipeline(
    Estimator estimator,
    Transformers... transformers
)
{
    return Pipeline<
        Estimator,
        Transformers...
    >(
        std::move(estimator),
        std::move(transformers)...
    );
}

} // namespace ml