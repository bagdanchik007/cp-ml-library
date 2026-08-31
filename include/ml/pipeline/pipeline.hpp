#pragma once

#include "ml/core/matrix/matrix.hpp"

#include <stdexcept>
#include <tuple>
#include <utility>

namespace ml {

template <typename Estimator, typename... Transformers>
class Pipeline {
public:
    Pipeline(
        Estimator estimator,
        Transformers... transformers
    )
        : estimator_(std::move(estimator)),
          transformers_(std::move(transformers)...) {}

    template <typename Targets>
    void fit(
        const Matrix& features,
        const Targets& targets
    ) {
        Matrix transformed = features;

        apply_fit_transform(transformed);

        estimator_.fit(
            transformed,
            targets
        );

        fitted_ = true;
    }

    Matrix predict(
        const Matrix& features
    ) const {
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

    bool is_fitted() const noexcept {
        return fitted_;
    }

private:
    Matrix transform(
        Matrix features
    ) const {
        apply_transform(features);

        return features;
    }

    template <std::size_t Index = 0>
    void apply_fit_transform(
        Matrix& features
    ) {
        if constexpr (
            Index < sizeof...(Transformers)
        ) {
            auto& transformer =
                std::get<Index>(transformers_);

            transformer.fit(features);

            features =
                transformer.transform(features);

            apply_fit_transform<
                Index + 1
            >(features);
        }
    }

    template <std::size_t Index = 0>
    void apply_transform(
        Matrix& features
    ) const {
        if constexpr (
            Index < sizeof...(Transformers)
        ) {
            const auto& transformer =
                std::get<Index>(transformers_);

            features =
                transformer.transform(features);

            apply_transform<
                Index + 1
            >(features);
        }
    }

private:
    Estimator estimator_;

    std::tuple<Transformers...>
        transformers_;

    bool fitted_ = false;
};

template <typename Estimator, typename... Transformers>
auto make_pipeline(
    Estimator estimator,
    Transformers... transformers
) {
    return Pipeline<
        Estimator,
        Transformers...
    >(
        std::move(estimator),
        std::move(transformers)...
    );
}

} // namespace ml