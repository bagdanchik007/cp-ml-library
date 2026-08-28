# Architecture

The library deliberately separates reusable building blocks:

```text
Matrix core
  ├─ preprocessing and feature selection
  ├─ data and model-selection helpers
  └─ algorithms
       └─ metrics, pipelines and examples consume these APIs
```

`Matrix` remains the common numeric container. Preprocessing modules are header-only transformers with `fit`, `transform` and `fit_transform`; compiled algorithm sources are linked into `ml_library`. Tests are small executables registered through CTest, so no third-party test framework is needed.

The CMake target exports as `CppMLLibrary::ml_library`. Benchmark targets are intentionally separate and opt-in.
