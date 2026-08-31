# Cpp ML Library

Eine kleine, dependency-freie Machine-Learning-Bibliothek für C++17.

Das Projekt wurde für Lernen, Experimente und Portfolio-Projekte entwickelt. Alle Algorithmen sind in verständlichem C++ implementiert, und die öffentliche API wird durch automatisierte CTest-Tests überprüft.

## Funktionen

### Matrix und mathematische Grundlagen

* Dichte `Matrix`-Klasse
* Matrix-Arithmetik
* Lineare Algebra
* Matrix-Zerlegungen
* Statistische Funktionen
* Normen und weitere Matrix-Operationen

### Machine Learning

#### Regression und Clustering

* Lineare Regression
* Ridge Regression
* K-Means
* DBSCAN

#### Klassifikation

* Decision Tree
* Random Forest
* K-Nearest Neighbors
* Gaussian Naive Bayes
* Logistic Regression

### Datenvorverarbeitung

* Missing-Value-Imputation
* Standardisierung
* Normalisierung
* Min-Max Scaling
* Robust Scaling
* Max-Abs Scaling
* One-Hot-Encoding
* Label-Encoding
* Binning und Diskretisierung
* Logarithmische Transformationen
* Polynomial Features
* Train/Test-Splitting

### Feature Selection

* Variance Threshold
* Univariate Feature Selection
* `SelectKBest`
* `SelectPercentile`
* `SelectFPR`
* `SelectFDR`
* Mutual Information
* Chi-Square Tests

### Model Selection

* K-Fold Cross-Validation
* Stratified K-Fold
* Cross-Validation
* Grid Search

### Evaluation

* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* Mean Absolute Error (MAE)
* R² Score
* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* Silhouette Score

### Weitere Funktionen

* Numerischer CSV-Reader
* `Dataset`-Klasse
* Composable `Pipeline` für Transformer und Modelle

---

## Build und Tests

Das Projekt verwendet **CMake**.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Die Tests werden automatisch mit **CTest** ausgeführt.

Die CI-Pipeline baut und testet das Projekt mit:

* GCC
* Clang
* MSVC

---

## Quick Start

```cpp
#include <ml/ml.hpp>

int main() {
    ml::Matrix features{
        {0.0},
        {1.0},
        {2.0},
        {3.0}
    };

    std::vector<double> targets{
        1.0,
        3.0,
        5.0,
        7.0
    };

    auto model = ml::make_pipeline(
        ml::LinearRegression{},
        ml::StandardScaler{}
    );

    model.fit(features, targets);

    const auto prediction =
        model.predict(
            ml::Matrix{{4.0}}
        );
}
```

Weitere Beispiele findest du in der [Beispieldokumentation](docs/examples.md).

Die öffentliche API und die Architektur des Projekts werden im Verzeichnis [`docs/`](docs/) beschrieben.

---

## Integration als CMake-Package

Die Bibliothek kann als CMake-Package installiert werden:

```bash
cmake -S . -B build
cmake --install build --prefix /desired/prefix
```

Anschließend kann das Projekt in einer anderen CMake-Anwendung verwendet werden:

```cmake
find_package(CppMLLibrary CONFIG REQUIRED)

target_link_libraries(
    your_target
    PRIVATE
    CppMLLibrary::ml_library
)
```

---

## Benchmarks

Benchmarks sind optional, damit normale Builds schnell bleiben.

Aktivierung:

```bash
cmake -S . -B build -DML_BUILD_BENCHMARKS=ON
```

Anschließend können die Benchmarks gebaut werden:

```bash
cmake --build build --target matrix_benchmark algorithm_benchmark
```

---

## Versionierung

Das Projekt orientiert sich an [Semantic Versioning](https://semver.org/).

Das aktuelle Ziel für die öffentliche API ist **v1.1.0**.

Weitere Informationen zu Änderungen und Releases findest du in der [CHANGELOG.md](CHANGELOG.md).

---

## Lizenz

Aktuell wurde noch keine Lizenz ausgewählt.

Vor einer öffentlichen Veröffentlichung oder dem Akzeptieren externer Contributions sollte eine passende `LICENSE`-Datei hinzugefügt werden.
