# Cpp ML Library

Eine leichtgewichtige C++-Bibliothek für grundlegende Machine-Learning-Algorithmen mit Fokus auf einer einfachen, modularen und gut wartbaren Struktur.

## Features

* Matrix-Operationen
* Lineare Regression
* K-Means Clustering
* C++17
* CMake
* Unit-Tests mit CTest
* Modulare Projektstruktur

## Projektstruktur

```text
.
├── include/ml/
│   ├── core/matrix/
│   ├── algorithms/
│   └── ml.hpp
├── src/
│   └── algorithms/
├── tests/
│   ├── core/matrix/
│   └── algorithms/
├── examples/
├── CMakeLists.txt
├── LICENSE
└── README.md
```

## Build

```bash
cmake -S . -B build
cmake --build build -j$(nproc)
```

## Tests

```bash
ctest --test-dir build --output-on-failure
```

## Beispiel

```cpp
#include <ml/ml.hpp>

int main()
{
    ml::Matrix<double> matrix{
        {1.0, 2.0},
        {3.0, 4.0}
    };

    return 0;
}
```

## Algorithmen

### Lineare Regression

Implementierung einer einfachen linearen Regression für numerische Daten.

### K-Means

Clustering-Algorithmus zur Gruppierung numerischer Daten in mehrere Cluster.

## Autor

**bagdanchik007** — C++ / Machine Learning

[GitHub](https://github.com/bagdanchik007)

## Lizenz

Dieses Projekt steht unter der in der Datei [LICENSE](LICENSE) angegebenen Lizenz.
