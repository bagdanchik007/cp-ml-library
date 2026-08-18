# cp-ml-library 

Eine leichtgewichtige Machine-Learning-Bibliothek, die mit modernem C++ entwickelt wurde.

Dieses Projekt implementiert grundlegende Algorithmen des maschinellen Lernens sowie mathematische Matrixoperationen von Grund auf. Der Fokus liegt auf einer klaren Architektur, verständlichem Code, guter Wartbarkeit und dem Verzicht auf externe Machine-Learning-Frameworks.

Die Bibliothek enthält derzeit eine eigene Matrix-Implementierung, Linear Regression und K-Means Clustering.

## Features

### Core

- Leichtgewichtige `Matrix`-Klasse
- Matrixaddition und -subtraktion
- Matrix-Matrix-Multiplikation
- Matrix-Vektor-Multiplikation
- Skalarmultiplikation und -division
- In-Place-Skalaroperationen
- Transposition von Matrizen
- Zugriff und Änderung einzelner Zeilen
- Erzeugung von Null- und Einsmatrizen
- Zufällige Matrixgenerierung
- Dimensionsvalidierung
- Fehlerbehandlung bei ungültigen Operationen

### Machine-Learning-Algorithmen

#### Lineare Regression

- Optimierung mittels Gradient Descent
- Unterstützung mehrerer Eingangsfeatures
- Konfigurierbare Learning Rate
- Konfigurierbare Anzahl von Trainingsepochen
- Vorhersagen für neue Daten
- Bewertung mittels Mean Squared Error
- Validierung des Modellzustands

#### K-Means Clustering

- Konfigurierbare Anzahl von Clustern
- Konfigurierbare maximale Anzahl von Iterationen
- Reproduzierbare Initialisierung durch einen Random Seed
- Automatische Konvergenzerkennung
- Behandlung leerer Cluster
- Vorhersage von Clusterzugehörigkeiten
- Zugriff auf berechnete Zentroiden

## Voraussetzungen

Für das Projekt werden folgende Komponenten benötigt:

- C++17-kompatibler Compiler
- CMake ab Version 3.16

Das Projekt besitzt aktuell keine externen Abhängigkeiten.

### Unterstützte Compiler

Das Projekt sollte mit aktuellen Versionen folgender Compiler funktionieren:

- GCC
- Clang
- MSVC

---

# Build

Repository klonen:

```bash
git clone git@github.com:bagdanchik007/cp-ml-library.git
cd cp-ml-library
```

Projekt konfigurieren:

```bash
cmake -S . -B build
```

Projekt kompilieren:

```bash
cmake --build build
```

Für einen parallelen Build:

```bash
cmake --build build -j
```

---

# Demo ausführen

Nach dem erfolgreichen Build:

```bash
./build/ml_demo
```

Die Demo zeigt die Verwendung von Linear Regression und K-Means Clustering.

Beispielausgabe:

```text
=== Cpp ML Library Demo ===

--- Linear Regression ---

Learned weights: 2.1398 2.8315
Bias: 4.8960

Predictions for test set: 29.0608 34.0322
MSE on training set: 0.0057

--- K-Means Clustering ---

Cluster labels:
0 0 0 1 1 1 0 1

Centroids:

[   1.2000   1.2500 ]
[   8.1250   7.9750 ]

Demo finished successfully.
```

---

# Tests

Zuerst das Projekt konfigurieren und kompilieren:

```bash
cmake -S . -B build
cmake --build build
```

Alle Tests ausführen:

```bash
ctest --test-dir build --output-on-failure
```

Alternativ können die einzelnen Tests direkt gestartet werden:

```bash
./build/matrix_tests
```

```bash
./build/linear_regression_tests
```

```bash
./build/kmeans_tests
```

Die Tests überprüfen unter anderem:

- Matrixoperationen
- Matrixarithmetik
- Fehlerbehandlung bei ungültigen Matrixoperationen
- Training und Vorhersagen der linearen Regression
- K-Means Clustering
- Validierung ungültiger Eingabedaten

---

# Verwendung

Die gesamte Bibliothek kann über den zentralen Header eingebunden werden:

```cpp
#include "ml/ml.hpp"
```

## Matrix

```cpp
#include "ml/matrix.hpp"

using namespace ml;

Matrix A = {
    {1.0, 2.0},
    {3.0, 4.0}
};

Matrix B = {
    {5.0, 6.0},
    {7.0, 8.0}
};

Matrix sum = A + B;
Matrix difference = A - B;
Matrix scaled = A * 2.0;

Matrix transposed = A.transpose();

A.print();
```

### Matrix-Vektor-Multiplikation

```cpp
Matrix A = {
    {1.0, 2.0},
    {3.0, 4.0}
};

std::vector<double> vector = {1.0, 2.0};

std::vector<double> result = A * vector;
```

---

# Lineare Regression

```cpp
#include "ml/ml.hpp"

using namespace ml;

Matrix X = {
    {1.0},
    {2.0},
    {3.0},
    {4.0},
    {5.0}
};

std::vector<double> y = {
    5.0,
    8.0,
    11.0,
    14.0,
    17.0
};

LinearRegression model;

model.fit(
    X,
    y,
    0.05,
    3000,
    false
);

auto predictions = model.predict(X);

double mse = model.score(X, y);
```

Das Beispiel lernt näherungsweise die folgende Funktion:

```text
y = 3x + 2
```

---

# K-Means Clustering

```cpp
#include "ml/ml.hpp"

using namespace ml;

Matrix data = {
    {1.0, 1.0},
    {1.2, 1.1},
    {0.9, 1.3},

    {8.0, 8.0},
    {8.1, 7.9},
    {7.8, 8.2}
};

KMeans model(
    2,
    100,
    42
);

auto labels = model.fit_predict(data);

model.centroids().print();
```

Training und Vorhersage können auch getrennt durchgeführt werden:

```cpp
KMeans model(2);

model.fit(data);

auto labels = model.predict(data);
```

---

# Projektstruktur

```text
CppMLLibrary
│
├── include/
│   └── ml/
│       ├── matrix.hpp
│       ├── linear_regression.hpp
│       ├── kmeans.hpp
│       └── ml.hpp
│
├── src/
│   ├── linear_regression.cpp
│   └── kmeans.cpp
│
├── examples/
│   ├── demo.cpp
│   ├── linear_regression_demo.cpp
│   └── kmeans_demo.cpp
│
├── tests/
│   ├── matrix_tests.cpp
│   ├── linear_regression_tests.cpp
│   └── kmeans_tests.cpp
│
├── CMakeLists.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

# Architektur und Designziele

Das Projekt wurde mit folgenden Zielen entwickelt:

- Klarer und verständlicher C++-Code
- Moderne C++-Standards
- Geringe Anzahl an Abhängigkeiten
- Saubere Trennung zwischen öffentlicher API und Implementierung
- Reproduzierbare Ergebnisse
- Validierung von Eingabedaten
- Fehlerbehandlung
- Automatisierte Tests
- CMake als Build-System

Das Projekt ist primär als Lern-, Portfolio- und Demonstrationsprojekt konzipiert.

Für produktive Anwendungen mit großen Datenmengen oder hohen Performance-Anforderungen sind spezialisierte Bibliotheken wie Eigen, BLAS oder etablierte Machine-Learning-Frameworks besser geeignet.

---

# Roadmap

## Core

- [x] Dense Matrix Representation
- [x] Matrix Addition
- [x] Matrix Subtraction
- [x] Matrix Multiplication
- [x] Matrix-Vector Multiplication
- [x] Scalar Operations
- [x] Matrix Transpose
- [x] Random Matrix Generation
- [x] Input Validation

## Machine Learning

- [x] Lineare Regression
- [x] K-Means Clustering
- [ ] Logistische Regression
- [ ] Decision Tree
- [ ] K-Nearest Neighbors
- [ ] Naive Bayes
- [ ] Principal Component Analysis

## Erweiterungen

- [ ] Weitere Matrixoperationen
- [ ] Feature Normalization
- [ ] Train/Test Split
- [ ] Zusätzliche Evaluierungsmetriken
- [ ] Cross-Validation
- [ ] Verbesserte K-Means-Initialisierung
- [ ] Serialisierung trainierter Modelle

## Engineering

- [x] CMake Build System
- [x] Unit Tests
- [x] CTest Integration
- [x] Compiler Warnings
- [ ] Code Coverage
- [ ] Continuous Integration
- [ ] Code Formatting
- [ ] Static Analysis

---

# Mitwirken

Verbesserungsvorschläge, Issues und Pull Requests sind willkommen.

Falls du einen Fehler findest oder eine Idee zur Erweiterung des Projekts hast, kannst du gerne ein Issue erstellen oder einen Pull Request einreichen.

---

# Lizenz

Dieses Projekt wird unter der im Repository enthaltenen Lizenz veröffentlicht.

Weitere Informationen befinden sich in der Datei `LICENSE`.

---

# Autor

Bohdan Skibitskyi

GitHub: :contentReference[oaicite:0]{index=0}

---

# Projektstatus

Das Projekt befindet sich in aktiver Entwicklung.

Ziel des Projekts ist es, die Bibliothek schrittweise um weitere Machine-Learning-Algorithmen, mathematische Werkzeuge, Evaluierungsmethoden und professionelle Entwicklerwerkzeuge zu erweitern.
