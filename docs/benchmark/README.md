# Benchmark-Skripte

Diese Skripte erzeugen die Zahlen und Plots aus
[`../deep-dive-logistic-regression-and-regularization.md`](../deep-dive-logistic-regression-and-regularization.md).

Sie vergleichen `LogisticRegression`, `RidgeRegression` und
`LassoRegression` aus dieser Bibliothek gegen die entsprechenden
scikit-learn-Implementierungen auf identischen synthetischen Datensätzen
mit bekannter wahrer Koeffizientenstruktur.

## Ausführen

Voraussetzungen: Python 3 mit `numpy`, `scikit-learn`, `matplotlib`
(`pip install numpy scikit-learn matplotlib`) sowie ein C++20-fähiger
Compiler.

```bash
python3 generate_data.py        # erzeugt classification_*.csv, regression_*.csv
python3 sklearn_reference.py    # trainiert sklearn-Modelle, schreibt sklearn_results.json
g++ -std=c++20 -O2 -I ../../include cpp_benchmark.cpp -o cpp_benchmark
./cpp_benchmark                 # trainiert unsere Modelle auf denselben Daten
python3 make_plots.py           # erzeugt die Plots in ../images/
```

Die generierten CSV-Dateien sind absichtlich nicht eingecheckt (siehe
`.gitignore`), da sie mit `generate_data.py` deterministisch (fester
Seed) neu erzeugt werden können.
