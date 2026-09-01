# Contributing

Vielen Dank für dein Interesse, zur Cpp ML Library beizutragen!

Beiträge, Verbesserungen, Bugfixes und neue Machine-Learning-Algorithmen sind willkommen.

## Voraussetzungen

Für die Entwicklung benötigst du:

* Einen C++17-kompatiblen Compiler
* CMake
* Git

## Repository klonen

```bash
git clone https://github.com/bagdanchik007/cp-ml-library.git
cd cp-ml-library
```

## Projekt bauen

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel
```

## Tests ausführen

```bash
ctest --test-dir build --output-on-failure
```

Bitte stelle sicher, dass alle Tests erfolgreich sind, bevor du einen Pull Request erstellst.

## Code Style

* Verwende modernes C++17.
* Schreibe verständlichen und gut strukturierten Code.
* Verwende aussagekräftige Namen für Klassen, Funktionen und Variablen.
* Halte Funktionen möglichst klein und fokussiert.
* Vermeide unnötige Abhängigkeiten.
* Die Library soll dependency-free bleiben.

## Neue Features

Wenn du ein neues Feature oder einen Algorithmus hinzufügst:

1. Implementiere das Feature.
2. Füge Unit Tests hinzu.
3. Aktualisiere die öffentliche API, falls erforderlich.
4. Ergänze die Dokumentation.
5. Stelle sicher, dass alle Tests erfolgreich sind.

## Tests

Neue Funktionalität sollte durch Tests abgedeckt werden.

Tests befinden sich im Verzeichnis:

```text
tests/
```

Neue Tests sollten über CMake registriert werden.

## Commits

Verwende kurze und aussagekräftige Commit-Nachrichten.

Beispiele:

```text
feat: add PCA implementation
fix: correct matrix dimension validation
test: add PCA unit tests
docs: improve installation documentation
refactor: simplify matrix operations
```

## Pull Requests

Bevor du einen Pull Request erstellst:

* Stelle sicher, dass das Projekt erfolgreich kompiliert.
* Führe alle Tests aus.
* Beschreibe deine Änderungen klar.
* Füge Tests für neue Funktionalität hinzu.

Vielen Dank für deinen Beitrag!
