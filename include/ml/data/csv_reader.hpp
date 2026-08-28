#pragma once

#include "ml/data/dataset.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ml {

/** Reads a numeric CSV whose target is stored in the last column. */
inline Dataset read_csv(const std::string& path, bool has_header = true, char delimiter = ',') {
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("read_csv: cannot open file '" + path + "'");
    }
    std::string line;
    if (has_header) {
        std::getline(stream, line);
    }
    std::vector<std::vector<double>> rows;
    size_t column_count = 0;
    while (std::getline(stream, line)) {
        if (line.empty()) continue;
        std::stringstream values(line);
        std::string cell;
        std::vector<double> row;
        while (std::getline(values, cell, delimiter)) {
            size_t parsed = 0;
            const double value = std::stod(cell, &parsed);
            if (parsed != cell.size()) {
                throw std::invalid_argument("read_csv: non-numeric cell in '" + path + "'");
            }
            row.push_back(value);
        }
        if (row.size() < 2) {
            throw std::invalid_argument("read_csv: each row needs at least one feature and one target");
        }
        if (column_count == 0) column_count = row.size();
        if (row.size() != column_count) {
            throw std::invalid_argument("read_csv: rows have inconsistent column counts");
        }
        rows.push_back(std::move(row));
    }
    if (rows.empty()) {
        throw std::invalid_argument("read_csv: file contains no data rows");
    }
    Matrix features(rows.size(), column_count - 1);
    std::vector<double> targets(rows.size());
    for (size_t row = 0; row < rows.size(); ++row) {
        for (size_t column = 0; column + 1 < column_count; ++column) features(row, column) = rows[row][column];
        targets[row] = rows[row].back();
    }
    return Dataset(std::move(features), std::move(targets));
}

} // namespace ml
