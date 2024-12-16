#include "Graph.h"
#include <iostream>
#include <limits>

// Constructor to initialize the grid and set edge weights to -infinity
Graph::Graph(int rows, int cols) : rows(rows), cols(cols) {
    for (int i = 0; i < rows; ++i) {
        std::vector<Node> row;
        for (int j = 0; j < cols; ++j) {
            row.emplace_back(i, j);  // Create Node at (i, j)
        }
        grid.push_back(row);
    }

    // Initialize the edges between adjacent nodes with weight -infinity
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            for (auto [dx, dy] : DIRECTIONS) {
                int newX = i + dx;
                int newY = j + dy;
                if (isValid(newX, newY)) {
                    grid[i][j].addNeighbour(newX, newY, -std::numeric_limits<double>::infinity());
                }
            }
        }
    }
}

// Check if the cell is within bounds
bool Graph::isValid(int x, int y) const {
    return x >= 0 && x < rows && y >= 0 && y < cols;
}

// Print the entire graph
void Graph::printGraph() const {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            grid[i][j].printNeighbours();
        }
    }
}
