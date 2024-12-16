#ifndef GRAPH_H
#define GRAPH_H

#include <vector>
#include "Node.h"

class Graph {
public:
    int rows;
    int cols;
    std::vector<std::vector<Node>> grid;
    const std::vector<std::pair<int, int>> DIRECTIONS = {
        {0, 1},   // Right
        {0, -1},  // Left
        {1, 0},   // Down
        {-1, 0}   // Up
    };

    Graph(int rows, int cols);

    bool isValid(int x, int y) const;
    void printGraph() const;
};

#endif // GRAPH_H

