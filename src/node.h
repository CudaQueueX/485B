//node.h

#ifndef NODE_H
#define NODE_H

#include <vector>
#include <utility>
#include <iostream>
#include <limits>

class Node {
public:
    int x;
    int y;
    std::vector<std::pair<std::pair<int, int>, double>> neighbours;  // Neighbors with weights

    Node(int x, int y);

    void addNeighbour(int x, int y, double weight);
    void printNeighbours() const;
};

#endif // NODE_H

