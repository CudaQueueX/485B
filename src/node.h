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
    bool isTraversible = true; // true means the node can be traversed to (ie not a barrier node) 
    
    std::pair<int,int> parent; // parent node
    
    int g = 0; // cost from starting node (s) to current node (n)
    int h = 0; // from current node (s) to the end node (t)
    int f = 0; // f(n) = g(n) + h(n)
    std::vector<std::pair<std::pair<int, int>, double>> neighbours;  // Neighbors with weights

    Node(int x, int y, bool isTraversible);

    void addNeighbour(int x, int y, double weight);
    void printNeighbours() const;
};

#endif // NODE_H

