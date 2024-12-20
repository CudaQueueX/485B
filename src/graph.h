#ifndef GRAPH_H
#define GRAPH_H

#include <vector>
#include "Node.h"

class Graph {
public:
    const int rows;
    const int cols;
    const std::pair<int,int> startNode;
    const std::pair<int,int> endNode;
    
    // std::vector<std::vector<int>> adjacency;
    const std::vector<std::pair<int, int>> open; // contains nodes which are being considered for traversal. 
    const std::vector<std::pair<int, int>> closed; // contains nodes which have been visited. 

    std::vector<std::vector<Node>> grid;
    const std::vector<std::pair<int, int>> DIRECTIONS = {
        {0, 1},   // Right
        {0, -1},  // Left
        {1, 0},   // Down
        {-1, 0}   // Up
    };



    Graph(int rows, int cols, std::pair<int,int> startNode, std::pair<int,int> endNode);

    bool isValid(int x, int y) const;
    void print_neighbours() const;
    void printGraph() const;
    void printAdajacency() const;

    int distance_euclidean(std::pair<int, int> p1, std::pair<int, int> p2) const;
    int distance_manhattan(std::pair<int, int> p1, std::pair<int, int> p2) const;
};

#endif // GRAPH_H