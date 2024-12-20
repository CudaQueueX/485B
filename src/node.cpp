// node.cpp

#include "Node.h"

// Constructor initializes the node's coordinates
Node::Node(int x, int y, bool isTraversible) : x(x), y(y), isTraversible(isTraversible), g(0), h(0), f(0) {}

// Add a neighbor with a weight
void Node::addNeighbour(int neighborX, int neighborY, double weight) {
    neighbours.push_back({{neighborX, neighborY}, weight});
}

// Print the neighbors and the weight of this node
void Node::printNeighbours() const {
    std::cout << "Node (" << x << ", " << y << ") Neighbours: ";
    for (const auto& neighbor : neighbours) {
        std::cout << "(" << neighbor.first.first << ", " << neighbor.first.second 
                  << ", weight: " << neighbor.second << ") ";
    }
    std::cout << std::endl;
}

