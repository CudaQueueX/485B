// node.cpp

#include "Node.h"

// Constructor initializes the node's coordinates
Node::Node(int x, int y) : x(x), y(y) {}

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

