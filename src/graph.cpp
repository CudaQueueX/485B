#include "Graph.h"
#include <iostream>
#include <cmath>


// Constructor to initialize the grid and set edge weights to -infinity
Graph::Graph(int rows, int cols, std::pair<int,int> startNode, std::pair<int,int> endNode)
            : rows(rows), cols(cols), startNode(startNode), endNode(endNode) {
    std::cout << "================================================================================" << std::endl;
    std::cout << "Initializing a " << rows << "x" << cols << " grid." << std::endl;
    std::cout << "Start Node: (" << startNode.first << ", " << startNode.second << ")" << std::endl;
    std::cout << "End Node: (" << endNode.first << ", " << endNode.second << ")" << std::endl;
    std::cout << "================================================================================" << std::endl;


    std::cout << "Creating rows and columns \t\t";
    for (int i = 0; i < rows; ++i) {
        std::vector<Node> row;
        for (int j = 0; j < cols; ++j) {
            row.emplace_back(i, j, true);  // Create Node at (i, j) setting it default to traversible
        }
        grid.push_back(row);
    }
    std::cout << "Rows and columns completed" << std::endl;
    std::cout << "--------------------------------------------------------------------------------" << std::endl;

    std::cout << "Setting up the edges \t\t:";
    for(int i = 0; i < rows; ++i) {
        for(int j = 0; j < cols; ++j) {
            if(i == 0 || i == rows - 1 || j == 0 || j == cols - 1) {
                grid[i][j].isTraversible = false;
            }
        }
    }
    std::cout << "Edges set up" << std::endl;
    std::cout << "--------------------------------------------------------------------------------" << std::endl;

    std::cout << "Setting up the neighbors \t\t:";
    // Initialize the edges between adjacent nodes with weight -infinity
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            for (auto [dx, dy] : DIRECTIONS) {
                int newX = i + dx;
                int newY = j + dy;
                if (isValid(newX, newY)) {
                    grid[i][j].addNeighbour(newX, newY, 1);
                }
            }
        }
    }
    std::cout << "Neighbors set up" << std::endl;
    std::cout << "--------------------------------------------------------------------------------" << std::endl;
    std::cout << "Graph Initialized" << std::endl;
}

// Check if the cell is within bounds
bool Graph::isValid(int x, int y) const {
    return x >= 0 && x < rows && y >= 0 && y < cols;
}

// Print the entire graph
void Graph::print_neighbours() const {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            grid[i][j].printNeighbours();
            }
        }    
}

void Graph::printGraph() const {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (i == startNode.first && j == startNode.second)std::cout << " S";
            else if (i == endNode.first && j == endNode.second) std::cout << " T";
            else std::cout << " " << (grid[i][j].isTraversible ? '0' : 'X');
        }
        std::cout << std::endl;
    }
}


int Graph::distance_euclidean(std::pair<int, int> p1, std::pair<int, int> p2) const {
    return sqrt(pow(p1.first - p2.first, 2) + pow(p1.second - p2.second, 2));
}

int Graph::distance_manhattan(std::pair<int, int> p1, std::pair<int, int> p2) const {
    return abs(p1.first - p2.first) + abs(p1.second - p2.second);
}