#include "Graph.h"
#include "./reference/ref_basic_pq.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <random>


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
    std::cout << "--------------------------------------------------------------------------------" << std::endl;

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

Node Graph::get_node(std::pair<int, int> node) const {
    return grid[node.first][node.second];
}

std::vector<Node> Graph::get_neighbours(std::pair<int, int> node) const {
    std::vector<Node> neighbours;
    
    // grid[node.first][node.second].printNeighbours();
    // iterate through the neighbours of the node and extract the nodes
    for (const auto& [neighbour, weight] : grid[node.first][node.second].neighbours) neighbours.push_back(grid[neighbour.first][neighbour.second]);
    
    return neighbours;
}

int Graph::distance_calculation(Node update_node, std::pair<int,int> start, std::pair<int,int> end, std::string heuristic) {
    if(heuristic == "euclidean") {
        update_node.h = distance_euclidean({update_node.x, update_node.y}, end);
    }
    return update_node.h = distance_manhattan({update_node.x, update_node.y}, end);
}

// Update the Graph implementation for improved clarity and correctness
void Graph::run_a_star(std::pair<int, int> start, std::pair<int, int> end, std::string heuristic) {
    std::cout << "Running A* Algorithm" << std::endl;
    std::cout << "--------------------------------------------------------------------------------" << std::endl;
    

    // Priority queue and auxiliary lists
    ref_pq::PriorityQueue PQ(rows * cols);
    std::vector<std::vector<bool>> inClosed(rows, std::vector<bool>(cols, false));
    std::vector<std::vector<bool>> inOpen(rows, std::vector<bool>(cols, false));

    // Initialize start node
    Node& startNode = grid[start.first][start.second];
    startNode.g = 0;
    startNode.h = distance_calculation(startNode, start, end, heuristic);
    startNode.f = startNode.g + startNode.h;
    PQ.insert(startNode);
    inOpen[start.first][start.second] = true;

    while (!PQ.isEmpty()) {
        Node current = PQ.extractNode();
        // std::cout << "Current Node: (" << current.x << ", " << current.y << ")" << std::endl;

        // Exit condition
        if (current.x == end.first && current.y == end.second) {
            std::cout << "Path Found!" << std::endl;
            std::cout << "--------------------------------------------------------------------------------" << std::endl;
            // std::cout << "Current Node: (" << current.x << ", " << current.y << ")" << "End Node: (" << end.first << ", " << end.second << ")" << std::endl;
            auto path = reconstruct_path(current);
            printPath(path);
            return;
        }

        // Mark current as processed
        inClosed[current.x][current.y] = true;

        // Process neighbors
        for (const auto& [neighborCoord, weight] : current.neighbours) {
            int nx = neighborCoord.first, ny = neighborCoord.second;

            // Skip if not traversible or already closed
            if (!grid[nx][ny].isTraversible || inClosed[nx][ny]) {
                continue;
                }


            Node& neighbor = grid[nx][ny];
            int tentative_g = current.g + weight;

            // If new path to neighbor is shorter
            if (!inOpen[nx][ny] || tentative_g < neighbor.g) {
                neighbor.g = tentative_g;
                neighbor.h = distance_calculation(neighbor, start, end, heuristic);
                neighbor.f = neighbor.g + neighbor.h;
                neighbor.parent = {current.x, current.y};

                if (!inOpen[nx][ny]) {
                    PQ.insert(neighbor);
                    inOpen[nx][ny] = true;
                }
            }
        }
    }

    std::cout << "No Path Found" << std::endl;
}

std::vector<Node> Graph::reconstruct_path(Node current) {
    std::vector<Node> path;

    // std::cout << "Reconstructing Path" << std::endl;

    // Follow the parent chain until reaching the start node
    while (current.parent != std::make_pair(-1, -1)) {
        path.push_back(current);
        current = grid[current.parent.first][current.parent.second]; // Get the parent node
        // std::cout << "Current Node: (" << current.x << ", " << current.y << ")" << std::endl;
        if(current.parent.first == 0 && current.parent.second == 0) break;
    }

    path.push_back(current); // Add the start node
    std::reverse(path.begin(), path.end()); // Reverse to get the path from start to end
    return path;
}


int Graph::distance_euclidean(std::pair<int, int> p1, std::pair<int, int> p2) const {
    return sqrt(pow(p1.first - p2.first, 2) + pow(p1.second - p2.second, 2));
}

int Graph::distance_manhattan(std::pair<int, int> p1, std::pair<int, int> p2) const {
    return abs(p1.first - p2.first) + abs(p1.second - p2.second);
}

void Graph::printPath(const std::vector<Node>& path) const {
    // Create a copy of the grid to mark the path
    std::vector<std::vector<char>> gridRepresentation(rows, std::vector<char>(cols, '0'));

    // Mark non-traversable tiles
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (!grid[i][j].isTraversible) {
                gridRepresentation[i][j] = 'X';
            }
        }
    }

    // Mark the path
    for (const auto& node : path) {
        gridRepresentation[node.x][node.y] = '*';
    }

    // Mark the start and end nodes
    gridRepresentation[startNode.first][startNode.second] = 'S';
    gridRepresentation[endNode.first][endNode.second] = 'T';

    // Print the grid
    std::cout << "Path Visualization:" << std::endl;
    for (const auto& row : gridRepresentation) {
        for (const auto& cell : row) {
            std::cout << cell << " ";
        }
        std::cout << std::endl;
    }
}

void Graph::setNonTraversable(const std::vector<std::pair<int, int>>& nodes){
    for (const auto& node : nodes) {
        int x = node.first;
        int y = node.second;
        if (isValid(x, y)) {
            grid[x][y].isTraversible = false;
        } else {
            std::cout << "Invalid node (" << x << ", " << y << "), skipping..." << std::endl;
        }
    }    
}

#include <random>

void Graph::generateRandomLines(int numLines, int maxLineLength) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> rowDist(0, rows - 1);
    std::uniform_int_distribution<> colDist(0, cols - 1);
    std::uniform_int_distribution<> directionDist(0, 1); // 0 for horizontal, 1 for vertical
    std::uniform_int_distribution<> lengthDist(1, maxLineLength);

    for (int i = 0; i < numLines; ++i) {
        int startX = rowDist(gen);
        int startY = colDist(gen);
        int direction = directionDist(gen);
        int lineLength = lengthDist(gen);

        for (int j = 0; j < lineLength; ++j) {
            int x = startX + (direction == 1 ? j : 0); // Vertical increment
            int y = startY + (direction == 0 ? j : 0); // Horizontal increment

            // Check if within bounds
            if (isValid(x, y)) {
                grid[x][y].isTraversible = false;
            }
        }
    }
}

