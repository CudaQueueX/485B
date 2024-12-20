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


    std::vector<std::vector<Node>> grid;
    const std::vector<std::pair<int, int>> DIRECTIONS = {
        {0, 1},   // Right
        {0, -1},  // Left
        {1, 0},   // Down
        {-1, 0}   // Up
    };



    Graph(int rows, int cols, std::pair<int,int> startNode, std::pair<int,int> endNode); // Constructor

    

    bool isValid(int x, int y) const;
    void print_neighbours() const;
    void printGraph() const;
    void printAdajacency() const;
    void printPath(const std::vector<Node>& path) const;

    Node get_node(std::pair<int, int> node) const; // Returns the node specified by coordinates
    std::vector<Node> get_neighbours(std::pair<int, int> node) const; // Returns the neighbours of the node specified by coordinates
    int distance_calculation(Node update_node, std::pair<int,int> start, std::pair<int,int> end, std::string heurtistic); 
    // A-star
    void run_a_star(std::pair<int,int> start, std::pair<int,int> end, std::string heurtistic);
    std::vector<Node> reconstruct_path(Node current);

    void setNonTraversable(const std::vector<std::pair<int, int>>& nodes);
    void generateRandomLines(int numLines, int maxLineLength);

    // Heuristics
    int distance_euclidean(std::pair<int, int> p1, std::pair<int, int> p2) const;
    int distance_manhattan(std::pair<int, int> p1, std::pair<int, int> p2) const;

    // export graph matrix
    void exportGraph(const std::vector<Node>& path,const std::string& filename) const;
};

#endif // GRAPH_H