#include "Graph.h"
#include "./reference/ref_basic_pq.h"
#include "node.h"

#include <iostream>

int main() {
    int rows = 30;
    int cols = 30;
    std::pair<int,int> startNode = {2, 2};
    std::pair<int, int> endNode = {28,25};

    std::vector<std::pair<int,int>> obstactles = {{2,1}};

    // Create a 5x5 graph
    Graph graph(rows, cols, startNode, endNode);
    // graph.setNonTraversable(obstactles);
    graph.generateRandomLines(60, 7);

    // std::cout << "Graph Structure (1 = Traversible, 0 = Non-Traversible):" << std::endl;
    // graph.printGraph();

    graph.run_a_star(startNode, endNode, "manhattan");

    // std::cout << "\nGraph Neighbors:" << std::endl;
    // graph.print_neighbours();

    // ref_pq::PriorityQueue pq(5);

    // // Insert some nodes

    // std::cout << "Inserting Nodes...\n";
    // pq.insert(Node(0, 0, true)); // f = 0
    // pq.insert(Node(1, 1, true)); // f = 0
    // pq.insert(Node(2, 2, true)); // f = 0
    // pq.insert(Node(3, 3, true)); // f = 0


    // // Extract the highest-priority node
    // Node topNode = pq.extractNode();
    // std::cout << "Extracted Node: (" << topNode.x << ", " << topNode.y << ")\n";
    // Node topNode2 = pq.extractNode();
    // std::cout << "Extracted Node: (" << topNode2.x << ", " << topNode2.y << ")\n";
    // Node topNode3 = pq.extractNode();
    // std::cout << "Extracted Node: (" << topNode3.x << ", " << topNode3.y << ")\n";
}




