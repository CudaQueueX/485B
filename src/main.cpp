#include "Graph.h"

int main() {
    int rows = 3;
    int cols = 3;

    // Create a 3x3 graph
    Graph graph(rows, cols);

    // Print the graph's adjacency list with weights initialized to -infinity
    graph.printGraph();

    return 0;
}
