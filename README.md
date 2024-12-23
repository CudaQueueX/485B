<a name="readme-top"></a>

<h3 align="center">GPU-Based Batched Priority Queue</h3>

<p align="center">
  An implementation of a Batched Priority Queue (BPQ) optimized for GPUs, serving as the foundation for future development of GPU-accelerated algorithms.
</p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#results">Results</a></li>
    <li><a href="#future-work">Future Work</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

Many modern applications rely on efficient data structures to ensure performance, especially in computationally intensive tasks like pathfinding and graph traversal. A priority queue is essential for algorithms like A*, but traditional implementations such as binary heaps are not inherently parallelizable on GPUs due to thread-level dependencies.

This project focuses on implementing a **Batched Priority Queue (BPQ)** optimized for GPU execution. The BPQ optimizes insertion and deletion operations through batching, making it well-suited for GPU-based implementations. It serves as a foundation for future development of GPU-accelerated algorithms.

**Note:** The A* pathfinding algorithm implementation is planned as future work and is not currently included in this project.

Key Features:

- **Batched Priority Queue (BPQ)**: An efficient priority queue implementation that processes multiple elements simultaneously, improving performance on GPUs.
- **C++ and CUDA Implementations**: Provides both a reference C++ implementation and a parallel CUDA version of the BPQ.
- **Performance Benchmarking**: Includes an experimental harness for testing correctness and benchmarking performance.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

- **CUDA Toolkit**: Required to compile and run the CUDA implementations. Download it from the [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads).
- **C++ Compiler**: A compiler that supports C++11 or later (e.g., GCC, Clang).
- **CMake**: For building the project. Download from the [CMake Official Website](https://cmake.org/download/).
- **NVIDIA GPU**: A CUDA-capable GPU is necessary to run the CUDA implementations.

### Installation

1. **Clone the Repository**

   ```sh
   git clone https://github.com/CudaQueueX/485B.git
   cd 485B
   ```

2. **Create a Build Directory**

   ```sh
   mkdir build
   cd build
   ```

3. **Configure the Project with CMake**

   ```sh
   cmake ..
   ```

4. **Build the Project**

   ```sh
   make
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE -->
## Usage

After building the project, you can run the provided tests and experiment harness to verify correctness and benchmark performance.

### Running the Experiment Harness

Execute the following command from the `build` directory:

```sh
./ExperimentHarness
```

This will run the correctness tests and the following performance tests for both the C++ and CUDA implementations:

- **Single Insertion and Extraction**: Measures the time taken to perform individual insertion and extraction operations.
- **Batch Insertion and Extraction**: Measures the time taken to perform insertion and extraction operations in batches.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- RESULTS -->
## Results

The results of the performance tests can be found [here](https://github.com/CudaQueueX/485B/tree/main/experiments/results).

***The results were obtained on a machine with the following hardware specifications:***

- **CPU**: Intel i7-10700K
- **GPU**: NVIDIA RTX 3060 with 12 GB VRAM

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- FUTURE WORK -->
## Future Work

- **GPU-Accelerated A\***: Implementing the A* pathfinding algorithm optimized for GPU execution using the Batched Priority Queue. This will enable efficient pathfinding in large graphs by leveraging GPU parallelism.

We plan to extend this project by incorporating the A* pathfinding algorithm in the future. The current BPQ implementation lays the groundwork for this development.

<p align="right">(<a href="#readme-top">back to top</a>)</p>
