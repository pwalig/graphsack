# graphsack

Solver for multiple knapsack problem variations.

# Problem

The main problem that this solver attempts to solve is a graph modification on the knapsack problem.

Items, apart from having value and weight, are also vertices in an directed graph. The goal is to find such cycle in the graph that items coresponding to selected vertices have the largest possible value, while fitting in the knapsack.

Additionaly instead of weights being single values, they are vectors of values. In order to fit inside the knpasack, vector sum of weights must be smaller than knapsack size vector in every dimension. This modification is known as a [multidimentional knapsack problem](https://en.wikipedia.org/wiki/Knapsack_problem#Multi-dimensional_weight).

# Implemented Algorithms / Solvers

## Helper algorithms

* DFS (is_structure) (both iterative and recursive)
* DFS (is_structure_possible) (recursive)
* Bitonic sort (CUDA)
* GNP graph generator
* Pick best element reduction (CUDA)

## Problem solving algorithms

* Brute Force (Iterative)
* Parallel Brute Force (Open MP)
* Parallel Brute Force (CUDA)
* Greedy Randomized Adaptive Search
* Parallel Greedy Randomized Adaptive Search (Open MP)
* Parallel Greedy Randomized Adaptive Search (CUDA)
* Greedy Algorithm
* Greedy Heuristic Search (recursive)
* Dynamic Programming
* Path constructing Brute Force (recursive)

# Implementation

Algoritms are designed in such a way that they can accept instances of the problem in any form as long as they follow a common interface. Instance interface description can be found [here](./src/inst/README.md). Instances may differ for example in memory layout or graph representation, common interface allows algorithms to interact with any type of instance. Mechamism implemented with C++ templates. Static dispatch based on graph representation is yet to be implemented. For now algorithms are more or less geared for operating on adjacency list.

Additionaly solvers can operate on any solution type (again, thanks to the common interface described [here](./src/res/README.md)). Solution to the knapsack problem can be represented in multiple ways:

* bit vector
* set of selected item id's
* list of selected items

Since graph modification and multidimentionality are generalizations of the knapsack problem, solver can solve the base problem too.

# Building

## Dependencies

Project depends on [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) being installed on your system.

## CMake

Project can be built with [CMake](https://cmake.org).

Run the following.

```
git clone https://github.com/pwalig/graphsack.git
cd graphsack
mkdir build
cd build
cmake ..
cmake --build .
```

to get release build with MSVC on Windows use:
```
cmake --build . --config Release
```

## Visual Studio 2022

Project can be built with [Visual Studio](https://visualstudio.microsoft.com/).

Just open grahpsack.sln and run.

Tested only on Visual Studio 2022 other versions are not guaranteed not work.

# [Code Documentation](./src/README.md)

* [Instances](./src/inst/README.md)
* [Solutions](./src/res/README.md)
* [Solvers](./src/solvers/README.md)
* [SolverRunner](/src/README.md#solver-runner)
* [Validator](/src/README.md#validator)
