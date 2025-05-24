# graphsack

Solver for multiple knapsack problem variations.

# Building

## Dependencies

Project depends on [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) being installed on your system.

## CMake

Project can be built with [CMake](https://cmake.org).

Run the following.

```
git clone https://github.com/pwalig/graphsack.git
cd DNA-sequencer
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
