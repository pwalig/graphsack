# Solvers

## Anatomy of a Solver

It is required that any class that tries to be a solver has the following:
* `typedef` or `using` for `instance_t` - type representing problem [instance](../inst/README.md)
* `typedef` or `using` for `solution_t` - type representing [solution](../res/README.md)
* `static` `std::string` `name` containing name of the solver
* `static` `solution_t` `solve(instance_t)` - method for solving the problem

### Example

``` c++
class Solver {
public:
    using instance_t = /*insert instance type here*/;
    using solution_t = /*insert solution type here*/;
    static std::string name;
    static solution_t solve(const instance_t& instance);
};
```

### In practice

Solvers are often parametrised and have disabled constructors:

``` c++
template <typename InstanceT, typename SolutionT>
class Solver {
public:
    using instance_t = InstanceT;
    using solution_t = SolutionT;
    inline static std::string name = "Solver";

    Solver() = delete;

    template <typename T = size_t>
    static solution_t solve(const instance_t& instance, T parameter1, int parameter2 = 1);
};
```

`Solver::solve` needs to be callable by [`SolverRunner`](../README.md#solver-runner), so all template parameters must be deductable from argument types or defaults must be provided.

All solver classes reside in `gs::solver` or `gs::cuda::solver` namespaces.

## Usage

``` c++
Solver::instance_t instance("path to instance file");
auto result = Solver::solve(instance);
```

## Available Solvers

### Single Threaded CPU

``` c++
namespace gs::solver
```

* [`Greedy`](./Greedy.hpp)
* [`GHS`](./GHS.hpp)
* [`GRASP`](./GRASP.hpp)
* [`Dynamic`](./Dynamic.hpp)
* [`BruteForce`](./BruteForce.hpp)
* [`PathBruteForce`](./PathBruteForce.hpp)

### CUDA

``` c++
namespace gs::cuda::solver
```

* [`BruteForce`](./CudaBruteForce.cu)
* [`BruteForce32`](./CudaBruteForce.cu)
* [`BruteForce64`](./CudaBruteForce.cu)

### Other

``` c++
namespace gs::solver
```

* [`MultiRun`](./MultiRun.hpp) - wrapper for any solver to be launched multiple times with different parameters and best solution selected.