# Code Documentation

* [Instances](./inst/README.md)
* [Solutions](./res/README.md)
* [Solvers](./solvers/README.md)
* [SolverRunner](#solver-runner)
* [Validator](#validator)

## Solver Runner

``` c++
namespace gs {
    template <typename Solver>
    class SolverRunner;
}
```

Wrapper for [solver](./solvers/README.md#solvers) type that can measure the time of `Solver::sove` call and also validate the answer.

### Usage

``` c++
gs::SolverRunner<MySolver>::run(instance, formatString, ostream);
```

or if `MySolver::solve` accepts any parameters:

``` c++
gs::SolverRunner<MySolver>::run<int, float>(instance, formatString, ostream, 5, 4.32f);
```

where:
* `instance` is instance of [Instance class](./inst/README.md), must be same as `Solver::instance_t`.
* `formatString` is aany string that contains [formatting parameters](#formatting-parameters)
* `ostream` is an `std::ostream` instance such as `std::cout`, `std::cout` is the default

#### formatting parameters
* `{solver name}` - name of the solver
* `{time}` - solve time in seconds
* `{instance}` - print problem instance
* `{result}` - print solution
* `{instance size}` - size of the instance
* `{result size}` - size of the result
* `{instance N}` - limit N of the instance
* `{result N}` - cost N of the result
* `{unique}` - unique validation result
* `{fitting}` - fitting validation result

#### Example

``` c++
gs::SolverRunner<MySolver>::run(
    MyInstance("instances/my-instance1.txt"),
    "{solver name}:\nresult: {result}\ntime: {time}s\n", std::cout
);
```

## Validator

``` c++
namespace gs {
    template <typename Solver>
    class Validator;
}
```

Validator can be found in [Validator.hpp](./Validator.hpp) file.

It is used internally by [SolverRunner](#solver-runner) for result validation.

Validator can check if [solution](./res/README.md) is a structure required by [instance](./inst/README.md) and if elements of the [solution](./res/README.md) fit inside limits of the [instance](./inst/README.md).

Usage of validator inside [Solvers](./solvers/README.md) is generaly discouraged as custom methods tailored for solver algorithm should yield better results.