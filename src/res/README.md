# Solutions

Solution to knapsack problem is a choice of elements.

Solutions can be implemented in two ways:
* list of indexes of elements
* bit vector

second option is prefered.

## Solution class

It is required that any class that tries to be a `solution class` has the following:
* `constructor` that takes `size_t` as an argument that indicates size of the [instance](../inst/README.md) (maximum amount of elements that can land in the solution)
* default `constructor` (assumes size is 0)
* `has(size_t)`, `add(size_t)`, `remove(size_t)` methods for checking if given element is present, adding element, removing element
* `size()` method that returns size provided in constructor
* `selected_count()` method that should return count of added elements
* `std::ostream& operator<< (std::ostream&, const bit_vector&)` operator for printing the solution

## In practice

``` c++
class solution {
public:
    solution();
    solution(size_t n);
    void add(size_t i);
    void remove(size_t i);
    bool has(size_t i) const;
    size_t size() const;
    size_t selected_count() const;
    friend std::ostream& operator<< (std::ostream& stream, const bit_vector& bv);
};
```

All solution classes reside in `gs::res` or `gs::cuda::res` namespaces.

## Available Solution Classes

### CPU

``` c++
namespace gs::res
```

* [`bit_vector`](./bit_vector.hpp)

### CUDA

``` c++
namespace gs::cuda::res
```

* [`solution8`](./cuda_solution.hpp)
* [`solution16`](./cuda_solution.hpp)
* [`solution32`](./cuda_solution.hpp)
* [`solution64`](./cuda_solution.hpp)

These solution classes can be used on CPU but have very tight max size limit and are best suited for CUDA BruteForce solvers.