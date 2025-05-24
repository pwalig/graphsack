# Instances

## Instance class

There is a lot of requirements that `instance class` needs to fulfill, just see [#in practice](#in-practice) section for extensive example.

## In practice

``` c++
template <typename ValueT, typename WeightT, typename GraphRepresentation>
class instance {
public:
    using value_type = ValueT;
    using weight_type = WeightT;
    using size_type = size_t;

    // using definitions are optional for following types (the types themselves should be some sort of views into the instance - should act like references), many instance classes use slice (defined in slice.hpp)
    // they all should have typedef or using for value_type
    using limits_type = /* */; // could be slice<weight_type>
    using const_limits_type = /* */; // could be slice<const weight_type>
    using weights_type = /* */;
    using const_weights_type = /* */;
    using nexts_type = /* */;
    using const_nexts_type = /* */;

private:
    // weight value and graph here
    gs::structure structureToFind;
    gs::weight_treatment weightTreatment;

public:
    template <typename LIter, typename VIter, typename WIter>
    instance(
        LIter LimitsBegin, LIter LimitsEnd,
        VIter ValuesBegin, VIter ValuesEnd,
        WIter WeightsBegin, WIter WeightsEnd,
        const graphs::adjacency_matrix& graph,
        structure Structure = structure::path,
        gs::weight_treatment WeightTreatment = gs::weight_treatment::full
    );
    size_type size() const; // returns size of the instace
    size_type dim() const; // returns dimention of weight vector

    // return limitId'th element of knapsack capacity vector
    weight_type& limit(size_type limitId);
    const weight_type& limit(size_type limitId) const;

    // return knapsack capacity vector (preferably a view into it)
    limits_type limits();
    const_limits_type limits() const;

    // if weight treatment is first_only then limit index becomes redundant and this method can be used
    weight_type& limit() {
        assert(weightTreatment == gs::weight_treatment::first_only);
        return limits()[0];
    }
    const weight_type& limit() const {
        assert(weightTreatment == gs::weight_treatment::first_only);
        return limits()[0];
    }

    // return value of itemId'th element
    value_type& value(size_type itemId);
    const value_type& value(size_type itemId) const;

    // return weightId'th component of weight vector of itemId'th element
    weight_type& weight(size_type itemId, size_type weightId);
    const weight_type& weight(size_type itemId, size_type weightId) const;

    // return weight vector of itemId'th element
    weights_type weights(size_t itemId);
    const_weights_type weights(size_t itemId) const;

    // return list of elements that itemId has arches to
    nexts_type nexts(size_t itemId);
    const_nexts_type nexts(size_t itemId) const;

    gs::structure& structure_to_find() { return structureToFind; }
    gs::weight_treatment& weight_treatment() { return weightTreatment; }
    const gs::structure& structure_to_find() const { return structureToFind; }
    const gs::weight_treatment& weight_treatment() const { return weightTreatment; }


    // returns true if there is an arch from from to to, false otherwise
    bool has_connection_to(size_type from, size_type to) const;

    // prints instance to stream (output should be readable to a human)
    friend std::ostream& operator<< (std::ostream& stream, const instance& ci);
}

```

All instance classes reside in `gs::inst` or `gs::cuda::inst` namespaces.

## Available Instance Classes

### CPU

``` c++
namespace gs::inst
```

* [`itemlocal_nlist`](./itemlocal_nlist.hpp)

### CUDA

``` c++
namespace gs::cuda::inst
```

* [`instance8`](./cuda_instance.hpp)
* [`instance16`](./cuda_instance.hpp)
* [`instance32`](./cuda_instance.hpp)
* [`instance64`](./cuda_instance.hpp)

These instances can be used on CPU, but they have very tight max size limit and are optimized mainly for copying their contents onto GPU.