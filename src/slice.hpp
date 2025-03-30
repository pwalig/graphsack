#pragma once
#include "iterator.hpp"

template <typename T, typename sizeT = size_t>
class slice {
public:
	slice(T* Ptr, sizeT Size) : ptr(Ptr), size(Size) {}
	T* ptr;
	const sizeT size;

	inline T& operator[] (sizeT i) { assert(i < size); return ptr[i]; }
	inline const T& operator[] (sizeT i) const { assert(i < size); return ptr[i]; }

	trivial_iterator_defs(T, ptr, size)
};
