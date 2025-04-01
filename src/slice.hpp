#pragma once
#include "iterator.hpp"

template <typename T, typename sizeT = size_t>
class slice {
public:
	using value_type = T;
	using reference = value_type&;
	using pointer = value_type*;
	using const_reference = const reference;
	using const_pointer = const pointer;
	using size_type = sizeT;

	pointer const ptr;
	const size_type siz;

	inline slice(pointer Ptr, size_type Size) : ptr(Ptr), siz(Size) {}
	inline slice& operator= (std::initializer_list<value_type> contents) {
		if (contents.size() > siz) throw std::length_error("contents size exeeds slice allocated memory segment");
		size_type i = 0;
		for (value_type val : contents) {
			ptr[i++] = val;
		}
	}
	inline reference operator[] (sizeT i) { assert(i < siz); return ptr[i]; }
	inline const_reference operator[] (sizeT i) const { assert(i < siz); return ptr[i]; }

	trivial_iterator_defs(value_type, ptr, siz)

	inline size_type size() const { return siz; }
	inline pointer data() { return ptr; }
	inline const_pointer data() const { return ptr; }

	inline slice sub_slice(size_type Offset, size_type Size) const { return slice(ptr + Offset, Size); }
};
