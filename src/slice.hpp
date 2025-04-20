#pragma once
#include "iterator.hpp"

// Warning! Slice has pointer semantics.
// Meaning a const slice can point to non const data.
// Copying slice does not copy contents.
template <typename T, typename sizeT = size_t>
class slice {
public:
	using value_type = T;
	using reference = value_type&;
	using pointer = value_type*;
	using size_type = sizeT;

	pointer ptr;
	size_type siz;

	inline slice(pointer Ptr, size_type Size) : ptr(Ptr), siz(Size) {}
	inline slice& operator= (std::initializer_list<value_type> contents) {
		if (contents.size() > siz) throw std::length_error("contents size exeeds slice allocated memory segment");
		std::copy(contents.begin(), contents.end(), begin());
		return (*this);
	}
	inline reference operator[] (sizeT i) const { assert(i < siz); return ptr[i]; }

	trivial_iterator_defs(value_type, ptr, siz)

	inline size_type size() const { return siz; }
	inline pointer data() const { return ptr; }

	inline slice sub_slice(size_type Offset, size_type Size) const { return slice(ptr + Offset, Size); }
};


// cslice is a slice but with container semantics
// copying slice copies contents
// const cslice does not allow to modifiy it's contents
template <typename T, typename sizeT = size_t>
class cslice {
public:
	using value_type = T;
	using reference = value_type&;
	using pointer = value_type*;
	using const_reference = const reference;
	using const_pointer = const pointer;
	using size_type = sizeT;

	pointer ptr;
	size_type siz;

	inline cslice(pointer Ptr, size_type Size) : ptr(Ptr), siz(Size) {}
	cslice(const cslice& other) = delete;
	cslice(cslice&& other) = delete;
	inline cslice& operator= (const cslice& other) {
		if (this->siz < other.siz) throw std::length_error("other cslice size exeeds cslie allocated memory segment");
		if (this->ptr < other.ptr) {
			for (size_type i = 0; i < other.siz; ++i) this->ptr[i] = other.ptr[i];
		}
		else if (this->ptr > other.ptr) {
			for (size_type i = other.siz - 1; i >= 0; --i) this->ptr[i] = other.ptr[i];
		}
		return (*this);
	}
	inline cslice& operator= (cslice&& other) = delete;
	inline cslice& operator= (std::initializer_list<value_type> contents) {
		if (contents.size() > siz) throw std::length_error("contents size exeeds slice allocated memory segment");
		size_type i = 0;
		for (value_type val : contents) {
			ptr[i++] = val;
		}
		return (*this);
	}
	inline reference operator[] (sizeT i) { assert(i < siz); return ptr[i]; }
	inline const_reference operator[] (sizeT i) const { assert(i < siz); return ptr[i]; }

	trivial_iterator_defs(value_type, ptr, siz)

	inline size_type size() const { return siz; }
	inline pointer data() { return ptr; }
	inline const_pointer data() const { return ptr; }

	inline cslice sub_slice(size_type Offset, size_type Size) { return cslice(ptr + Offset, Size); }
	inline const cslice sub_slice(size_type Offset, size_type Size) const { return cslice(ptr + Offset, Size); }
};
