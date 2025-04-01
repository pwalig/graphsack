#pragma once
#include <vector>
#include <iostream>

namespace gs {
	class bit_vector {
	private:
		std::vector<bool> _data;

	public:
		inline bit_vector(size_t n) : _data(n, false) {}
		inline void add(size_t i) { _data[i] = true; }
		inline void remove(size_t i) { _data[i] = false; }
		inline bool has(size_t i) const { return _data[i]; }
		inline size_t size() const { return _data.size(); }

		inline typename std::vector<bool>::reference operator[] (size_t i) { return _data[i]; }
		inline typename std::vector<bool>::const_reference operator[] (size_t i) const { return _data[i]; }

		inline friend std::ostream& operator<< (std::ostream& stream, const bit_vector& bv) {
			for (bool b : bv._data) {
				if (b) stream << 1;
				else stream << 0;
			}
			return stream;
		}
	};
}
