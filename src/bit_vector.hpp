#pragma once
#include <vector>
#include <iostream>

namespace gs {
	class bit_vector {
	private:
		std::vector<bool> _data;

	public:
		bit_vector(size_t n) : _data(n, false) {}
		inline void add(size_t i) { _data[i] = true; }

		inline friend std::ostream& operator<< (std::ostream& stream, const bit_vector& bv) {
			for (bool b : bv._data) {
				if (b) stream << 1;
				else stream << 0;
			}
			return stream;
		}
	};
}
