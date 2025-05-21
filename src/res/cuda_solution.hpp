#pragma once
#include <cstdint>
#include <ostream>

namespace gs {
	namespace cuda {
		namespace res {
			class solution64 {
			public:
				uint64_t _data;
				size_t _n;

				inline solution64() : _n(0), _data(0) {}
				inline solution64(size_t N) : _n(N), _data(0) {}
				inline void add(size_t i) { _data |= (uint64_t(1) << i); }
				inline void remove(size_t i) { _data &= ~(uint64_t(1) << i); }
				inline bool has(size_t i) const { return (_data & (uint64_t(1) << i)); }
				inline size_t size() const { return _n; }
				inline size_t selected_count() const {
					size_t sum = 0;
					for (int j = 0; j < _n; j++)
						if (has(j)) ++sum;
					return sum;
				}
				inline friend std::ostream& operator<< (std::ostream& stream, const solution64& sol64) {
					for (int j = 0; j < sol64.size(); j++) {
						if (sol64.has(j)) stream << 1;
						else stream << 0;
					}
					return stream;
				}
			};
		}
	}
}
