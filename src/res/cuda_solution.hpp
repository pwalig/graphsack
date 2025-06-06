#pragma once
#include <cstdint>
#include <ostream>

namespace gs {
	namespace cuda {
		namespace res {
			template <typename StorageBase>
			class solution {
			public:
				StorageBase _data;
				StorageBase _n;

				inline solution() : _n(0), _data(0) {}
				inline solution(size_t N) : _n(static_cast<StorageBase>(N)), _data(0) {}
				inline void add(size_t i) { _data |= (StorageBase(1) << i); }
				inline void remove(size_t i) { _data &= ~(StorageBase(1) << i); }
				inline bool has(size_t i) const { return (_data & (StorageBase(1) << i)); }
				inline void clear() { _data = StorageBase(0); }
				inline size_t size() const { return _n; }
				inline size_t selected_count() const {
					size_t sum = 0;
					for (StorageBase j = 0; j < _n; j++)
						if (has(j)) ++sum;
					return sum;
				}
				inline friend std::ostream& operator<< (std::ostream& stream, const solution& sol) {
					for (int j = 0; j < sol.size(); j++) {
						if (sol.has(j)) stream << '1';
						else stream << '0';
					}
					return stream;
				}
			};

			using solution8 = solution<uint8_t>;
			using solution16 = solution<uint16_t>;
			using solution32 = solution<uint32_t>;
			using solution64 = solution<uint64_t>;
		}
	}
}
