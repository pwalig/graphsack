#pragma once
#include <vector>
#include <cassert>

#include "../slice.hpp"
#include "../iterator.hpp"

namespace gs {
	namespace inst {
		template <typename valueT, typename weightT = valueT, typename indexT = size_t>
		class itemlocal_nlist {
		public:
			using value_type = valueT;
			using weight_type = weightT;
			using index_type = indexT;
			using size_type = size_t;

			class item_view {
			public:
				value_type& value;
				slice<weight_type> weights;
				slice<index_type> nexts;

				inline item_view(value_type& Value, slice<weight_type> Weights, slice<index_type> Nexts)
					: value(Value), weights(Weights), nexts(Nexts) { }
			};
		private:
			std::vector<uint8_t> storage;
			size_type n;
			size_type m;
		public:

			inline slice<size_type> item_data() {
				return slice<size_type>(static_cast<size_type*>(storage.data()), n);
			}

			inline value_type& value(size_type i) {
				value_type* ptr = static_cast<value_type*>(&storage[item_data()[i]])
			}
			inline item_view operator[] (size_type i) {
				assert(i < n); return item_view(*(static_cast<value_type*>(&(storage)
			}
		};
	}
}