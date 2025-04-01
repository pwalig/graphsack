#pragma once
#include <vector>
#include <cassert>
#include <tuple>
#include <iostream>
#include <numeric>

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

			using weights_type = slice<weight_type, size_type>;
			using const_weights_type = slice<const weight_type, size_type>;
			using nexts_type = slice<index_type, size_type>;
			using const_nexts_type = slice<const index_type, size_type>;

			template <typename valueT, typename weightT, typename indexT>
			class ItemView {
			public:
				using value_type = valueT;
				using weight_type = weightT;
				using index_type = indexT;

				value_type& value;
				slice<weight_type> weights;
				slice<index_type> nexts;

				inline ItemView(value_type& Value, slice<weight_type, size_type> Weights, slice<index_type, size_type> Nexts)
					: value(Value), weights(Weights), nexts(Nexts) { }

				template<typename T>
				inline ItemView(T* Ptr, size_type M, size_type Nexts)
					: ItemView(
						*((value_type*)(Ptr)),
						slice<weight_type, size_type>((weight_type*)(Ptr + sizeof(value_type)), M),
						slice<index_type, size_type>((index_type*)(Ptr + sizeof(value_type) + (M * sizeof(weight_type))), Nexts)
					) { static_assert(sizeof(T) == 1); }

				inline itemlocal_nlist::weight_type total_weight() const {
					return std::accumulate(weights.begin(), weights.end(), 0);
				}

				friend inline std::ostream& operator<< (std::ostream& stream, const ItemView& item) {
					stream << "value: " << item.value << "\nweights:";
					for (weight_type w : item.weights) { stream << " " << w; }
					stream << "\nnexts:";
					for (index_type i : item.nexts) { stream << " " << i; }
					return stream;
				}
			};

			using item_type = ItemView<value_type, weight_type, index_type>;
			using const_item_type = ItemView<const value_type, const weight_type, const index_type>;
		private:
			std::vector<uint8_t> storage;
			size_type n;
			size_type m;

			inline static std::vector<uint8_t>::size_type get_storage_size(size_type M,
				std::initializer_list<std::initializer_list<index_type>> nexts
			) {
				std::vector<uint8_t>::size_type res = (M * sizeof(weight_type)) +
				(nexts.size() * ((M * sizeof(weight_type)) + sizeof(value_type) + sizeof(size_type)));
				for (const auto& next : nexts) {
					res += next.size() * sizeof(index_type);
				}
				return res;
			}

		public:

			inline itemlocal_nlist(size_type N,
				std::initializer_list<weight_type> Limits,
				std::initializer_list<value_type> values,
				std::initializer_list<std::initializer_list<weight_type>> weights,
				std::initializer_list<std::initializer_list<index_type>> Nexts
			) : n(N), m(Limits.size()), storage(get_storage_size(Limits.size(), Nexts)) {
				size_type acc = (n * sizeof(size_type)) + (m * sizeof(weight_type));
				size_type i = 0;
				auto ids = item_data_slice();
				for (const auto& next : Nexts) {
					ids[i] = acc;
					acc += sizeof(value_type) + (m * sizeof(weight_type)) + (next.size() * sizeof(index_type));
					++i;
				}
				i = 0;
				for (weight_type w : Limits) { limit(i++) = w; }
				i = 0;
				for (const auto& next : Nexts) { nexts(i++) = next; }
				i = 0;
				for (const auto& val : values) { value(i++) = val; }
				i = 0;
				for (const auto& ws : weights) {
					size_type j = 0;
					for (weight_type w : ws) { weight(i, j++) = w; }
					++i;
				}
			}

			inline slice<size_type> item_data_slice() {
				return slice<size_type>((size_type*)(storage.data()), n);
			}

			inline slice<const size_type> item_data_slice() const {
				return slice<const size_type>((size_type*)(storage.data()), n);
			}

			inline slice<weight_type, size_type> limits() {
				return slice<weight_type, size_type>((weight_type*)(&storage[n * sizeof(size_type)]), m);
			}

			inline slice<const weight_type, size_type> limits() const {
				return slice<const weight_type, size_type>((weight_type*)(&storage[n * sizeof(size_type)]), m);
			}

			inline weight_type& limit(size_type i) {
				return limits()[i];
			}

			inline const weight_type& limit(size_type i) const {
				return limits()[i];
			}

			inline value_type& value(size_type i) {
				return *((value_type*)(&storage[item_data_slice()[i]]));
			}

			inline const value_type& value(size_type i) const {
				return *((value_type*)(&storage[item_data_slice()[i]]));
			}

			inline slice<weight_type, size_type> weights(size_type i) {
				return slice<weight_type, size_type>((weight_type*)(&storage[item_data_slice()[i] + sizeof(value_type)]), m);
			}

			inline slice<const weight_type, size_type> weights(size_type i) const {
				return slice<const weight_type, size_type>((const weight_type*)(&storage[item_data_slice()[i] + sizeof(value_type)]), m);
			}

			inline weight_type& weight(size_type i, size_type j) {
				return weights(i)[j];
			}

			inline const weight_type& weight(size_type i, size_type j) const {
				return weights(i)[j];
			}

			inline slice<index_type, size_type> nexts(size_type i) {
				assert(i < n);
				auto ids = item_data_slice();
				uint8_t* ptr = &storage[ids[i]];
				uint8_t* nextPtr = (i == n - 1 ? (&storage.back()) + 1 : &storage[ids[i + 1]]);
				size_type nextsCount = (nextPtr - ptr - sizeof(value_type) - (m * sizeof(weight_type))) / sizeof(index_type);
				return slice<index_type, size_type>((index_type*)(&storage[item_data_slice()[i] + sizeof(value_type) + (m * sizeof(weight_type))]), nextsCount);
			}
			
			inline slice<const index_type, size_type> nexts(size_type i) const {
				assert(i < n);
				auto ids = item_data_slice();
				const uint8_t* ptr = &storage[ids[i]];
				const uint8_t* nextPtr = (i == n - 1 ? (&storage.back()) + 1 : &storage[ids[i + 1]]);
				size_type nextsCount = (nextPtr - ptr - sizeof(value_type) - (m * sizeof(weight_type))) / sizeof(index_type);
				return slice<const index_type, size_type>((index_type*)(&storage[item_data_slice()[i] + sizeof(value_type) + (m * sizeof(weight_type))]), nextsCount);
			}

			inline item_type item(size_type i) {
				return item_type(value(i), weights(i), nexts(i));
			}

			inline const_item_type item(size_type i) const {
				return const_item_type(value(i), weights(i), nexts(i));
			}

			inline item_type operator[] (size_type i) { return item(i); }
			inline const_item_type operator[] (size_type i) const { return item(i); }

			inline size_type size() const { return n; }
			inline size_type dim() const { return m; }

			friend inline std::ostream& operator<< (std::ostream& stream, const itemlocal_nlist & itnl) {
				size_type i = 0;
				stream << "N: " << itnl.n << "\nlimits:";
				for (weight_type w : itnl.limits()) {
					stream << " " << w;
				}
				for (size_type i = 0; i < itnl.size(); ++i) {
					stream << "\n" << itnl.item(i);
				}
				return stream;
			}
		};
	}
}