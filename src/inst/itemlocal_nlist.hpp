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
		template <typename valueT, typename weightT = valueT, typename indexT = size_t, template<class ...> class Container = std::vector>
		class itemlocal_nlist {
		public:
			using value_type = valueT;
			using weight_type = weightT;
			using index_type = indexT;
			using size_type = typename Container<uint8_t>::size_type;

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
			/* storage scheme
			item indexes | limits | items
			where each item:
			value | weights | nexts
			*/
			Container<uint8_t> storage;
			size_type n;
			size_type m;

			inline static size_type get_storage_size(size_type M,
				std::initializer_list<std::initializer_list<index_type>> nexts
			) {
				size_type res = (M * sizeof(weight_type)) +
				(nexts.size() * ((M * sizeof(weight_type)) + sizeof(value_type) + sizeof(size_type)));
				for (const auto& next : nexts) {
					res += next.size() * sizeof(index_type);
				}
				return res;
			}

			inline void fill_data(
				std::initializer_list<weight_type> Limits,
				std::initializer_list<value_type> values,
				std::initializer_list<std::initializer_list<weight_type>> Weights,
				std::initializer_list<std::initializer_list<index_type>> Nexts
			) {
				assert(values.size() == Weights.size());
				assert(values.size() == Nexts.size());
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
				for (const auto& ws : Weights) {
					assert(ws.size() == Limits.size());
					weights(i++) = ws;
				}
			}

		public:

			itemlocal_nlist() = default;

			inline itemlocal_nlist(
				const Container<uint8_t>& data,
				std::initializer_list<weight_type> Limits,
				std::initializer_list<value_type> values,
				std::initializer_list<std::initializer_list<weight_type>> Weights,
				std::initializer_list<std::initializer_list<index_type>> Nexts
			) : n(values.size()), m(Limits.size()), storage(data) {
				assert(data.size() == get_storage_size(Limits.size(), Nexts));
				fill_data(Limits, values, Weights, Nexts);
			}

			inline itemlocal_nlist(
				std::initializer_list<weight_type> Limits,
				std::initializer_list<value_type> values,
				std::initializer_list<std::initializer_list<weight_type>> Weights,
				std::initializer_list<std::initializer_list<index_type>> Nexts
			) : n(values.size()), m(Limits.size()), storage(get_storage_size(Limits.size(), Nexts)) {
				fill_data(Limits, values, Weights, Nexts);
			}

			inline itemlocal_nlist(const std::string filename) : itemlocal_nlist() {
				std::ifstream fin(filename);
				if (!fin.is_open()) throw std::runtime_error("could not open file: " + filename);
				fin >> (*this);
				fin.close();
			}

			inline slice<size_type, size_type> item_data_slice() {
				return slice<size_type, size_type>((size_type*)(storage.data()), n);
			}

			inline slice<const size_type, size_type> item_data_slice() const {
				return slice<const size_type, size_type>((size_type*)(storage.data()), n);
			}

			inline weights_type limits() {
				return weights_type((weight_type*)(&storage[n * sizeof(size_type)]), m);
			}

			inline const_weights_type limits() const {
				return const_weights_type((weight_type*)(&storage[n * sizeof(size_type)]), m);
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

			inline weights_type weights(size_type i) {
				return weights_type((weight_type*)(&storage[item_data_slice()[i] + sizeof(value_type)]), m);
			}

			inline const_weights_type weights(size_type i) const {
				return const_weights_type((const weight_type*)(&storage[item_data_slice()[i] + sizeof(value_type)]), m);
			}

			inline weight_type& weight(size_type i, size_type j) {
				return weights(i)[j];
			}

			inline const weight_type& weight(size_type i, size_type j) const {
				return weights(i)[j];
			}

			inline nexts_type nexts(size_type i) {
				assert(i < n);
				auto ids = item_data_slice();
				uint8_t* ptr = &storage[ids[i]];
				uint8_t* nextPtr = (i == n - 1 ? (&storage.back()) + 1 : &storage[ids[i + 1]]);
				size_type nextsCount = (nextPtr - ptr - sizeof(value_type) - (m * sizeof(weight_type))) / sizeof(index_type);
				return nexts_type((index_type*)(&storage[item_data_slice()[i] + sizeof(value_type) + (m * sizeof(weight_type))]), nextsCount);
			}
			
			inline const_nexts_type nexts(size_type i) const {
				assert(i < n);
				auto ids = item_data_slice();
				const uint8_t* ptr = &storage[ids[i]];
				const uint8_t* nextPtr = (i == n - 1 ? (&storage.back()) + 1 : &storage[ids[i + 1]]);
				size_type nextsCount = (nextPtr - ptr - sizeof(value_type) - (m * sizeof(weight_type))) / sizeof(index_type);
				return const_nexts_type((index_type*)(&storage[item_data_slice()[i] + sizeof(value_type) + (m * sizeof(weight_type))]), nextsCount);
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

			friend inline std::istream& operator>> (std::istream& stream, itemlocal_nlist& itnl) {
				// read n and m
				stream >> itnl.n;
				stream >> itnl.m;

				// get storage parameters
				using sizt = std::vector<uint8_t>::size_type;
				sizt size = (sizeof(weight_type) * (itnl.n + 1) * itnl.m) + ((sizeof(value_type) + sizeof(size_type)) * itnl.n);
				sizt stride = (sizeof(weight_type) * itnl.m) + sizeof(value_type);
				sizt limitsOff = sizeof(size_type) * itnl.n;
				sizt itemsOff = limitsOff + sizeof(weight_type) * itnl.m;

				// read limitsOff, weights and values
				std::vector<uint8_t> weight_value(size);
				for (sizt i = limitsOff; i < itemsOff; i += sizeof(weight_type)) {
					stream >> (*((weight_type*)(&weight_value[i])));
				}
				for (sizt i = itemsOff; i < weight_value.size(); i += stride) {
					stream >> (*((value_type*)(&weight_value[i])));
				}
				for (sizt i = itemsOff; i < weight_value.size(); i += stride) {
					for (sizt j = i + sizeof(value_type); j < i + stride; j += sizeof(weight_type)) {
						stream >> (*((weight_type*)(&weight_value[j])));
					}
				}

				// read nexts lists
				std::vector<index_type> nlist;
				for (sizt i = 0; i < limitsOff; i += sizeof(size_type)) {
					size_type count;
					stream >> count;
					(*((size_type*)(&weight_value[i]))) = count;
					for (size_type j = 0; j < count; ++j) {
						index_type ind;
						stream >> ind;
						nlist.push_back(ind);
					}
				}

				// copy data
				itnl.storage.resize(weight_value.size() + nlist.size() * sizeof(index_type));
				size_type nlistInd = 0;
				size_type storageInd = itemsOff;
				sizt j = 0;
				for (sizt i = itemsOff; i < weight_value.size(); i += stride) {
					memcpy(itnl.storage.data() + storageInd, weight_value.data() + i, stride);
					size_type nextsCount = (*(size_type*)(&weight_value[j]));
					memcpy(itnl.storage.data() + storageInd + stride, nlist.data() + nlistInd, nextsCount * sizeof(size_type));
					(*(size_type*)(&weight_value[j])) = storageInd;
					storageInd += stride + nextsCount * sizeof(size_type);
					nlistInd += nextsCount;
					j += sizeof(size_type);
				}
				memcpy(itnl.storage.data(), weight_value.data(), itemsOff);

				// return
				return stream;
			}
		};
	}
}