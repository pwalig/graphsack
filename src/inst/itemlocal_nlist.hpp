#pragma once
#include <vector>
#include <cassert>
#include <tuple>
#include <iostream>
#include <numeric>

#include "../graphs/representation.hpp"
#include "../slice.hpp"
#include "../iterator.hpp"
#include "../structure.hpp"
#include "../weight_treatment.hpp"
#include "../graphs/adjacency_matrix.hpp"
#include "inst_macros.hpp"

namespace gs {
	namespace inst {
		// instance with nexts list graph representation
		// aiming to store all of heap allocated data in one contigous memory segment
		template <typename valueT, typename weightT = valueT, typename indexT = size_t, class Container = std::vector<uint8_t>>
		class itemlocal_nlist {
		public:
			using value_type = valueT;
			using weight_type = weightT;
			using index_type = indexT;
			using size_type = typename Container::size_type;
			inline const static graphs::representation representation = graphs::representation::nexts_list;

			static_assert(sizeof(value_type) % sizeof(typename Container::value_type) == 0);
			static_assert(sizeof(weight_type) % sizeof(typename Container::value_type) == 0);
			static_assert(sizeof(index_type) % sizeof(typename Container::value_type) == 0);
			static_assert(sizeof(size_type) % sizeof(typename Container::value_type) == 0);

			inline const static size_type sizeofValue = sizeof(value_type) / sizeof(typename Container::value_type);
			inline const static size_type sizeofWeight = sizeof(weight_type) / sizeof(typename Container::value_type);
			inline const static size_type sizeofIndex = sizeof(index_type) / sizeof(typename Container::value_type);
			inline const static size_type sizeofSize = sizeof(size_type) / sizeof(typename Container::value_type);

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
						slice<weight_type, size_type>((weight_type*)(Ptr + sizeofValue), M),
						slice<index_type, size_type>((index_type*)(Ptr + sizeofValue + (M * sizeofWeight)), Nexts)
					) { }

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
			item in memory indexes | limits | items
			where each item:
			value | weights | nexts
			*/
			Container storage;
			size_type n;
			size_type m;
			GS_INST_WEIGHT_TREATMENT_AND_STRUCTURE_MEMBERS

			template<typename Iterable>
			inline static size_type get_storage_size(
				size_type M, Iterable Nexts
			) {
				size_type res = (M * sizeofWeight) +
				(Nexts.size() * ((M * sizeofWeight) + sizeofValue + sizeofSize));
				for (const auto& next : Nexts) {
					res += next.size() * sizeofIndex;
				}
				return res;
			}

			inline static size_type get_storage_size(
				size_type M, const graphs::adjacency_matrix& graph
			) {
				using g_size_t = typename graphs::adjacency_matrix::size_type;

				size_type n = graph.size();
				size_type res = (M * sizeofWeight) +
				(n * ((M * sizeofWeight) + sizeofValue + sizeofSize));
				for (g_size_t i = 0; i < n; ++i) {
					size_type nextsCount = std::count_if(graph[i].begin(), graph[i].end(), [](bool set) { return set; });
					res += nextsCount * sizeofIndex;
				}
				return res;
			}

			template <typename Iter>
			inline void fill_data_slice(
				Iter NextsBegin, Iter NextsEnd
			) {
				size_type acc = (n * sizeofSize) + (m * sizeofWeight);
				size_type i = 0;
				auto ids = item_data_slice();
				for (Iter it = NextsBegin; it != NextsEnd; ++it) {
					ids[i] = acc;
					acc += sizeofValue + (m * sizeofWeight) + ((*it).size() * sizeofIndex);
					++i;
				}
			}

			inline void fill_data_slice(
				const graphs::adjacency_matrix& graph
			) {
				assert(graph.size() == n);
				using g_size_t = typename graphs::adjacency_matrix::size_type;

				size_type acc = (n * sizeofSize) + (m * sizeofWeight);
				auto ids = item_data_slice();
				for (g_size_t i = 0; i < n; ++i) {
					ids[static_cast<size_type>(i)] = acc;
					size_type nextsCount = std::count_if(graph[i].begin(), graph[i].end(), [](bool set) { return set; });
					acc += sizeofValue + (m * sizeofWeight) + (nextsCount * sizeofIndex);
				}
			}

			template <typename Iter>
			inline void fill_nexts(
				Iter NextsBegin, Iter NextsEnd
			) {
				size_type i = 0;
				for (auto it = NextsBegin; it != NextsEnd; ++it) 
					std::copy((*it).begin(), (*it).end(), nexts(i++).begin());
			}

			inline void fill_nexts(
				const graphs::adjacency_matrix& graph
			) {
				assert(graph.size() == n);
				using g_size_t = typename graphs::adjacency_matrix::size_type;

				for (g_size_t i = 0; i < n; ++i) {
					size_type k = 0;
					for (g_size_t j = 0; j < n; ++j) {
						if (graph.at(i, j)) nexts(i)[k++] = static_cast<index_type>(j);
					}
				}
			}

			template <typename Iter>
			inline void fill_limits( Iter Begin, Iter End )
			{ std::copy(Begin, End, limits().begin()); }

			template <typename Iter>
			inline void fill_values(Iter Begin, Iter End) {
				size_type i = 0;
				for (auto it = Begin; it != End; ++it) value(i++) = *it;
			}

			template <typename Iter>
			inline void fill_weights(Iter Begin, Iter End) {
				assert(std::distance(Begin, End) == n);
				size_type i = 0;
				for (auto it = Begin; it != End; ++it) {
					assert(std::distance((*it).begin(), (*it).end()) == m);
					std::copy((*it).begin(), (*it).end(), weights(i++).begin());
				}
			}

			template <typename Iter>
			inline void fill_contigous_weights(Iter Begin, Iter End) {
				assert(std::distance(Begin, End) == n * m);
				auto it = Begin;
				for (size_type i = 0; i < n; ++i) {
					std::copy(it, it + m, weights(i).begin());
					it += m;
				}
			}

		public:

			itemlocal_nlist(
				structure Structure = structure::path,
				weight_treatment WeightTreatment = weight_treatment::full
			) : n(0), m(0), structureToFind(Structure), weightTreatment(WeightTreatment) {}

			template <typename LIter, typename VIter, typename WIter>
			inline itemlocal_nlist(
				LIter LimitsBegin, LIter LimitsEnd,
				VIter ValuesBegin, VIter ValuesEnd,
				WIter WeightsBegin, WIter WeightsEnd,
				const graphs::adjacency_matrix& graph,
				structure Structure = structure::path,
				gs::weight_treatment WeightTreatment = gs::weight_treatment::full
			) : n(std::distance(ValuesBegin, ValuesEnd)), m(std::distance(LimitsBegin, LimitsEnd)),
				storage(get_storage_size(std::distance(LimitsBegin, LimitsEnd), graph)),
				structureToFind(Structure), weightTreatment(WeightTreatment)
			{
				static_assert(std::is_same<typename LIter::value_type, weight_type>::value);
				static_assert(std::is_same<typename VIter::value_type, value_type>::value);
				fill_data_slice(graph);
				fill_nexts(graph);
				fill_limits(LimitsBegin, LimitsEnd);
				fill_values(ValuesBegin, ValuesEnd);
				if constexpr (std::is_same<typename WIter::value_type, weight_type>::value) {
					fill_contigous_weights(WeightsBegin, WeightsEnd);
				}
				else {
					static_assert(std::is_same<typename WIter::value_type::value_type, weight_type>::value);
					fill_weights(WeightsBegin, WeightsEnd);
				}
			}

			inline itemlocal_nlist(
				const Container& data,
				std::initializer_list<weight_type> Limits,
				std::initializer_list<value_type> Values,
				std::initializer_list<std::initializer_list<weight_type>> Weights,
				std::initializer_list<std::initializer_list<index_type>> Nexts,
				structure Structure = structure::path,
				gs::weight_treatment WeightTreatment = gs::weight_treatment::full
			) : n(Values.size()), m(Limits.size()), storage(data),
				structureToFind(Structure), weightTreatment(WeightTreatment)
			{
				assert(data.size() == get_storage_size(Limits.size(), Nexts));
				fill_data_slice(Nexts.begin(), Nexts.end());
				fill_nexts(Nexts.begin(), Nexts.end());
				fill_limits(Limits.begin(), Limits.end());
				fill_values(Values.begin(), Values.end());
				fill_weights(Weights.begin(), Weights.end());
			}

			inline itemlocal_nlist(
				std::initializer_list<weight_type> Limits,
				std::initializer_list<value_type> Values,
				std::initializer_list<std::initializer_list<weight_type>> Weights,
				std::initializer_list<std::initializer_list<index_type>> Nexts,
				structure Structure = structure::path,
				gs::weight_treatment WeightTreatment = gs::weight_treatment::full
			) : n(Values.size()), m(Limits.size()), storage(get_storage_size(Limits.size(), Nexts)),
				structureToFind(Structure), weightTreatment(WeightTreatment)
			{
				fill_data_slice(Nexts.begin(), Nexts.end());
				fill_nexts(Nexts.begin(), Nexts.end());
				fill_limits(Limits.begin(), Limits.end());
				fill_values(Values.begin(), Values.end());
				fill_weights(Weights.begin(), Weights.end());
			}

			inline itemlocal_nlist(
				std::initializer_list<weight_type> Limits,
				std::initializer_list<value_type> Values,
				std::initializer_list<std::initializer_list<weight_type>> Weights,
				const graphs::adjacency_matrix& graph,
				structure Structure = structure::path,
				gs::weight_treatment WeightTreatment = gs::weight_treatment::full
			) : n(Values.size()), m(Limits.size()), storage(get_storage_size(Limits.size(), graph)),
				structureToFind(Structure), weightTreatment(WeightTreatment)
			{
				fill_data_slice(graph);
				fill_nexts(graph);
				fill_limits(Limits.begin(), Limits.end());
				fill_values(Values.begin(), Values.end());
				fill_weights(Weights.begin(), Weights.end());
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
				return weights_type((weight_type*)(&storage[n * sizeofSize]), m);
			}

			inline const_weights_type limits() const {
				return const_weights_type((weight_type*)(&storage[n * sizeofSize]), m);
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
				return weights_type((weight_type*)(&storage[item_data_slice()[i] + sizeofValue]), m);
			}

			inline const_weights_type weights(size_type i) const {
				return const_weights_type((const weight_type*)(&storage[item_data_slice()[i] + sizeofValue]), m);
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
				typename Container::value_type* ptr = &storage[ids[i]];
				typename Container::value_type* nextPtr = (i == n - 1 ? (&storage.back()) + 1 : &storage[ids[i + 1]]);
				size_type nextsCount = (nextPtr - ptr - sizeofValue - (m * sizeofWeight)) / sizeofIndex;
				return nexts_type((index_type*)(storage.data() + ids[i] + sizeofValue + (m * sizeofWeight)), nextsCount);
			}
			
			inline const_nexts_type nexts(size_type i) const {
				assert(i < n);
				auto ids = item_data_slice();
				const typename Container::value_type* ptr = &storage[ids[i]];
				const typename Container::value_type* nextPtr = (i == n - 1 ? (&storage.back()) + 1 : &storage[ids[i + 1]]);
				size_type nextsCount = (nextPtr - ptr - sizeofValue - (m * sizeofWeight)) / sizeofIndex;
				return const_nexts_type((index_type*)(storage.data() + ids[i] + sizeofValue + (m * sizeofWeight)), nextsCount);
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

			GS_INST_WEIGHT_TREATMENT_AND_STRUCTURE_ACCESS_ALL

			inline bool has_connection_to(size_type from, size_type to) const {
				return std::find(nexts(from).begin(), nexts(from).end(), to) != nexts(from).end();
			}

			template <typename Rand, typename ValueGen, typename WeightGen, typename LimitGen>
			inline itemlocal_nlist random(size_type N, size_type M, double P,
				ValueGen vg, WeightGen wg, LimitGen lg, Rand& gen,
				gs::structure StructureToFind = gs::structure::cycle,
				gs::weight_treatment WeightTreatment = gs::weight_treatment::full
			) {
				std::vector<value_type> randomValues(N);
				std::vector<weight_type> randomWeights(N * M + M);
				random::into<value_type>(randomValues.begin(), randomValues.end(), vg);
				random::into<weight_type>(randomWeights.begin() + M, randomWeights.end(), wg);
				random::into<weight_type>(randomWeights.begin(), randomWeights.begin() + M, lg);

				return inst::itemlocal_nlist(
					randomWeights.begin(), randomWeights.begin() + M,
					randomValues.begin(), randomValues.end(),
					randomWeights.begin() + M, randomWeights.end(),
					graphs::adjacency_matrix::from_gnp(N, P, gen),
					StructureToFind, WeightTreatment
				);
			}

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
				using sizt = typename Container::size_type;
				sizt size = (sizeofWeight * (itnl.n + 1) * itnl.m) + ((sizeofValue + sizeofSize) * itnl.n);
				sizt stride = (sizeofWeight * itnl.m) + sizeofValue;
				sizt limitsOff = sizeofSize * itnl.n;
				sizt itemsOff = limitsOff + sizeofWeight * itnl.m;

				// read limitsOff, weights and values
				std::vector<typename Container::value_type> weight_value(size);
				for (sizt i = limitsOff; i < itemsOff; i += sizeofWeight) {
					stream >> (*((weight_type*)(&weight_value[i])));
				}
				for (sizt i = itemsOff; i < weight_value.size(); i += stride) {
					stream >> (*((value_type*)(&weight_value[i])));
				}
				for (sizt i = itemsOff; i < weight_value.size(); i += stride) {
					for (sizt j = i + sizeofValue; j < i + stride; j += sizeofWeight) {
						stream >> (*((weight_type*)(&weight_value[j])));
					}
				}

				// read nexts lists
				std::vector<index_type> nlist;
				for (sizt i = 0; i < limitsOff; i += sizeofSize) {
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
				itnl.storage.resize(weight_value.size() + nlist.size() * sizeofIndex);
				size_type nlistInd = 0;
				size_type storageInd = itemsOff;
				sizt j = 0;
				for (sizt i = itemsOff; i < weight_value.size(); i += stride) {
					memcpy(itnl.storage.data() + storageInd, weight_value.data() + i, stride);
					size_type nextsCount = (*(size_type*)(&weight_value[j]));
					memcpy(itnl.storage.data() + storageInd + stride, nlist.data() + nlistInd, nextsCount * sizeofSize);
					(*(size_type*)(&weight_value[j])) = storageInd;
					storageInd += stride + nextsCount * sizeofSize;
					nlistInd += nextsCount;
					j += sizeofSize;
				}
				memcpy(itnl.storage.data(), weight_value.data(), itemsOff);

				// return
				return stream;
			}
		};
	}
}