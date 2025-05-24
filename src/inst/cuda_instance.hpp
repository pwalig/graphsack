#pragma once
#include <vector>
#include <iostream>

#include "../graphs/adjacency_matrix.hpp"
#include "../structure.hpp"
#include "../weight_treatment.hpp"

namespace gs {
	namespace cuda {
		namespace inst {
			template <typename StorageBase, typename ValueT, typename WeightT>
			class instance {
			public:
				using value_type = ValueT;
				using weight_type = WeightT;
				using index_type = uint32_t;
				using size_type = size_t;

			private:
				std::vector<weight_type> _limits;
				std::vector<value_type> _values;
				std::vector<weight_type> _weights;
				std::vector<StorageBase> adjacency;
				gs::structure structureToFind;
				gs::weight_treatment weightTreatment;

			public:
				template <typename LIter, typename VIter, typename WIter>
				inline instance(
					LIter LimitsBegin, LIter LimitsEnd,
					VIter ValuesBegin, VIter ValuesEnd,
					WIter WeightsBegin, WIter WeightsEnd,
					const graphs::adjacency_matrix& graph,
					structure Structure = structure::path,
					gs::weight_treatment WeightTreatment = gs::weight_treatment::full
				) : _values(ValuesBegin, ValuesEnd), _limits(LimitsBegin, LimitsEnd), _weights(WeightsBegin, WeightsEnd),
					structureToFind(Structure), weightTreatment(WeightTreatment)
				{
					using size_type = typename graphs::adjacency_matrix::size_type;
					size_type n = graph.size();
					assert(n <= sizeof(StorageBase) * 8);
					adjacency.resize(n, 0);
					for (size_type i = 0; i < n; ++i) {
						for (size_type j = 0; j < n; ++j) {
							if (graph.at(i, j))
								adjacency[i] |= (StorageBase(1) << j);
						}
					}
				}

				inline size_type size() const { return adjacency.size(); }
				inline size_type dim() const { return _limits.size(); }

				inline weight_type& limit(size_type limitId) { return _limits[limitId]; }
				inline const weight_type& limit(size_type limitId) const { return _limits[limitId]; }

				inline std::vector<weight_type>& limits() { return _limits; }
				inline const std::vector<weight_type>& limits() const { return _limits; }

				inline value_type& value(size_type itemId) { return _values[itemId]; }
				inline const value_type& value(size_type itemId) const { return _values[itemId]; }

				inline std::vector<value_type>& values() { return _values; }
				inline const std::vector<value_type>& values() const { return _values; }

				inline weight_type& weight(size_type itemId, size_type weightId) { return _weights[itemId * dim() + weightId]; }
				inline const weight_type& weight(size_type itemId, size_type weightId) const { return _weights[itemId * dim() + weightId]; }

				inline std::vector<weight_type>& weights() { return _weights; }
				inline const std::vector<weight_type>& weights() const { return _weights; }

				inline slice<weight_type> weights(size_t itemId) { return slice<weight_type>(_weights.data() + (dim() * itemId), dim()); }
				inline slice<const weight_type> weights(size_t itemId) const { return slice<const weight_type>(_weights.data() + (dim() * itemId), dim()); }

				inline std::vector<index_type> nexts(size_t itemId) const {
					std::vector<index_type> res;
					StorageBase mask = 1;
					for (index_type i = 0; i < size(); ++i) {
						if (mask & adjacency[itemId]) res.push_back(i);
						mask <<= 1;
					}
					return res;
				}

				inline StorageBase* graph_data() { return adjacency.data(); }
				inline const StorageBase* graph_data() const { return adjacency.data(); }

				inline gs::structure& structure_to_find() { return structureToFind; }
				inline gs::weight_treatment& weight_treatment() { return weightTreatment; }
				inline const gs::structure& structure_to_find() const { return structureToFind; }
				inline const gs::weight_treatment& weight_treatment() const { return weightTreatment; }

				inline bool has_connection_to(size_type from, size_type to) const {
					return adjacency[from] & (StorageBase(1) << to);
				}

				friend inline std::ostream& operator<< (std::ostream& stream, const instance& ci) {
					stream << "limits:";
					for (auto i : ci._limits) stream << " " << i;
					stream << "\nvalues:";
					for (auto i : ci._values) stream << " " << i;
					stream << "\nweights:";
					for (auto i : ci._weights) stream << " " << i;
					stream << "\ngraph:\n";
					for (size_type i = 0; i < ci.adjacency.size(); ++i) {
						for (size_type j = 0; j < ci.adjacency.size(); ++j) {
							if (ci.adjacency[i] & (StorageBase(1) << j)) std::cout << '1';
							else std::cout << '0';
						}
						std::cout << '\n';
					}
					return stream;
				}
			};

			template<typename ValueT, typename WeightT>
			using instance8 = instance<uint8_t, ValueT, WeightT>;

			template<typename ValueT, typename WeightT>
			using instance16 = instance<uint16_t, ValueT, WeightT>;

			template<typename ValueT, typename WeightT>
			using instance32 = instance<uint32_t, ValueT, WeightT>;

			template<typename ValueT, typename WeightT>
			using instance64 = instance<uint64_t, ValueT, WeightT>;
		}
	}
}
