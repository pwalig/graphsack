#pragma once
#include <vector>
#include <iostream>

#include "../graphs/adjacency_matrix.hpp"
#include "../structure.hpp"
#include "../weight_treatment.hpp"

namespace gs {
	namespace cuda {
		namespace inst {
			template <typename ValueT, typename WeightT>
			class instance64 {
			public:
				using value_type = ValueT;
				using weight_type = WeightT;
				using index_type = uint32_t;
				using size_type = size_t;

			private:
				std::vector<weight_type> _limits;
				std::vector<value_type> _values;
				std::vector<weight_type> _weights;
				std::vector<uint64_t> adjacency;
				gs::structure structureToFind;
				gs::weight_treatment weightTreatment;

			public:
				template <typename LIter, typename VIter, typename WIter>
				inline instance64(
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
					assert(n <= 64);
					adjacency.resize(n, 0);
					for (size_type i = 0; i < n; ++i) {
						for (size_type j = 0; j < n; ++j) {
							if (graph.at(i, j))
								adjacency[i] |= (1 << j);
						}
					}
				}

				inline size_type size() const { return adjacency.size(); }
				inline size_type dim() const { return _limits.size(); }

				inline std::vector<weight_type>& limits() { return _limits; }
				inline const std::vector<weight_type>& limits() const { return _limits; }

				inline weight_type* limits_data() { return _limits.data(); }
				inline const weight_type* limits_data() const { return _limits.data(); }

				inline weight_type& limit(size_type limitId) { return _limits[limitId]; }
				inline const weight_type& limit(size_type limitId) const { return _limits[limitId]; }

				inline value_type& value(size_type itemId) { return _values[itemId]; }
				inline const value_type& value(size_type itemId) const { return _values[itemId]; }

				inline value_type* values_data() { return _values.data(); }
				inline const value_type* values_data() const { return _values.data(); }

				inline weight_type& weight(size_type itemId, size_type weightId) { return _weights[itemId * dim() + weightId]; }
				inline const weight_type& weight(size_type itemId, size_type weightId) const { return _weights[itemId * dim() + weightId]; }

				inline weight_type* weights_data() { return _weights.data(); }
				inline const weight_type* weights_data() const { return _weights.data(); }

				inline uint64_t* graph_data() { return adjacency.data(); }
				inline const uint64_t* graph_data() const { return adjacency.data(); }

				friend inline std::ostream& operator<< (std::ostream& stream, const instance64& ci) {
					stream << "limits:";
					for (auto i : ci._limits) stream << " " << i;
					stream << "\nvalues:";
					for (auto i : ci._values) stream << " " << i;
					stream << "\nweights:";
					for (auto i : ci._weights) stream << " " << i;
					stream << "\ngraph:\n";
					for (size_type i = 0; i < ci.adjacency.size(); ++i) {
						for (size_type j = 0; j < ci.adjacency.size(); ++j) {
							if (ci.adjacency[i] & (1 << j)) std::cout << '1';
							else std::cout << '0';
						}
						std::cout << '\n';
					}
					return stream;
				}
			};

			using instance_t = instance64<uint32_t, uint32_t>;

			void copy_to_symbol(const instance64<uint32_t, uint32_t>& inst);
		}
	}
}
