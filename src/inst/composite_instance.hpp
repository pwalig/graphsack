#pragma once
#include <cassert>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <utility>
#include <fstream>

#include "../graphs/representation.hpp"
#include "../graphs/adjacency_matrix.hpp"
#include "inst_macros.hpp"

namespace gs {
	namespace inst {
		template <typename WeightValueT, typename graphT>
		class composite {
		public:
			using weight_value_t = WeightValueT;
			using graph_t = graphT;

			using value_type = typename weight_value_t::value_type;
			using weight_type = typename weight_value_t::weight_type;
			using weights_type = typename weight_value_t::weights_type;
			using const_weights_type = typename weight_value_t::const_weights_type;
			using size_type = typename weight_value_t::size_type;
			using index_type = size_type;
			inline const static graphs::representation representation = graph_t::representation;

		public:
			weight_value_t wv;
			graph_t g;
			GS_INST_STRUCTURE_MEMBER

			inline composite() {}

			inline composite(const std::string filename) : composite() {
				std::ifstream fin(filename);
				if (!fin.is_open()) throw std::runtime_error("could not open file " + filename);
				fin >> (*this);
				fin.close();
			}
			
			template <typename LIter, typename VIter, typename WIter>
			inline composite(
				LIter LimitsBegin, LIter LimitsEnd,
				VIter ValuesBegin, VIter ValuesEnd,
				WIter WeightsBegin, WIter WeightsEnd,
				const graphs::adjacency_matrix& graph,
				structure Structure = structure::path,
				gs::weight_treatment WeightTreatment = gs::weight_treatment::full
			) : wv(LimitsBegin, LimitsEnd, ValuesBegin, ValuesEnd, WeightsBegin, WeightsEnd, WeightTreatment),
				g(graph), structureToFind(Structure) { }

			inline size_type size() const { return wv.size(); } // returns size of the instace
			inline size_type dim() const { return wv.dim(); }// returns dimention of weight vector

			// return limitId'th element of knapsack capacity vector
			inline weight_type& limit(size_type limitId) { return wv.limit(limitId); }
			inline const weight_type& limit(size_type limitId) const { return wv.limit(limitId); }

			// return knapsack capacity vector (preferably a view into it)
			inline weights_type& limits() { return wv.limits(); }
			inline const_weights_type& limits() const { return wv.limits(); }

			// return value of itemId'th element
			inline value_type& value(size_type itemId) { return wv.value(itemId); }
			inline const value_type& value(size_type itemId) const { return wv.value(itemId); }

			// return weightId'th component of weight vector of itemId'th element
			inline weight_type& weight(size_type itemId, size_type weightId) { return wv.weight(itemId, weightId); }
			inline const weight_type& weight(size_type itemId, size_type weightId) const { return wv.weight(itemId, weightId); }

			// return weight vector of itemId'th element
			inline weights_type& weights(size_t itemId) { return wv.weights(itemId); }
			inline const_weights_type& weights(size_t itemId) const { return wv.weights(itemId); }

			inline gs::weight_treatment& weight_treatment() {
				return wv.weight_treatment();
			}
			inline const gs::weight_treatment& weight_treatment() const {
				return wv.weight_treatment();
			}
			GS_INST_STRUCTURE_ACCESS
			inline weight_type& limit() {
				return wv.limit();
			}
			inline const weight_type& limit() const {
				return wv.limit();
			}
			inline weight_type& weight(size_type itemId) {
				return wv.weight(itemId);
			}
			inline const weight_type& weight(size_type itemId) const {
				return wv.weight(itemId);
			}

			// returns true if there is an arch from from to to, false otherwise
			inline bool has_connection_to(size_type from, size_type to) const {
				return g.has_connection_to(from, to);
			}
			
			friend inline std::ostream& operator<< (std::ostream& stream, const composite& inst) {
				return stream << inst.wv << inst.g;
			}

			friend inline std::istream& operator>> (std::istream& stream, composite& inst) {
				return inst.wv >> inst.g;
			}
		};
	}
}
