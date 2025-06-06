#pragma once
#include <vector>

#include "../graphs/adjacency_matrix.hpp"
#include "inst_macros.hpp"
#include "../graphs/representation.hpp"

namespace gs::inst {
	template <typename ValueT, typename WeightT>
	class naive_item_vector {
	public:
		using value_type = ValueT;
		using weight_type = WeightT;
		using index_type = size_t;
		using size_type = size_t;
		inline const static graphs::representation representation = graphs::representation::nexts_list;

		using limits_type = std::vector<weight_type>;
		using const_limits_type = const std::vector<weight_type>;
		using weights_type = std::vector<weight_type>;
		using const_weights_type = const std::vector<weight_type>;
		using nexts_type = std::vector<index_type>;
		using const_nexts_type = const std::vector<index_type>;

		class Item {
		public:
			value_type value;
			std::vector<weight_type> weights;
			std::vector<index_type> nexts;
		};

		using item_type = Item;
		using const_item_type = const Item;

	private:
		std::vector<weight_type> _limits;
		std::vector<item_type> _items;
		gs::structure structureToFind;
		gs::weight_treatment weightTreatment;

	public:
		template <typename LIter, typename VIter, typename WIter>
		inline naive_item_vector(
			LIter LimitsBegin, LIter LimitsEnd,
			VIter ValuesBegin, VIter ValuesEnd,
			WIter WeightsBegin, WIter WeightsEnd,
			const graphs::adjacency_matrix& graph,
			structure Structure = structure::path,
			gs::weight_treatment WeightTreatment = gs::weight_treatment::full
		) : _limits(LimitsBegin, LimitsEnd), structureToFind(Structure), weightTreatment(WeightTreatment) {
			_items.reserve(std::distance(ValuesBegin, ValuesEnd));
			for (size_type i = 0; i < _items.capacity(); ++i) {
				item_type to_add;
				to_add.value = *(ValuesBegin++);
				to_add.weights.reserve(_limits.size());
				for (size_type wid = 0; wid < _limits.size(); ++wid) to_add.weights.push_back(*(WeightsBegin++));
				for (index_type n = 0; n < _items.capacity(); ++n) if (graph.at(i, n)) to_add.nexts.push_back(n);
				_items.push_back(to_add);
			}
		}
		inline size_type size() const { return _items.size(); } // returns size of the instace
		inline size_type dim() const { return _limits.size(); }// returns dimention of weight vector

		// return limitId'th element of knapsack capacity vector
		inline weight_type& limit(size_type limitId) { return _limits[limitId]; }
		inline const weight_type& limit(size_type limitId) const { return _limits[limitId]; }

		// return knapsack capacity vector (preferably a view into it)
		inline limits_type& limits() { return _limits; }
		inline const_limits_type& limits() const { return _limits; }

		// if weight treatment is first_only then limit index becomes redundant and this method can be used
		inline weight_type& limit() {
			assert(weightTreatment == gs::weight_treatment::first_only);
			return limits()[0];
		}
		inline const weight_type& limit() const {
			assert(weightTreatment == gs::weight_treatment::first_only);
			return limits()[0];
		}

		// return value of itemId'th element
		inline value_type& value(size_type itemId) { return _items[itemId].value; }
		inline const value_type& value(size_type itemId) const { return _items[itemId].value; }

		// return weightId'th component of weight vector of itemId'th element
		inline weight_type& weight(size_type itemId, size_type weightId) { return _items[itemId].weights[weightId]; }
		inline const weight_type& weight(size_type itemId, size_type weightId) const { return _items[itemId].weights[weightId]; }

		// return weight vector of itemId'th element
		inline weights_type& weights(size_t itemId) { return _items[itemId].weights; }
		inline const_weights_type& weights(size_t itemId) const { return _items[itemId].weights; }

		// return list of elements that itemId has arches to
		inline nexts_type& nexts(size_t itemId) { return _items[itemId].nexts; }
		inline const_nexts_type& nexts(size_t itemId) const { return _items[itemId].nexts; }

		inline item_type& item(size_type itemId) { return _items[itemId]; }
		inline const_item_type& item(size_type itemId) const { return _items[itemId]; }
		inline std::vector<item_type>& items() { return _items; }
		inline const std::vector<item_type>& items() const { return _items; }

		inline gs::structure& structure_to_find() { return structureToFind; }
		inline gs::weight_treatment& weight_treatment() { return weightTreatment; }
		inline const gs::structure& structure_to_find() const { return structureToFind; }
		inline const gs::weight_treatment& weight_treatment() const { return weightTreatment; }

		// returns true if there is an arch from from to to, false otherwise
		inline bool has_connection_to(size_type from, size_type to) const {
			return (std::find(_items[from].nexts.begin(), _items[from].nexts.end(), to) != _items[from].nexts.end());
		}

		inline friend std::ostream& operator<< (std::ostream& stream, const naive_item_vector& niv) {
			stream << "limits:";
			for (const weight_type& limit : niv.limits()) stream << ' ' << limit;
			stream << "\n_items:\n";
			for (const item_type& item : niv._items) {
				stream << "value: " << item.value << "\nweights:";
				for (const weight_type& weight : item.weights) stream << ' ' << weight;
				stream << "\nnexts:";
				for (const index_type& next : item.nexts) stream << ' ' << next;
				stream << '\n';
			}
			return stream;
		}
	};
}
