#pragma once
#include <cassert>

namespace gs {
	template <typename T, typename U, typename V>
	inline bool would_fit(const T& weights, const U& item_weights, const V& limits) {
		assert((weights.size() == limits.size()) && (limits.size() == item_weights.size()));
		for (size_t i = 0; i < limits.size(); ++i) {
			if (weights[i] + item_weights[i] > limits[i]) return false;
		}
		return true;
	}

	template <typename T, typename U>
	inline void add_weights(T& weights, const U& item_weights) {
		assert(weights.size() == item_weights.size());
		for (size_t i = 0; i < weights.size(); ++i) {
			weights[i] += item_weights[i];
		}
	}
}
