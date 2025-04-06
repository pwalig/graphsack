#pragma once
#include <cassert>
#include <functional>

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
	inline bool fits(const T& item_weights, const U& limits) {
		assert(limits.size() == item_weights.size());
		for (size_t i = 0; i < limits.size(); ++i) {
			if (item_weights[i] > limits[i]) return false;
		}
		return true;
	}

	namespace element_wise {
		template <typename T, typename U>
		using in_place_op = void(*)(T&, const U&);

		template <typename V, typename T, typename U>
		using ret_op = V(*)(T, U);

		template <typename T, typename U>
		inline void in_place_add(T& lhs, const U& rhs) {
			lhs += rhs;
		}

		template <typename T, typename U>
		inline void in_place_sub(T& lhs, const U& rhs) {
			lhs -= rhs;
		}

		template <typename T, typename U>
		inline void copy(T& lhs, const U& rhs) {
			lhs = rhs;
		}

		template <typename T, typename U>
		inline void operate(
			T& lhs, const U& rhs, in_place_op<typename T::value_type, typename U::value_type> op
		) {
			assert(lhs.size() == rhs.size());
			for (size_t i = 0; i < lhs.size(); ++i) {
				op(lhs[i], rhs[i]);
			}
		}
	}

	template <typename T, typename U>
	inline void add_to_weights(T& weights, const U& item_weights) {
		element_wise::operate(
			weights, item_weights,
			element_wise::in_place_add
		);
	}

	template <typename T, typename U>
	inline void sub_from_weights(T& weights, const U& item_weights) {
		element_wise::operate(
			weights, item_weights,
			element_wise::in_place_sub
		);
	}
}
