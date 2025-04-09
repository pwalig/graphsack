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
		template <typename T, typename... Args>
		using op = T(*)(Args...);

		template <typename T, typename U>
		using in_place_op = void(*)(T&, const U&);

		template <typename V, typename T, typename U>
		using ret_op = op<V, T, U>;

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

		template <typename V, typename T, typename U>
		inline V add(T lhs, U rhs) {
			return lhs + rhs;
		}

		template <typename V, typename T, typename U>
		inline V sub(T lhs, U rhs) {
			return lhs - rhs;
		}
		
		template <typename V, typename T, typename U>
		inline V div(T lhs, U rhs) {
			return lhs / rhs;
		}

		template <typename T, typename U>
		inline void in_place_operate(
			T& lhs, const U& rhs, in_place_op<typename T::value_type, typename U::value_type> op
		) {
			assert(lhs.size() == rhs.size());
			for (size_t i = 0; i < lhs.size(); ++i) {
				op(lhs[i], rhs[i]);
			}
		}

		template <typename T, typename... Args>
		inline T operate_variadic(
			Args... args,
			op<typename T::value_type, typename Args::value_type...> op
		) {
			T res(args.size());
			for (size_t i = 0; i < args.size(); ++i) {
				res[i] = op(args[i]);
			}
			return res;
		}

		template <typename V, typename T, typename U>
		inline V operate(
			const T& lhs, const U& rhs, ret_op<typename V::value_type, typename T::value_type, typename U::value_type> op
		) {
			assert(lhs.size() == rhs.size());
			return operate_variadic<V, const T&, const U&>(lhs, rhs, op);
			//V res(lhs.size());
			//for (size_t i = 0; i < lhs.size(); ++i) {
			//	res[i] = op(lhs[i], rhs[i]);
			//}
			//return res;
		}
	}

	template <typename T, typename U>
	inline void add_to_weights(T& weights, const U& item_weights) {
		element_wise::in_place_operate(
			weights, item_weights,
			element_wise::in_place_add
		);
	}

	template <typename T, typename U>
	inline void sub_from_weights(T& weights, const U& item_weights) {
		element_wise::in_place_operate(
			weights, item_weights,
			element_wise::in_place_sub
		);
	}

	template <typename V, typename T, typename U>
	inline V add_weights(const T& lhs, const U& rhs) {
		return element_wise::operate<V, T, U>(
			lhs, rhs,
			element_wise::add<typename V::value_type>
		);
	}

	template <typename V, typename T, typename U>
	inline V sub_weights(const T& lhs, const U& rhs) {
		return element_wise::operate<V, T, U>(
			lhs, rhs,
			element_wise::sub<typename V::value_type>
		);
	}
}
