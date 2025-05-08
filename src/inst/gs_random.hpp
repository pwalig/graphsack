#pragma once
#include <functional>

namespace gs {
	namespace random {
		template <typename T, typename Iter>
		inline void into(
			Iter begin, Iter end,
			T(*randomFunction)()
		) {
			for (Iter it = begin; it != end; ++it) (*it) = randomFunction();
		}
		template <typename T, typename Iter>
		inline void into(
			Iter begin, Iter end,
			std::function<T()> randomFunction
		) {
			for (Iter it = begin; it != end; ++it) (*it) = randomFunction();
		}
	}
}
