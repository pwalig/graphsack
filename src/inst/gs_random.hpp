#pragma once
#include <functional>
#include <random>

namespace gs {
	namespace random {
		template<typename Rand>
		struct function {
			template <typename T>
			static T from_uniform_distribution(Rand& gen, T min, T max) {
				std::uniform_int_distribution<T> d(min, max);
				return d(gen);
			}

			template <>
			static float from_uniform_distribution(Rand& gen, float min, float max) {
				std::uniform_real_distribution<float> d(min, max);
				return d(gen);
			}
			template <>
			static double from_uniform_distribution(Rand& gen, double min, double max) {
				std::uniform_real_distribution<double> d(min, max);
				return d(gen);
			}
			template <>
			static long double from_uniform_distribution(Rand& gen, long double min, long double max) {
				std::uniform_real_distribution<long double> d(min, max);
				return d(gen);
			}

			template <typename T>
			static T from_normal_distribution(Rand& gen, T mean, T stddiv) {
				std::normal_distribution<T> d(mean, stddiv);
				return d(gen);
			}
		};

		template <typename Iter, typename RandomCallable>
		inline void into(
			Iter begin, Iter end,
			RandomCallable randomFunction
		) {
			for (Iter it = begin; it != end; ++it) (*it) = randomFunction();
		}
	}
}
