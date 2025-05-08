#pragma once
#include <utility>
#include <functional>

#include "../graphs/adjacency_matrix.hpp"
#include "gs_random.hpp"
#include "../structure.hpp"
#include "../weight_treatment.hpp"

namespace gs::inst {
	template <typename InstanceT>
	class Generator {
	public:
		using instance_t = InstanceT;
		using value_type = typename instance_t::value_type;
		using weight_type = typename instance_t::weight_type;
		using size_type = typename instance_t::size_type;

		template <typename Rand>
		static instance_t random(
			size_type N, size_type M, double P,
			Rand& gen,
			weight_type MinLimit, weight_type MaxLimit,
			value_type MinValue, value_type MaxValue,
			weight_type MinWeight, weight_type MaxWeight,
			bool unidirectional = false, bool selfArches = true,
			structure Structure = structure::cycle, weight_treatment WeightTreatment = weight_treatment::full
		) {
			auto ld = [&gen, MinLimit, MaxLimit]() { return gs::random::function<Rand>::from_uniform_distribution(gen, MinLimit, MaxLimit); };
			auto vd = [&gen, MinValue, MaxValue]() { return gs::random::function<Rand>::from_uniform_distribution(gen, MinValue, MaxValue); };
			auto wd = [&gen, MinWeight, MaxWeight]() { return gs::random::function<Rand>::from_uniform_distribution(gen, MinWeight, MaxWeight); };
			
			std::vector<weight_type> randomWeights(N * M + M);
			std::vector<weight_type> randomValues(N);

			random::into(randomWeights.begin(), randomWeights.begin() + M, ld);
			random::into(randomValues.begin(), randomValues.end(), vd);
			random::into(randomWeights.begin() + M, randomWeights.end(), wd);

			return instance_t(
				randomWeights.begin(), randomWeights.begin() + M,
				randomValues.begin(), randomValues.end(),
				randomWeights.begin() + M, randomWeights.end(),
				gs::graphs::adjacency_matrix::from_gnp(N, P, gen, unidirectional, selfArches),
				Structure, WeightTreatment
			);
		}

		// generates a random instance with known optimum
		// structure must be either path or cycle
		template <typename Rand>
		static std::pair<instance_t, value_type> known_looping_path_or_cycle_gnp(
			size_type N, size_type M, double P,
			Rand& gen,
			value_type MinValue, value_type MaxValue,
			weight_type MinWeight, weight_type MaxWeight,
			bool unidirectional = false, bool selfArches = true,
			structure Structure = structure::cycle, weight_treatment WeightTreatment = weight_treatment::full
		) {
			assert(Structure == structure::cycle || Structure == structure::path);

			// generate random weights and values
			std::vector<weight_type> randomWeights(N * M);
			std::vector<value_type> randomValues(N);

			random::into(randomWeights.begin(), randomWeights.end(),
				[&gen, MinWeight, MaxWeight]() {
					return gs::random::function<Rand>::from_uniform_distribution(gen, MinWeight, MaxWeight);
				});

			random::into(randomValues.begin(), randomValues.end(),
				[&gen, MinValue, MaxValue]() {
					return gs::random::function<Rand>::from_uniform_distribution(gen, MinValue, MaxValue);
				});

			// calculate cycle / path length
			size_type count_base = N * N;
			if (!selfArches) count_base -= N;
			if (unidirectional) count_base /= 2; // TO DO -> calculate it precisely
			std::normal_distribution<float> ld(count_base * P, count_base * P * (1 - P));
			size_type reslen = (size_type)std::max(0.0f, ld(gen));
			std::cout << "reslen: " << reslen << "\n";

			// construct a path / cycle
			std::vector<size_type> path;
			path.reserve(reslen);
			std::vector<bool> solution(N, false);
			std::uniform_int_distribution<size_type> sd(0, N - 1);
			if (reslen > 0 && Structure == structure::cycle) reslen -= 1;
			for (size_type i = 0; i < reslen; ++i) {
				size_type select = sd(gen);
				if (!selfArches) while (path.back() == select) select = sd(gen);
				path.push_back(select);
				solution[select] = true;
			}
			if (!path.empty() && Structure == structure::cycle) {
				path.push_back(path.front());
			}

			// calculate limits and optimum
			std::vector<weight_type> limits(M, 0);
			value_type optimum = 0;
			for (size_type itemId = 0; itemId < N; ++itemId) {
				if (solution[itemId]) {
					for (size_type i = 0; i < M; ++i)
						limits[i] += randomWeights[itemId * M + i];
					optimum += randomValues[itemId];
				}
			}

#ifndef NDEBUG
			for (auto elem : path)
				std::cout << elem << " ";
			std::cout << "\n";
#endif

			// create instance
			return std::make_pair(instance_t(
				limits.begin(), limits.end(),
				randomValues.begin(), randomValues.end(),
				randomWeights.begin(), randomWeights.end(),
				gs::graphs::adjacency_matrix::from_path(N, path.begin(), path.end(), unidirectional),
				Structure, WeightTreatment
			), optimum);
		}
	};
}
