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
		using index_type = typename instance_t::index_type;
		using size_type = typename instance_t::size_type;
		
	private:
		template <typename Rand>
		static std::vector<index_type> random_path(
			size_type N, size_type PathLen, Rand& gen
		) {
			assert(PathLen <= N && PathLen >= 0);
			std::uniform_int_distribution<index_type> sd(0, static_cast<index_type>(N-1));
			std::vector<index_type> path;
			path.reserve(N + 1); // +1 is for potential cycle
			std::vector<bool> selected(N, false);
			for (size_type i = 0; i < PathLen; ++i) {
				index_type select = sd(gen);
				while (selected[select]) select = sd(gen);
				path.push_back(select);
				selected[select] = true;
			}
			return path;
		}

	public:

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
			std::vector<value_type> randomValues(N);

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
		static std::pair<instance_t, value_type> known_path_or_cycle_gnp(
			size_type N, size_type M, size_type PathLen,
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

			// construct a path / cycle
			std::vector<index_type> path = random_path<Rand>(N, PathLen, gen);

			// calculate limits and optimum
			std::vector<weight_type> limits(M, 0);
			value_type optimum = 0;
			for (index_type itemId : path) {
				for (size_type i = 0; i < M; ++i)
					limits[i] += randomWeights[itemId * M + i];
				optimum += randomValues[itemId];
			}

			// close path if cycle required
			if (!path.empty() && Structure == structure::cycle) {
				path.push_back(path.front());
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
