#pragma once
#include <vector>

#include "weight_vector_operations.hpp"

namespace gs {
	template <typename instanceT, typename solutionT>
	class Validator {
	public:
		using instance_t = instanceT;
		using solution_t = solutionT;

		Validator() = delete;

		inline static typename instance_t::value_type getResultValue(
			const instance_t& instance,
			const solution_t& solution
		) {
			typename instance_t::value_type res = 0;
			for (size_t i = 0; i < instance.size(); ++i) {
				res += instance.value(i);
			}
			return res;
		}

		inline static std::vector<typename instance_t::weight_type> getResultWeights(
			const instance_t& instance,
			const solution_t& solution
		) {
			std::vector<typename instance_t::weight_type> res(instance.dim());
			for (size_t i = 0; i < instance.size(); ++i) {
				add_to_weights(res, instance.weights(i));
			}
			return res;
		}

		inline static bool validateFit(
			const typename instance_t::weights_type& instanceN,
			const std::vector<typename instance_t::weight_type>& resultN
		) {
			return fits(resultN, instanceN);
		}

		inline static bool validateFit(
			const instance_t& instance,
			const std::vector<typename instance_t::weight_type>& resultN
		) {
			return validateFit(instance.limits(), resultN);
		}

		inline static bool validateFit(
			const instance_t& instance,
			const solution_t& result
		) {
			return validateFit(instance.limits(), getResultN(instance, result));
		}
	};
}
