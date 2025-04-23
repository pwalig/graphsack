#pragma once
#include <vector>

#include "weight_vector_operations.hpp"
#include "structure_check.hpp"
#include "weight_treatment.hpp"
#include "structure.hpp"

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
				if (solution.has(i)) res += instance.value(i);
			}
			return res;
		}

		inline static std::vector<typename instance_t::weight_type> getResultWeights(
			const instance_t& instance,
			const solution_t& solution
		) {
			std::vector<typename instance_t::weight_type> res(instance.dim());
			for (size_t i = 0; i < instance.size(); ++i) {
				if (solution.has(i)) add_to_weights(res, instance.weights(i));
			}
			return res;
		}

		inline static bool validateFit(
			const instance_t& instance,
			const std::vector<typename instance_t::weight_type>& resultN
		) {
			switch (instance.weight_treatment())
			{
			case weight_treatment::ignore:
				return true;
				break;
			case weight_treatment::first_only:
				return instance.limit(0) >= resultN[0];
				break;
			case weight_treatment::as_ones:
				throw std::logic_error("as ones fit check requires result instance");
				break;
			case weight_treatment::full:
				return fits(resultN, instance.limits());
				break;
			default:
				throw std::logic_error("unknown weight treatment");
				break;
			}
		}

		inline static bool validateFit(
			const instance_t& instance,
			const solution_t& result
		) {
			switch (instance.weight_treatment())
			{
			case weight_treatment::ignore:
				return true;
				break;
			case weight_treatment::first_only:
				typename instance_t::weight_type w = 0;
				for (typename instance_t::size_type i = 0; i < instance.size(); ++i) if (result.has(i)) w += instance.weight(i, 0);
				return instance.limit(0) >= w;
				break;
			case weight_treatment::as_ones:
				typename instance_t::weight_type w = result.selected_count();
				for (auto limit : instance.limits()) if (limit < w) return false;
				return true;
				break;
			case weight_treatment::full:
				return fits(getResultWeights(instance, result), instance.limits());
				break;
			default:
				throw std::logic_error("unknown weight treatment");
				break;
			}
		}

		inline static bool validateStructure(
			const instance_t& instance,
			const solution_t& result
		) {
			switch (instance.structure_to_find())
			{
			case structure::none:
				return true;
				break;
			case structure::path:
				return is_path(instance, result);
				break;
			case structure::cycle:
				return is_cycle(instance, result);
				break;
			default:
				throw std::logic_error("not implemented");
				break;
			}
		}
	};
}
