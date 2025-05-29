#pragma once
#include <vector>
#include "../structure.hpp"
#include "../res/cuda_solution.hpp"
#include "../inst/cuda_instance.hpp"
#include "size_string.hpp"

namespace gs {
	namespace cuda {
		namespace solver {
			namespace grasp {
				template <typename instance_t, typename result_type>
				res::solution<result_type> runner(
					const instance_t& instance, uint32_t blocksCount, typename instance_t::index_type choose_from
				);

				extern template res::solution32 runner(const inst::instance32<uint32_t, uint32_t>&, uint32_t, typename inst::instance32<uint32_t, uint32_t>::index_type );
				extern template res::solution64 runner(const inst::instance64<uint32_t, uint32_t>&, uint32_t, typename inst::instance32<uint32_t, uint32_t>::index_type );
			}

			template <typename InstanceT>
			class GRASP {
			public:
				using instance_t = InstanceT;
				using storage_t = typename instance_t::adjacency_base_type;
				using solution_t = res::solution<storage_t>;
				const static std::string name;

				GRASP() = delete;

				inline static solution_t solve(const instance_t& instance, uint32_t blocksCount = 1, typename inst::instance32<uint32_t, uint32_t>::index_type choose_from = 0) 
				{
					if (choose_from == 0) choose_from = 1;
					return grasp::runner<instance_t, storage_t>(instance, blocksCount, choose_from);
				}
			};
			template <typename InstanceT>
			const std::string GRASP<InstanceT>::name = std::string("cudaGRASP") + size_string<typename GRASP<InstanceT>::storage_t>();
		}
	}
}

