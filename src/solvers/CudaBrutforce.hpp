#pragma once
#include <vector>

namespace gs {
	namespace solver {
		namespace cuda {
			namespace brute_force {
				std::vector<uint32_t> runner_u32_u32(uint32_t* data, uint32_t N, uint32_t M);
			}
		}
	}
}
