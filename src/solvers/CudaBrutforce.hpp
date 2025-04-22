#pragma once
#include <vector>

#include "../bit_vector.hpp"

namespace gs {
	namespace solver {
		namespace cuda {
			class BruteForce {
			private:
				static uint32_t runner_u32_u32(uint32_t* data, uint32_t N, uint32_t M);
			public:
				using solution_t = bit_vector;
				static solution_t solve(uint32_t* data, uint32_t N, uint32_t M);
			};
		}
	}
}
