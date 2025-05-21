#pragma once
#include <device_launch_parameters.h>
#include <cstdint>

namespace gs {
	namespace cuda {
		namespace inst {
			const size_t maxn = 64;
			const size_t maxm = 5;

			__constant__ uint32_t limits[maxm];
			__constant__ uint32_t values[maxn];
			__constant__ uint32_t weights[maxn * maxm];
			__constant__ uint64_t adjacency[maxn];

			__device__  bool has_connection_to(const uint64_t* adjacency, uint32_t from, uint32_t to);
		}
	}
}
