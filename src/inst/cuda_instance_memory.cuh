#pragma once
#include <device_launch_parameters.h>
#include <cstdint>

namespace gs::inst::cuda {
	const inline size_t maxn = 100;
	const inline size_t maxm = 5;

	__constant__ uint32_t limits[maxm];
	__constant__ uint32_t values[maxn];
	__constant__ uint32_t weights[maxn * maxm];
	__constant__ uint64_t adjacency[maxm];
}
