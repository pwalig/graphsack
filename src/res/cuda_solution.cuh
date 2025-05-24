#pragma once
#include <device_launch_parameters.h>
#include <cstdint>

namespace gs {
	namespace cuda {
		namespace res {
			template <typename res_type, typename index_type>
			inline __device__ bool has(res_type solution, index_type itemId) {
				if (solution & (res_type(1) << itemId)) return true;
				else return false;
			}

			template <typename res_type, typename index_type>
			inline __device__ void add(res_type& solution, index_type itemId)
			{
				solution |= (res_type(1) << itemId);
			}

			template <typename res_type, typename index_type>
			inline __device__ void remove(res_type& solution, index_type itemId)
			{
				solution &= ~(res_type(1) << itemId);
			}
		}
	}
}
