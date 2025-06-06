#pragma once
#include <cstdint>

namespace gs {
	namespace graphs {
		enum class representation : uint8_t {
			adjacency, nexts_list, arch_list
		};
	}
}
