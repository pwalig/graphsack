#pragma once
#include <string>

namespace gs {
	namespace cuda {
		template <typename size_type>
		constexpr const char* size_string();

		template <>
		constexpr const char* size_string<uint8_t>() {
			return "8";
		}
		template <>
		constexpr const char* size_string<uint16_t>() {
			return "16";
		}
		template <>
		constexpr const char* size_string<uint32_t>() {
			return "32";
		}
		template <>
		constexpr const char* size_string<uint64_t>() {
			return "64";
		}
	}
}