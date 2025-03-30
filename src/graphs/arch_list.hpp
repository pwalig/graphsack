#pragma once
#include <utility>
#include <vector>

namespace gs {
	namespace graphs {
		template <typename T = size_t>
		class arch {
		public:
			using index_type = T;
			index_type from;
			index_type to;
		};

		template <typename T = size_t>
		class arch_list {
		public:
			using index_type = T;
			using arch_type = std::pair<index_type, index_type>;

			std::vector<arch_type> list;

			const size_t size() const { return list.size(); }
			const size_t capacity() const { return list.capacity(); }
		};
	}
}
