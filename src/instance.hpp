#pragma once
#include <cassert>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <utility>
#include <fstream>

#include "iterator.hpp"

namespace gs {
	template <typename WeightValueT, typename graphT>
	class composite_instance {
	public:
		using weight_value = WeightValueT;
		using graph_type = graphT;

	public:
		weight_value wv;
		graph_type g;

		inline composite_instance() {}

		inline composite_instance(const std::string filename) : composite_instance() {
			std::ifstream fin(filename);
			fin >> (*this);
			fin.close();
		}
		
		friend inline std::ostream& operator<< (std::ostream& stream, const composite_instance& inst) {
			return stream << inst.wv << inst.g;
		}

		friend inline std::istream& operator>> (std::istream& stream, composite_instance& inst) {
			return inst.wv >> inst.g;
		}
	};
}
