#pragma once
#include <cassert>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <utility>
#include <fstream>

namespace gs {
	namespace inst {
		template <typename WeightValueT, typename graphT>
		class composite {
		public:
			using weight_value_t = WeightValueT;
			using graph_t = graphT;

		public:
			weight_value_t wv;
			graph_t g;

			inline composite() {}

			inline composite(const std::string filename) : composite() {
				std::ifstream fin(filename);
				if (!fin.is_open()) throw std::runtime_error("could not open file + " filename);
				fin >> (*this);
				fin.close();
			}
			
			friend inline std::ostream& operator<< (std::ostream& stream, const composite& inst) {
				return stream << inst.wv << inst.g;
			}

			friend inline std::istream& operator>> (std::istream& stream, composite& inst) {
				return inst.wv >> inst.g;
			}
		};
	}
}
