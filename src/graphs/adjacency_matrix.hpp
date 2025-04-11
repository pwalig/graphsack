#pragma once

#include <vector>
#include <cassert>
#include <ostream>
#include <random>

#include "../slice.hpp"

namespace gs {
	namespace graphs {
		class adjacency_matrix {
		public:
			using value_type = bool;
			using reference = typename std::vector<bool>::reference;
			using pointer = typename std::vector<bool>::pointer;
			using const_reference = typename std::vector<bool>::const_reference;
			using const_pointer = typename std::vector<bool>::const_pointer;
			using size_type = typename std::vector<bool>::size_type;

		private:
			std::vector<bool> storage;
		public:
			adjacency_matrix() = default;
			inline adjacency_matrix(std::initializer_list<std::initializer_list<bool>> init)
				: storage(init.size()* init.begin()->size()) {
				size_type i = 0;
				for (const auto& w : init) {
					assert(w.size() == init.size());
					for (bool c : w) {
						storage[i] = c;
						++i;
					}
				}
			}

			inline adjacency_matrix(size_type n) : storage(n * n) { }

			inline adjacency_matrix(size_type n, bool val)
				: storage(n * n) {
				std::fill(storage.begin(), storage.end(), val);
			}

			inline size_type size() const {
				return sqrt(storage.size());
			}

			inline reference at(size_type x, size_type y) {
				return storage[x * size() + y];
			}

			inline const_reference at(size_type x, size_type y) const {
				return storage[x * size() + y];
			}

			template <typename Engine>
			inline static adjacency_matrix from_gnp(size_type n, double p, Engine& eng) {
				adjacency_matrix res(n);
				for (size_type i = 0; i < n; ++i) {
					for (size_type j = 0; j < i; ++j) {
						res.at(i, j) = res.at(j, i) = std::bernoulli_distribution(p)(eng);
					}
				}
				return res;
			}

			inline static adjacency_matrix from_g6(const std::string& buff) {
				int bit = 32;
				typename std::string::size_type poz = 1;
				size_type n = buff[0] - 63;
				adjacency_matrix res(n);
				for (size_type i = 1; i < n; i++) {
					for (size_type j = 0; j < i; j++)
					{
						if (bit == 0) { bit = 32;  poz++; }
						if ((buff[poz] - 63) & bit)
							res.at(i, j) = res.at(j, i) = true;
						else
							res.at(i, j) = res.at(j, i) = false;
						bit >>= 1;
					}
				}
				return res;
			}

			friend inline std::ostream& operator<< (std::ostream& stream, const adjacency_matrix& matr) {
				for (size_type i = 0; i < matr.size(); ++i) {
					for (size_type j = 0; j < matr.size(); ++j) {
						if (matr.at(i, j)) stream << "1";
						else stream << "0";
					}
					stream << "\n";
				}
				return stream;
			}
		};
	}
}
