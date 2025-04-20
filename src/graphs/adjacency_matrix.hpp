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
			adjacency_matrix(std::initializer_list<std::initializer_list<bool>> init);

			inline adjacency_matrix(size_type n) : storage(n * n) { }

			inline adjacency_matrix(size_type n, bool val)
				: storage(n * n) {
				std::fill(storage.begin(), storage.end(), val);
			}

			inline size_type size() const {
				return static_cast<size_type>(sqrt(storage.size()));
			}

			inline reference at(size_type x, size_type y) {
				return storage[x * size() + y];
			}

			inline const_reference at(size_type x, size_type y) const {
				return storage[x * size() + y];
			}

			template <typename Engine>
			inline static adjacency_matrix from_gnp(
				size_type n, double p, Engine& randomEngine,
				bool unidirectional = false, bool selfArches = true
			) {
				adjacency_matrix res(n);
				for (size_type i = 0; i < n; ++i) {
					for (size_type j = 0; j < n; ++j) {
						if (!selfArches && i == j) res.at(i, j) = false;
						else res.at(i, j) = std::bernoulli_distribution(p)(randomEngine);
						if (unidirectional) {
							if (i == j) break;
							res.at(j, i) = res.at(i, j);
						}
					}
				}
				return res;
			}

			static adjacency_matrix from_graph6(const std::string& buff);

			std::string graph6();

			template <typename T>
			inline std::vector<T> flatten() const {
				std::vector<T> out(size() + 1);
				size_type poz;
				out[0] = T(0);
				poz = 1; 
				for (size_type i = 0; i < size(); ++i) {
					for (size_type j = 0; j <= i; ++j) {
						if (i == j) out[poz++] = T(0);
						else out[poz++] = at(i, j) ? T(1) : T(0);
					}
				}
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
