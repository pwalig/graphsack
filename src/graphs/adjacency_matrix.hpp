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

			inline static adjacency_matrix from_graph6(const std::string& buff) {
				char bit = 32;
				typename std::string::size_type poz = 0;
				size_type n = 0;
				if (buff[0] == '~') {
					if (buff[1] == '~') {
						poz = 2;
						while (poz < 8) {
							n <<= 6;
							n += buff[poz++] - 63;
						}
					} else {
						poz = 1;
						while (poz < 4) {
							n <<= 6;
							n += buff[poz++] - 63;
						}
					}
				}
				else {
					n = buff[poz++] - 63;
				}
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

			inline std::string graph6() {
				size_type n = size();
				std::string result;

				if (n <= 62) {
					result += (char)(n + 63);
				}
				else if (n <= 258047) {
					result += '~';
					for (size_type i = 1 << 12; i > 0; i >>= 6) {
						result += (char)(((n / i) % 64) + 63);
					}
				}
				else if (n <= 68719476735) {
					result += "~~";
					for (size_type i = 1 << 12; i > 0; i >>= 6) {
						result += (char)(((n / i) % 64) + 63);
					}
				}
				else throw std::logic_error("graph size is to large for graph6 representation");

				std::vector<bool> bitVector;
				for (size_type j = 1; j < n; ++j) {
					for (size_type i = 0; i < j; ++i) {
						bitVector.push_back(at(i, j));
					}
				}

				// Make the length a multiple of 6 by padding with 0s
				size_type len = bitVector.size();
				while (len % 6 != 0) {
					bitVector.push_back(false);
					len++;
				}

				// Convert every 6 bits to a base64 character
				for (size_type i = 0; i < len; i += 6) {
					unsigned char val = 0;
					for (size_type j = 0; j < 6; ++j) {
						val = val * 2 + (bitVector[i + j] ? 1 : 0);
					}
					result += (char)(val + 63);
				}

				return result;
			}

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
