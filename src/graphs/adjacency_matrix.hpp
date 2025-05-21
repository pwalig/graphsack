#pragma once

#include <vector>
#include <cassert>
#include <ostream>
#include <random>
#include <type_traits>

#include "../slice.hpp"
#include "../inst/gs_random.hpp"

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

			template<typename T>
			class Row {
			private:
				T& storage;
				const size_type x;

			public:
				inline Row(T& Storage, size_type& X) : storage(Storage), x(X) {}

				inline size_type size() const {
					return static_cast<size_type>(sqrt(storage.size()));
				}
				inline typename std::vector<bool>::iterator begin() {
					return storage.begin() + (x * size());
				}
				inline typename std::vector<bool>::const_iterator begin() const {
					return storage.begin() + (x * size());
				}
				inline typename std::vector<bool>::const_iterator cbegin() const {
					return storage.cbegin() + (x * size());
				}
				inline typename std::vector<bool>::iterator end() {
					return storage.begin() + ((x + 1) * size());
				}
				inline typename std::vector<bool>::const_iterator end() const {
					return storage.begin() + ((x + 1) * size());
				}
				inline typename std::vector<bool>::const_iterator cend() const {
					return storage.cbegin() + ((x + 1) * size());
				}
				inline reference at(size_type y) {
					return storage[x * size() + y];
				}
				inline const_reference at(size_type y) const {
					return storage[x * size() + y];
				}
				inline reference operator[](size_type y) {
					return storage[x * size() + y];
				}
				inline const_reference operator[](size_type y) const {
					return storage[x * size() + y];
				}
			};

			using row = Row<std::vector<bool>>;
			using const_row = const Row<const std::vector<bool>>;

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

			inline row at(size_type x) {
				return row(storage, x);
			}
			inline const_row at(size_type x) const {
				return const_row(storage, x);
			}
			inline row operator[](size_type x) {
				return row(storage, x);
			}
			inline const_row operator[](size_type x) const {
				return const_row(storage, x);
			}

			size_type ones();

			template <typename Engine>
			inline static adjacency_matrix from_gnp(
				size_type n, double p, Engine& randomEngine,
				bool unidirectional = false, bool selfArches = true
			) {
				adjacency_matrix res(n);
				for (size_type i = 0; i < n; ++i) {
					for (size_type j = 0; j < (unidirectional ? i + 1 : n); ++j) {
						if (!selfArches && i == j) res.at(i, j) = false;
						else {
							res.at(i, j) = std::bernoulli_distribution(p)(randomEngine);
							if (unidirectional) res.at(j, i) = res.at(i, j);
						}
					}
				}
				return res;
			}

			template <typename Engine>
			inline void gnp_fill(
				double p, Engine& randomEngine,
				bool unidirectional = false, bool selfArches = true
			) {
				size_type n = size();
				p -= ((double)ones() / storage.size());
				for (size_type i = 0; i < n; ++i) {
					for (size_type j = 0; j < (unidirectional ? i + 1 : n); ++j) {
						if (!at(i, j)) {
							at(i, j) = std::bernoulli_distribution(p)(randomEngine);
							if (unidirectional) at(j, i) = at(i, j);
						}
					}
				}
			}

			template <typename Iter>
			static adjacency_matrix from_path(size_type N, Iter Begin, Iter End, bool unidirectional = false) {
				static_assert(std::is_integral_v<typename Iter::value_type>, "indices of path must be numbers");
				adjacency_matrix res(N, false);
				if (Begin == End) return res;
				auto it = Begin;
				auto prev = *it;
				++it;
				for (; it != End; ++it) {
					res.at(prev, *it) = true;
					if (unidirectional) res.at(*it, prev) = true;
					prev = *it;
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
#ifndef NDEBUG
			class test {
			public:
				static void initializer_list_constructable();
				template <typename Engine>
				inline static void from_gnp(
					size_type n, double p, Engine& gen,
					bool unidirectional = false, bool selfArches = true
				) {
					adjacency_matrix am = graphs::adjacency_matrix::from_gnp(n, p, gen, unidirectional, selfArches);
					assert(am.size() == n);
					if (unidirectional) {
						for (size_type i = 0; i < am.size(); ++i) {
							for (size_type j = 0; j < i; ++j) {
								assert(am[i][j] && am[j][i]);
							}
						}
					}
					if (!selfArches) {
						for (size_type i = 0; i < am.size(); ++i) assert(am[i][i] == false);
					}
				}
				static void g6_converter_consistency_from_file(const std::string& filename);
				static void g6_converter_consistency(const std::string& graph6);
				static void all();
			};
#endif

		};
	}
}
