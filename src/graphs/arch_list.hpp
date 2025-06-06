#pragma once
#include <utility>
#include <cassert>
#include "representation.hpp"

namespace gs {
	namespace graphs {
		template <typename T = size_t>
		class arch {
		public:
			using value_type = T;
			value_type from;
			value_type to;
		};

		template <typename T = size_t>
		using vertex_pair = std::pair<T, T>;

		template <class Container>
		class arch_list {
		public:
			using value_type = typename Container::value_type;
			using reference = typename Container::reference;
			using const_reference = typename Container::const_reference;
			using size_type = typename Container::size_type;
			using iterator = typename Container::iterator;
			using const_iterator = typename Container::iterator;
			inline const static graphs::representation representation = graphs::representation::arch_list;

		private:
			Container storage;
			size_type n;

		public:
			inline arch_list(size_type N, std::initializer_list<value_type> init) : storage(init), n(N) { }

			const size_type vertices() const { return n; }
			const size_type arches() const { return storage.size() / 2; }
			const size_type size() const { return storage.size(); }
			const size_type capacity() const { return storage.capacity(); }

			inline reference operator[] (size_type i) { return storage[i]; }
			inline const_reference operator[] (size_type i) const { return storage[i]; }

			inline iterator begin() { return storage.begin(); }
			inline iterator end() { return storage.end(); }
		};
	}
}
