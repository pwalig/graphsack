#pragma once
#include <vector>
#include <cassert>
#include <initializer_list>
#include <iostream>
#include "../slice.hpp"

namespace gs {
	namespace grahps {
		template <typename indexT, typename sizeT = size_t>
		class nexts_proxy {
		public:
			using index_t = indexT;
			using size_type = sizeT;

			index_t* ptr;
			size_type siz;

			trivial_iterator_defs(index_t, ptr, siz)
		};

		template <typename indexT, typename sizeT = size_t>
		class nexts_list_view {
		public:
			using index_t = indexT;
			using size_type = sizeT;
			using proxy_t = slice<index_t, size_type>;
			using const_proxy_t = const slice<const index_t, size_type>;
			using slice_t = slice<index_t, size_type>;
		private:
			slice_t _data;
			size_type _n;
		public:
			inline nexts_list_view(const slice<index_t, size_type>& data, size_type N) : _data(data), _n(N) {}
			inline nexts_list_view(
				const slice_t& data,
				std::initializer_list<std::initializer_list<index_t>> init
			) : _data(data), _n(init.size()) {
				size_type acc = init.size();
				size_type i = 0;
				for (const auto& vert : init) {
					_data[i] = acc;
					acc += vert.size();
					size_type j = 0;
					for (index_t ind : vert) {
						_data[_data[i] + j] = ind;
						++j;
					}
					++i;
				}
			}

			inline proxy_t operator[] (size_type i) {
				assert(i < _n);
				return proxy_t(
					&(_data[_data[i]]), 
					(i + 1 == _n ? _data.size() : _data[i + 1]) - _data[i]
				);
			}

			inline const_proxy_t operator[] (size_type i) const {
				assert(i < _n);
				return const_proxy_t(
					&(_data[_data[i]]), 
					(i + 1 == _n ? _data.size() : _data[i + 1]) - _data[i]
				);
			}

			inline size_type size() const { return _n; }

			inline friend std::ostream& operator<< (std::ostream& stream, const nexts_list_view& x) {
				for (size_type i = 0; i < x.size(); ++i) {
					for (index_t ind : x[i])
						stream << ind << " ";
					stream << "\n";
				}
				return stream;
			}
		};

		template <typename indexT, typename sizeT = size_t>
		class nexts_list {
		public:
			using index_t = indexT;
			using size_type = sizeT;
			using slice_t = slice<index_t, size_type>;
			using const_slice_t = const slice<const index_t, size_type>;
		private:
			std::vector<indexT> _data;
			inline slice_t get_slice() { return slice_t(_data.data(), _data.size()); }
			inline const_slice_t get_slice() const { return const_slice_t(_data.data(), _data.size()); }
			inline static size_type get_required_size(std::initializer_list<std::initializer_list<index_t>> init) {
				size_type acc = init.size();
				for (const auto& vert : init) acc += vert.size();
				return acc;
			}
		public:
			nexts_list_view<index_t, size_type> view;

			inline nexts_list() : _data(), view(get_slice(), 0) {}
			inline nexts_list(std::initializer_list<std::initializer_list<index_t>> init)
				: _data(get_required_size(init)), view(get_slice(), init) {
			}
		};
	}
}
