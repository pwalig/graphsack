#pragma once
#include <cassert>
#include <initializer_list>
#include <iostream>
#include "../slice.hpp"

namespace gs {
	namespace grahps {
		template <class Container>
		class nexts_list {
		public:
			using value_type = typename Container::value_type;
			using size_type = typename Container::size_type;
			using proxy_t = slice<value_type, size_type>;
			using const_proxy_t = const slice<const value_type, size_type>;
		private:
			Container _data;
			size_type _n;
			inline static size_type get_required_size(std::initializer_list<std::initializer_list<value_type>> init) {
				size_type acc = init.size();
				for (const auto& vert : init) acc += vert.size();
				return acc;
			}
			inline void fill_data(std::initializer_list<std::initializer_list<value_type>> init) {
				size_type acc = init.size();
				size_type i = 0;
				for (const auto& vert : init) {
					_data[i] = static_cast<value_type>(acc);
					acc += vert.size();
					size_type j = 0;
					for (value_type ind : vert) {
						_data[_data[i] + j] = ind;
						++j;
					}
					++i;
				}
			}
		public:
			inline nexts_list(const Container& data, size_type N) : _data(data), _n(N) {}
			inline nexts_list(
				std::initializer_list<std::initializer_list<value_type>> init
			) : _data(get_required_size(init)), _n(init.size()) {
				fill_data(init);
			}
			inline nexts_list(
				const Container& data,
				std::initializer_list<std::initializer_list<value_type>> init
			) : _data(data), _n(init.size()) {
				assert(data.size() == get_required_size(init));
				fill_data(init);
			}

			inline nexts_list(const std::string filename) : _data(), _n(0) {
				std::ifstream fin(filename);
				fin >> (*this);
				fin.close();
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

			inline friend std::istream& operator>> (std::istream& stream, nexts_list& x) {
				x._data.clear();
				stream >> x._n;
				value_type acc = static_cast<value_type>(x._n);
				x._data.resize(acc);
				for (size_type i = 0; i < x._n; ++i) {
					x._data[i] = acc;
					value_type val;
					stream >> val;
					acc += val;
					x._data.resize(x._data.size() + val);
					for (size_type j = x._data[i]; j < acc; ++j) {
						stream >> x._data[j];
					}
				}
				return stream;
			}

			inline friend std::ostream& operator<< (std::ostream& stream, const nexts_list& x) {
				stream << x._n << "\n";
				for (size_type i = 0; i < x.size(); ++i) {
					for (value_type ind : x[i])
						stream << ind << " ";
					stream << "\n";
				}
				return stream;
			}
		};

		template<typename T>
		using nexts_list_view = nexts_list<slice<T>>;
	}
}
