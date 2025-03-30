#pragma once
#include <cassert>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <utility>
#include <fstream>

#include "iterator.hpp"

namespace gs {
	template <typename weightT = size_t>
	class weight_value_vector {
	public:
		using value_type = weightT;

		// proxy class for accessing single item weights
		template <typename T = weight_value_vector::value_type>
		class Item;

		template <typename T = weight_value_vector::value_type>
		class Weights {
		public:
			using value_type = T;
			using reference = value_type&;
			using pointer = value_type*;
			using const_reference = const reference;
			using const_pointer = const pointer;
		private:
			pointer ptr;
			const size_t siz;
		public:
			Weights(pointer data_pointer, size_t M) : ptr(data_pointer), siz(M) {}
			friend class Item<T>;
			inline reference operator[] (size_t i) { assert(i < siz); return ptr[i]; }
			inline const_reference operator[] (size_t i) const { assert(i < siz); return ptr[i]; }
			inline reference at(size_t i) {
				if (i >= siz) throw std::out_of_range("weight subscript out of range!");
				return ptr[i];
			}
			inline const_reference at(size_t i) const {
				if (i >= siz) throw std::out_of_range("weight subscript out of range!");
				return ptr[i];
			}
			inline size_t size() const { return siz; }
			inline pointer data() const { return ptr; }
			inline value_type total() const {
				value_type sum;
				for (size_t i = 0; i < siz; ++i) sum += ptr[i];
				return sum;
			}

			trivial_iterator_defs(value_type, ptr, siz)

			friend inline std::ostream& operator<< (std::ostream& stream, const Weights& iw) {
				for (const auto& elem : iw) {
					stream << elem << " ";
				}
				return stream;
			}
		};

		template<typename T>
		class Item {
		public:
			using value_type = T;
			using reference = value_type&;
			using pointer = value_type*;
			using const_reference = const reference;
			using const_pointer = const pointer;
		public:
			Weights<T> weights;
			friend class weight_value_vector;
			Item(pointer item_pointer, size_t item_size) : weights(item_pointer, item_size - 1) {}

			inline reference value() { return weights.ptr[weights.siz]; }
			inline const_reference value() const { return weights.ptr[weights.siz]; }

			friend inline std::ostream& operator<< (std::ostream& stream, const Item& it) {
				for (const auto& elem : it.weights) {
					stream << elem << " ";
				}
				stream << it.weights.data()[it.weights.size()];
				return stream;
			}
		};


		using weights = Weights<value_type>;
		using const_weights = const Weights<const value_type>;
		using item = Item<value_type>;
		using const_item = const Item<const value_type>;

	private:
		std::vector<value_type> _data;
		size_t _n;
		size_t _m;
		inline size_t its() const { return _m + 1; }
	public:

		weight_value_vector() : _data(), _n(0), _m(0) {}
		weight_value_vector(
			std::initializer_list<value_type> limits,
			std::initializer_list<std::pair<value_type, std::initializer_list<value_type>>> items
		) : _data(limits.size() + (items.size() * (limits.size() + 1))), _n(items.size()), _m(limits.size()) {
			size_t i = 0;
			for (const auto& limit : limits) {
				_data[i++] = limit;
			}
			for (const auto& item : items) {
				assert(item.second.size() == limits.size());
				for (const auto& weight : item.second) {
					_data[i++] = weight;
				}
				_data[i++] = item.first;
			}
		}

		weight_value_vector(size_t M, size_t N, const std::vector<value_type>& data) : _data(data), _m(M), _n(N) {}
		weight_value_vector(const std::string filename) : weight_value_vector() {
			std::ifstream fin(filename);
			fin >> (*this);
			fin.close();
		}
		
		using iterator = trivial_proxy_random_access_iterator<value_type, item>;
		using const_iterator = trivial_proxy_random_access_iterator<const value_type, const_item>;
		proxy_all_iterator_defs(_data.data() + _m, its(), _n)

		inline weights limits() { return weights(_data.data(), _m); }
		inline const_weights limits() const { return const_weights(_data.data(), _m); }

		inline item operator[](size_t i) { return item(_data.data() + _m + (i * its()), its()); }
		inline const_item operator[](size_t i) const { return const_item(_data.data() + _m + (i * its()), its()); }

		inline item at(size_t i) {
			if (i >= _n) throw std::out_of_range("item index out of range!");
			return (*this)[i];
		}
		inline const_item at(size_t i) const {
			if (i >= _n) throw std::out_of_range("item index out of range!");
			return (*this)[i];
		}

		inline size_t size() const { return _n; }
		inline size_t dim() const { return _m; }

		inline item get_item(size_t i) { return (*this)[i]; }
		inline const_item get_item(size_t i) const { (*this)[i]; }

		inline value_type value(size_t i) { _data[_m + _m + (i * its())]; }
		inline const value_type& value(size_t i) const { _data[_m + _m + (i * its())]; }
		
		inline value_type& weight(size_t i, size_t w) { return _data[_m + (i * its()) + w]; }
		inline const value_type& weight(size_t i, size_t w) const { return _data[_m + (i * its()) + w]; }

		friend inline std::ostream& operator<< (std::ostream& stream, const weight_value_vector& inst) {
			stream << inst.limits() << "\n";
			for (const auto& item : inst) {
				stream << item << "\n";
			}
			return stream;
		}

		friend inline std::istream& operator>> (std::istream& stream, weight_value_vector& inst) {
			stream >> inst._m >> inst._n;
			inst._data.resize(inst._m + ((inst._m + 1) * inst._n));
			for (auto& value : inst._data) {
				stream >> value;
			}
			return stream;
		}
	};
}
