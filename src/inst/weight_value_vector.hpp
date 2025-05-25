#pragma once
#include <cassert>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <utility>
#include <fstream>

#include "../iterator.hpp"
#include "inst_macros.hpp"

namespace gs {
	template <typename T = unsigned int>
	class weight_value_vector {
	public:
		using value_type = T;
		using weight_type = T;
		using size_type = size_t;

		// proxy class for accessing single items
		template <typename T = value_type>
		class Item;

		template <typename T = weight_type>
		class Weights {
		public:
			using weight_type = T;
			using reference = weight_type&;
			using pointer = weight_type*;
			using const_reference = const reference;
			using const_pointer = const pointer;
		private:
			pointer ptr;
			const size_t siz;
		public:
			inline Weights(pointer data_pointer, size_t M) : ptr(data_pointer), siz(M) {}
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
			inline weight_type total() const {
				typename std::remove_const<weight_type>::type sum = 0;
				for (size_t i = 0; i < siz; ++i) sum += ptr[i];
				return sum;
			}

			trivial_iterator_defs(weight_type, ptr, siz)

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
			inline Item(pointer item_pointer, size_t item_size) : weights(item_pointer, item_size - 1) {}

			inline reference value() { return weights.ptr[weights.siz]; }
			inline const_reference value() const { return weights.ptr[weights.siz]; }

			inline T total_weight() const { return weights.total(); }

			friend inline std::ostream& operator<< (std::ostream& stream, const Item& it) {
				for (const auto& elem : it.weights) {
					stream << elem << " ";
				}
				stream << it.weights.data()[it.weights.size()];
				return stream;
			}
		};


		using weights_type = Weights<weight_type>;
		using const_weights_t = const Weights<const weight_type>;
		using item_type = Item<value_type>;
		using const_item_type = const Item<const value_type>;

	private:
		std::vector<value_type> _data;
		size_t _n;
		size_t _m;
		GS_INST_WEIGHT_TREATMENT_MEMBER

		inline size_t its() const { return _m + 1; }
	public:

		inline weight_value_vector() : _data(), _n(0), _m(0) {}
		inline weight_value_vector(
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

		inline weight_value_vector(size_t M, size_t N, const std::vector<value_type>& data) : _data(data), _m(M), _n(N) {}
		inline weight_value_vector(const std::string filename) : weight_value_vector() {
			std::ifstream fin(filename);
			if (!fin.is_open()) throw std::runtime_error("could not open file: " + filename);
			fin >> (*this);
			fin.close();
		}
		
		using iterator = trivial_proxy_random_access_iterator<value_type, item_type>;
		using const_iterator = trivial_proxy_random_access_iterator<const value_type, const_item_type>;
		proxy_all_iterator_defs(_data.data() + _m, its(), _n)

		inline weights_type limits() { return weights_type(_data.data(), _m); }
		inline const_weights_t limits() const { return const_weights_t(_data.data(), _m); }

		inline value_type limit(size_t i) { assert(i < _m);  return _data[i]; }
		inline const value_type& limit(size_t i) const { assert(i < _m);  return _data[i]; }

		inline item_type operator[](size_t i) { return item_type(_data.data() + _m + (i * its()), its()); }
		inline const_item_type operator[](size_t i) const { return const_item_type(_data.data() + _m + (i * its()), its()); }

		inline item_type at(size_t i) {
			if (i >= _n) throw std::out_of_range("item index out of range!");
			return (*this)[i];
		}
		inline const_item_type at(size_t i) const {
			if (i >= _n) throw std::out_of_range("item index out of range!");
			return (*this)[i];
		}

		inline size_t size() const { return _n; }
		inline size_t dim() const { return _m; }

		inline item_type item(size_t i) { return (*this)[i]; }
		inline const_item_type item(size_t i) const { return (*this)[i]; }

		inline value_type value(size_t i) { return _data[_m + _m + (i * its())]; }
		inline const value_type& value(size_t i) const { return _data[_m + _m + (i * its())]; }
		
		inline value_type& weight(size_t i, size_t w) { return _data[_m + (i * its()) + w]; }
		inline const value_type& weight(size_t i, size_t w) const { return _data[_m + (i * its()) + w]; }

		GS_INST_WEIGHT_TREATMENT_ACCESS
		GS_INST_WEIGHT_TREATMENT_FIRST_ONLY_ACCESS

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
