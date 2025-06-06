#pragma once
#include <rapidjson/prettywriter.h>
#include <rapidjson/stringbuffer.h>
#include <iostream>

namespace gs {
	namespace json {
		using namespace rapidjson;

		enum class key : uint8_t {
			limits, values, weights, weight_value_items, graph6
		};

		template <typename instance_t>
		inline void AddLimits(
			const instance_t& instance,
			PrettyWriter<StringBuffer>& writer
		) {
			using weight_type = typename instance_t::weight_type;

			writer.Key("limits");
			writer.StartArray();
			for (weight_type limit : instance.limits()) {
				if constexpr (std::is_floating_point_v<weight_type>)
					writer.Double(static_cast<double>(limit));
				else
					writer.Int64(static_cast<int64_t>(limit));
			}
			writer.EndArray();
		}

		template <typename instance_t>
		inline void AddValues(
			const instance_t& instance,
			PrettyWriter<StringBuffer>& writer
		) {
			using size_type = typename instance_t::size_type;
			using value_type = typename instance_t::value_type;

			writer.Key("values");
			writer.StartArray();
			for (size_type i = 0; i < instance.size(); ++i) {
				if constexpr (std::is_floating_point_v<value_type>)
					writer.Double(static_cast<double>(instance.value(i)));
				else
					writer.Int64(static_cast<int64_t>(instance.value(i)));
			}
			writer.EndArray();
		}

		template <typename instance_t>
		inline void AddWeights(
			const instance_t& instance,
			PrettyWriter<StringBuffer>& writer
		) {
			using size_type = typename instance_t::size_type;
			using weight_type = typename instance_t::weights_type;

			writer.Key("weights");
			writer.StartArray();
			for (size_type i = 0; i < instance.size(); ++i) {
				writer.StartArray();
				for (size_type wid = 0; wid < instance.dim(); ++wid) {
					if constexpr (std::is_floating_point_v<weight_type>)
						writer.Double(static_cast<double>(instance.weight(i, wid)));
					else
						writer.Int64(static_cast<int64_t>(instance.weight(i, wid)));
				}
				writer.EndArray();
			}
			writer.EndArray();
		}

		template <typename instance_t>
		inline void AddWeightValueItems(
			const instance_t& instance,
			PrettyWriter<StringBuffer>& writer
		) {
			using size_type = typename instance_t::size_type;
			using weight_type = typename instance_t::weights_type;
			using value_type = typename instance_t::value_type;

			writer.Key("items");
			writer.StartArray();
			for (size_type i = 0; i < instance.size(); ++i) {
				writer.StartObject();
				writer.Key("value");
				if constexpr (std::is_floating_point_v<value_type>)
					writer.Double(static_cast<double>(instance.value(i)));
				else
					writer.Int64(static_cast<int64_t>(instance.value(i)));
				writer.Key("weights");
				writer.StartArray();
				for (size_type wid = 0; wid < instance.dim(); ++wid) {
					if constexpr (std::is_floating_point_v<weight_type>)
						writer.Double(static_cast<double>(instance.weight(i, wid)));
					else
						writer.Int64(static_cast<int64_t>(instance.weight(i, wid)));
				}
				writer.EndArray();
				writer.EndObject();
			}
			writer.EndArray();
		}

		template <typename instance_t>
		using fill_function = void(*)(
			const instance_t& instance,
			PrettyWriter<StringBuffer>& writer
			);


		template <typename instance_t>
		inline std::vector<fill_function<instance_t>> fill_functions = {
			AddLimits, AddValues, AddWeights, AddWeightValueItems
		};

		template <typename instance_t>
		inline void Export(const instance_t& instance, std::initializer_list<key> keys, std::ostream& stream) {
			using size_type = typename instance_t::size_type;
			using weight_type = typename instance_t::weight_type;
			using value_type = typename instance_t::value_type;
		
			StringBuffer s;
			PrettyWriter<StringBuffer> writer(s);

			writer.StartObject();
			for (key k : keys) {
				fill_functions<instance_t>[static_cast<uint8_t>(k)](instance, writer);
			}
			writer.EndObject();

			stream << s.GetString();
		}
		template <typename instance_t>
		inline void Export(const instance_t& instance, std::initializer_list<key> keys, const char* filename) {
			std::ofstream fout(filename);
			Export(instance, keys, fout);
		}
	}
}
