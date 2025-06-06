#pragma once
#include <rapidjson/prettywriter.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>
#include <iostream>

namespace gs {
	namespace json {
		enum class key : uint8_t {
			limits, values, weights, weight_value_items, graph6
		};

		template <typename instance_t, template<class> class WriterT = rapidjson::Writer>
		struct writer {
		private:
			using RapidJsonWriterT = WriterT<rapidjson::StringBuffer>;

			using size_type = typename instance_t::size_type;
			using value_type = typename instance_t::value_type;
			using weight_type = typename instance_t::weight_type;

			static void add_limits(
				const instance_t& instance,
				RapidJsonWriterT& writer
			) {
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

			static void add_values(
				const instance_t& instance,
				RapidJsonWriterT& writer
			) {

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

			static void add_weights(
				const instance_t& instance,
				RapidJsonWriterT& writer
			) {
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

			static void add_weight_value_items(
				const instance_t& instance,
				RapidJsonWriterT& writer
			) {
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

			using fill_function = void(*)(
				const instance_t& instance,
				RapidJsonWriterT& writer
				);


			inline static const std::vector<fill_function> fill_functions = {
				add_limits, add_values, add_weights, add_weight_value_items
			};

		public:
			static void write(const instance_t& instance, std::initializer_list<key> keys, std::ostream& stream) {
				rapidjson::StringBuffer s;
				RapidJsonWriterT writer(s);

				writer.StartObject();
				for (key k : keys) {
					fill_functions[static_cast<uint8_t>(k)](instance, writer);
				}
				writer.EndObject();

				stream << s.GetString();
			}

			static void write(const instance_t& instance, std::initializer_list<key> keys, const char* filename) {
				std::ofstream fout(filename);
				write(instance, keys, fout);
			}
		};

		template <typename instance_t>
		using pretty_writer = writer<instance_t, rapidjson::PrettyWriter>;
	}
}
