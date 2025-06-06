#pragma once
#include <rapidjson/prettywriter.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <iostream>
#include <fstream>

namespace gs {
	namespace json {
		enum class key : uint8_t {
			limits, values, weights, weight_value_items, graph6
		};

		template <typename instance_t, template<class> class WriterT = rapidjson::Writer>
		struct writer {
		private:
			using RapidJsonWriterT = WriterT<rapidjson::StringBuffer>;
			//using RapidJsonWriterT = rapidjson::Writer<rapidjson::StringBuffer>;

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

		template<typename instance_t>
		struct reader {
			using size_type = typename instance_t::size_type;
			using value_type = typename instance_t::value_type;
			using weight_type = typename instance_t::weight_type;

			static instance_t read(const char* filename) {
				std::ifstream ifs(filename);
				rapidjson::IStreamWrapper isw(ifs);

				rapidjson::Document document;
				document.ParseStream(isw);
				assert(document.IsObject());

				std::vector<weight_type> limits;
				std::vector<value_type> values;
				std::vector<weight_type> weights;
				graphs::adjacency_matrix graph;
				gs::weight_treatment wt = gs::weight_treatment::ignore;
				gs::structure st = gs::structure::none;

				if (document.HasMember("limits")) {
					wt = gs::weight_treatment::full;
					const rapidjson::Value& limitsNode = document["limits"];
					limits.reserve(limitsNode.Size());
					for (rapidjson::SizeType i = 0; i < limitsNode.Size(); ++i) {
						limits.push_back(limitsNode[i].Get<weight_type>());
					}
				}

				if (document.HasMember("values")) {
					const rapidjson::Value& valuesNode = document["values"];
					assert(valuesNode.IsArray());
					values.reserve(valuesNode.Size());
					for (rapidjson::SizeType i = 0; i < valuesNode.Size(); ++i) {
						values.push_back(valuesNode[i].Get<value_type>());
					}
				}

				if (document.HasMember("weights")) {
					const rapidjson::Value& weightsNode = document["weights"];
					assert(weightsNode.IsArray());
					weights.reserve(weightsNode.Size() * limits.size());
					for (rapidjson::SizeType i = 0; i < weightsNode.Size(); ++i) {
						const rapidjson::Value& wNode = weightsNode[i];
						assert(wNode.IsArray());
						assert(wNode.Size() == limits.size());
						for (rapidjson::SizeType wid = 0; wid < limits.size(); ++wid) {
							weights.push_back(wNode[wid].Get<weight_type>());
						}
					}
				}

				if (document.HasMember("graph6")) {
					st = gs::structure::cycle;
					const rapidjson::Value& graph6Node = document["graph6"];
					graph = graphs::adjacency_matrix::from_graph6(graph6Node.GetString());
				}

				if (graph.size() == 0) graph = graphs::adjacency_matrix(values.size(), true);

				return instance_t(
					limits.begin(), limits.end(),
					values.begin(), values.end(),
					weights.begin(), weights.end(),
					graph, st, wt
				);
			}
		};
	}
}
