#pragma once
#include <vector>
#include <numeric>

namespace gs {
	namespace metric {
		template <typename T, typename instanceT>
		using function = T(*)(const instanceT&, typename instanceT::size_type);


		template <typename T, typename instanceT>
		inline T value(const instanceT& instance, typename instanceT::size_type itemId) {
			return static_cast<T>(instance.value(itemId));
		}

		template <typename T, typename instanceT>
		inline T total_weight(const instanceT& instance, typename instanceT::size_type itemId) {
			const auto& weights = instance.weights(itemId);
			return static_cast<T>(std::accumulate(weights.begin(), weights.end(), 0));
		}

		template <typename T, typename instanceT>
		inline T nexts_count(const instanceT& instance, typename instanceT::size_type itemId) {
			return instance.nexts(itemId).size();
		}

		template <typename T, typename instanceT>
		inline T value_weight_ratio(const instanceT& instance, typename instanceT::size_type itemId) {
			return value<T, instanceT>(instance, itemId) / total_weight<T, instanceT>(instance, itemId);
		}

		template <typename T, typename instanceT>
		inline T nexts_count_value(const instanceT& instance, typename instanceT::size_type itemId) {
			return nexts_count<T, instanceT>(instance, itemId) * value<T, instanceT>(instance, itemId);
		}

		template <typename T, typename instanceT>
		inline T nexts_count_weight_ratio(const instanceT& instance, typename instanceT::size_type itemId) {
			return nexts_count<T, instanceT>(instance, itemId) / total_weight<T, instanceT>(instance, itemId);
		}

		template <typename T, typename instanceT>
		inline T nexts_count_value_weight_ratio(const instanceT& instance, typename instanceT::size_type itemId) {
			return nexts_count_value<T, instanceT>(instance, itemId) / total_weight<T, instanceT>(instance, itemId);
		}

		template <typename T, typename instanceT>
		inline void calculate_into(const instanceT& instance, function<T, instanceT> metricFunction) {
			std::vector<T> metrix(instance.size());
			for (typename instanceT::size_type i = 0; i < instance.size(); ++i) {
				metrix[i] = metricFunction(instance, i);
			}
			return metrix;
		}

		template <typename T, typename instanceT>
		inline std::vector<T> calculate(const instanceT& instance, function<T, instanceT> metricFunction) {
			std::vector<T> metrix(instance.size());
			for (typename instanceT::size_type i = 0; i < instance.size(); ++i) {
				metrix[i] = metricFunction(instance, i);
			}
			return metrix;
		}

		template <typename T, typename instanceT, typename indexT = typename instanceT::index_type>
		inline static std::vector<indexT> sorted_indexes(
			const instanceT& instance,
			function<T, instanceT> metricFunction,
			bool ascending = false
		) {
			std::vector<indexT> res(instance.size());
			for (indexT i = 0; i < instance.size(); ++i) {
				res[i] = i;
			}
			std::vector<T> metric = calculate<T, instanceT>(instance, metricFunction);
			if (ascending) std::sort(res.begin(), res.end(), [&metric](indexT a, indexT b) { return metric[a] < metric[b]; });
			else std::sort(res.begin(), res.end(), [&metric](indexT a, indexT b) { return metric[a] > metric[b]; });
			return res;
		}

		template <typename T>
		class Value {
		public:
			using value_type = T;
			inline static std::string name = "Value";

			template <typename instanceT>
			inline static value_type function(const instanceT& instance, typename instanceT::size_type itemId) {
				return metric::value<T, instanceT>(instance, itemId);
			}
		};

		template <typename T>
		class TotalWeight {
		public:
			using value_type = T;
			inline static std::string name = "TotalWeight";

			template <typename instanceT>
			inline static value_type function(const instanceT& instance, typename instanceT::size_type itemId) {
				return metric::total_weight<T, instanceT>(instance, itemId);
			}
		};

		template <typename T>
		class NextsCount {
		public:
			using value_type = T;
			inline static std::string name = "NextsCount";

			template <typename instanceT>
			inline static value_type function(const instanceT& instance, typename instanceT::size_type itemId) {
				return metric::nexts_count<T, instanceT>(instance, itemId);
			}
		};

		template <typename T>
		class NextsCountValue {
		public:
			using value_type = T;
			inline static std::string name = "NextsCountValue";

			template <typename instanceT>
			inline static value_type function(const instanceT& instance, typename instanceT::size_type itemId) {
				return metric::nexts_count_value<T, instanceT>(instance, itemId);
			}
		};

		template <typename T>
		class ValueWeightRatio {
		public:
			using value_type = T;
			inline static std::string name = "ValueWeightRatio";

			template <typename instanceT>
			inline static value_type function(const instanceT& instance, typename instanceT::size_type itemId) {
				return metric::value_weight_ratio<T, instanceT>(instance, itemId);
			}
		};

		template <typename T>
		class NextsCountWeightRatio {
		public:
			using value_type = T;
			inline static std::string name = "NextsCountWeightRatio";

			template <typename instanceT>
			inline static value_type function(const instanceT& instance, typename instanceT::size_type itemId) {
				return metric::nexts_count_weight_ratio<T, instanceT>(instance, itemId);
			}
		};

		template <typename T>
		class NextsCountValueWeightRatio {
		public:
			using value_type = T;
			inline static std::string name = "NextsCountValueWeightRatio";

			template <typename instanceT>
			inline static value_type function(const instanceT& instance, typename instanceT::size_type itemId) {
				return metric::nexts_count_value_weight_ratio<T, instanceT>(instance, itemId);
			}
		};


		template <typename metricT, typename instanceT>
		inline std::vector<typename metricT::value_type> calculate(const instanceT& instance) {
			return calculate<typename metricT::value_type, instanceT>(instance, metricT::function);
		}

		template <typename metricT, typename instanceT, typename indexT = typename instanceT::index_type>
		inline std::vector<indexT> sorted_indexes(const instanceT& instance, bool ascending = false) {
			return sorted_indexes<typename metricT::value_type, instanceT, indexT>(instance, metricT::function, ascending);
		}
	}
}
