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
		inline T value_weight_ratio(const instanceT& instance, typename instanceT::size_type itemId) {
			return value<T, instanceT>(instance, itemId) / total_weight<T, instanceT>(instance, itemId);
		}

		template <typename T, typename instanceT>
		inline std::vector<T> calculate(const instanceT& instance, function<T, instanceT> metricFunction) {
			std::vector<T> metrix(instance.size());
			for (typename instanceT::size_type i = 0; i < instance.size(); ++i) {
				metrix[i] = metricFunction(instance, i);
			}
			return metrix;
		}

		template <typename T>
		class Value {
		public:
			using value_type = T;

			template <typename instanceT>
			inline static value_type function(const instanceT& instance, typename instanceT::size_type itemId) {
				return metric::value<T, instanceT>(instance, itemId);
			}
		};

		template <typename T>
		class TotalWeight {
		public:
			using value_type = T;

			template <typename instanceT>
			inline static value_type function(const instanceT& instance, typename instanceT::size_type itemId) {
				return metric::total_weight<T, instanceT>(instance, itemId);
			}
		};

		template <typename T>
		class ValueWeightRatio {
		public:
			using value_type = T;

			template <typename instanceT>
			inline static value_type function(const instanceT& instance, typename instanceT::size_type itemId) {
				return metric::value_weight_ratio<T, instanceT>(instance, itemId);
			}
		};

		template <typename metricT, typename instanceT>
		inline std::vector<typename metricT::value_type> calculate(const instanceT& instance) {
			return calculate<typename metricT::value_type, instanceT>(instance, metricT::function);
		}
	}
}
