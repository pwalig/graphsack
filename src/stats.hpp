#pragma once
#include "weight_vector_operations.hpp"

namespace gs {
	template <typename instanceT>
	struct stats {
		using instance_t = instanceT;
		using value_type = typename instance_t::value_type;
		using weight_type = typename instance_t::weight_type;

		class avg {
		public:
			double time = 0.0;
			double value = 0.0;
			std::vector<double> weights;
			double structure = 0.0;
			double fitting = 0.0;
		};

		class sum;

		class single {
		public:
			std::vector<weight_type> weights;
			double time = 0.0;
			value_type value = 0;
			uint8_t validations = 0;

			inline sum operator+(const single& other) {
				sum res;
				res.time = time + other.time;
				res.value = value + other.value;
				res.weights = add_weights<std::vector<size_t>>(weights, other.weights);
				res.fitting = 0;
				if (validations & 1) res.fitting += 1;
				if (other.validations & 1) res.fitting += 1;
				res.structure = 0;
				if (validations & 2) res.structure += 1;
				if (other.validations & 2) res.structure += 1;
				return res;
			}

		};

		class sum {
		public:
			double time = 0.0;
			value_type value = 0;
			std::vector<weight_type> weights;
			size_t structure = 0;
			size_t fitting = 0;

			inline sum& operator+=(const sum& other) {
				time += other.time;
				structure += other.structure;
				fitting += other.fitting;
				value += other.value;
				add_to_weights(weights, other.weights);
				return (*this);
			}

			inline sum& operator+=(const single& other) {
				time += other.time;
				if (other.validations & 1) fitting += 1;
				if (other.validations & 2) structure += 1;
				value += other.value;
				add_to_weights(weights, other.weights);
				return (*this);
			}

			inline sum& operator-=(const sum& other) {
				time -= other.time;
				structure -= other.structure;
				fitting -= other.fitting;
				value -= other.value;
				sub_from_weights(weights, other.weights);
				return (*this);
			}

			inline sum& operator-=(const single& other) {
				time -= other.time;
				if (other.validations & 1) fitting -= 1;
				if (other.validations & 2) structure -= 1;
				value -= other.value;
				sub_from_weights(weights, other.weights);
				return (*this);
			}

			inline sum operator+(const sum& other) {
				sum res = (*this);
				res += other;
				return res;
			}

			inline sum operator+(const single& other) {
				sum res = (*this);
				res += other;
				return res;
			}


			inline sum operator-(const sum& other) {
				sum res = (*this);
				res -= other;
				return res;
			}

			inline sum operator-(const single& other) {
				sum res = (*this);
				res -= other;
				return res;
			}

			inline avg operator/(size_t n) {
				avg res;
				res.time = time / (double)n;
				res.value = (double)value / (double)n;
				for (size_t i = 0; i < weights.size(); ++i) {
					res.weights.push_back((double)weights[i] / (double)n);
				}
				res.structure = (double)structure / (double)n;
				res.fitting = (double)fitting / (double)n;
				return res;
			}
		};

		class accumulate {
		public:
			std::vector<double> times;
			std::vector<value_type> values;
			std::vector<std::vector<weight_type>> weights;
			std::vector<uint8_t> validations;

			inline void insert(const single& stat) {
				times.push_back(stat.time);
				values.push_back(stat.value);
				weights.push_back(stat.weights);
				validations.push_back(stat.validations);
			}

			inline accumulate& operator+=(const single& stat) {
				insert(stat);
			}
			inline accumulate& operator+(const single& stat) {
				accumulate tmp = (*this);
				tmp.insert(stat);
				return tmp;
			}

			inline double avg_time() const {
				double sum = std::accumulate(times.begin(), times.end(), 0.0);
				return sum / times.size();
			}

			inline double avg_value() const {
				double sum = std::accumulate(values.begin(), values.end(), 0.0);
				return sum / values.size();
			}

			inline std::vector<double> avg_weights() const {
				std::vector<double> sum(weights.size(), 0.0);
				for (size_t i = 0; i < weights.back().size(); ++i) {
					for (size_t j = 0; j < weights.size(); ++j) {
						sum[i] += weights[j][i];
					}
					sum[i] /= weights.size();
				}
				return sum;
			}

			inline double avg_valid(uint8_t validation_number) const {
				uint8_t mask = 1 << validation_number;
				double sum = std::count_if(validations.begin(), validations.end(), [](uint8_t val) {return val & mask; });
				return sum / validations.size();
			}

			inline avg operator/(size_t n) {
				avg res;
				res.time = avg_time();
				res.value = avg_value();
				res.weights = avg_weights();
				res.fitting = avg_valid(0);
				res.structure = avg_valid(1);
				return res;
			}
		};
	};
}
