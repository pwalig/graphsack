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
			double S = 0.0;
			std::vector<double> N;
			double structure = 0.0;
			double fitting = 0.0;
		};

		class sum;

		class single {
		public:
			double time = 0.0;
			value_type S = 0;
			std::vector<weight_type> N;
			bool structure = false;
			bool fitting = false;

			inline sum operator+(const single& other) {
				sum res;
				res.time = time + other.time;
				res.S = S + other.S;
				res.N = add_weights<std::vector<size_t>>(N, other.N);
				res.structure = 0;
				if (structure) res.structure += 1;
				if (other.structure) res.structure += 1;
				res.fitting = 0;
				if (fitting) res.fitting += 1;
				if (other.fitting) res.fitting += 1;
				return res;
			}

		};

		class sum {
		public:
			double time = 0.0;
			value_type S = 0;
			std::vector<weight_type> N;
			size_t structure = 0;
			size_t fitting = 0;

			inline sum& operator+=(const sum& other) {
				time += other.time;
				structure += other.structure;
				fitting += other.fitting;
				S += other.S;
				add_to_weights(N, other.N);
				return (*this);
			}

			inline sum& operator+=(const single& other) {
				time += other.time;
				if (other.structure) structure += 1;
				if (other.fitting) fitting += 1;
				S += other.S;
				add_to_weights(N, other.N);
				return (*this);
			}

			inline sum& operator-=(const sum& other) {
				time -= other.time;
				structure -= other.structure;
				fitting -= other.fitting;
				S -= other.S;
				sub_from_weights(N, other.N);
				return (*this);
			}

			inline sum& operator-=(const single& other) {
				time -= other.time;
				if (other.structure) structure -= 1;
				if (other.fitting) fitting -= 1;
				S -= other.S;
				sub_from_weights(N, other.N);
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
				res.S = (double)S / (double)n;
				for (size_t i = 0; i < N.size(); ++i) {
					res.N.push_back((double)N[i] / (double)n);
				}
				res.structure = (double)structure / (double)n;
				res.fitting = (double)fitting / (double)n;
				return res;
			}
		};

	};
}
