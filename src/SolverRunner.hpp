#pragma once
#include <chrono>
#include <iostream>
#include <vector>

#include "Validator.hpp"

namespace gs {

	// time measuring, result validatin wrapper for dna sequencing solvers
	// Solver is required to have:
	// using word_set_t = ... (or equivalent typedef)
	// using instance_t = Instance<word_set_t> (or equivalent typedef)
	// static std::vector<size_t> solve(instance_t);
	// static std::string name;
	template <typename Solver>
	class SolverRunner {
	public:
		using instance_t = typename Solver::instance_t;
		using solution_t = typename Solver::solution_t;

		class avg_stats {
		public:
			double time = 0.0;
			double S = 0.0;
			double N = 0.0;
			double unique = 0.0;
			double fitting = 0.0;
		};

		class statistics;

		class sum_stats {
		public:
			double time = 0.0;
			size_t S = 0;
			size_t N = 0;
			size_t unique = 0;
			size_t fitting = 0;

			inline sum_stats& operator+=(const sum_stats& other) {
				time += other.time;
				unique += other.unique;
				fitting += other.fitting;
				S += other.S;
				N += other.N;
				return (*this);
			}

			inline sum_stats& operator+=(const statistics& other) {
				time += other.time;
				if (other.unique) unique += 1;
				if (other.fitting) fitting += 1;
				S += other.S;
				N += other.N;
				return (*this);
			}

			inline sum_stats& operator-=(const sum_stats& other) {
				time -= other.time;
				unique -= other.unique;
				fitting -= other.fitting;
				S -= other.S;
				N -= other.N;
				return (*this);
			}

			inline sum_stats& operator-=(const statistics& other) {
				time -= other.time;
				if (other.unique) unique -= 1;
				if (other.fitting) fitting -= 1;
				S -= other.S;
				N -= other.N;
				return (*this);
			}

			inline sum_stats operator+(const sum_stats& other) {
				sum_stats res = (*this);
				res += other;
				return res;
			}

			inline sum_stats operator+(const statistics& other) {
				sum_stats res = (*this);
				res += other;
				return res;
			}


			inline sum_stats operator-(const sum_stats& other) {
				sum_stats res = (*this);
				res -= other;
				return res;
			}

			inline sum_stats operator-(const statistics& other) {
				sum_stats res = (*this);
				res -= other;
				return res;
			}

			inline avg_stats operator/(size_t n) {
				avg_stats res;
				res.time = time / (double)n;
				res.S = (double)S / (double)n;
				res.N = (double)N / (double)n;
				res.unique = (double)unique / (double)n;
				res.fitting = (double)fitting / (double)n;
				return res;
			}
		};

		class statistics {
		public:
			double time = 0.0;
			size_t S = 0;
			size_t N = 0;
			bool unique = false;
			bool fitting = false;

			inline sum_stats operator+(const statistics& other) {
				sum_stats res;
				res.time = time + other.time;
				res.S = S + other.S;
				res.N = N + other.N;
				res.unique = 0;
				if (unique) res.unique += 1;
				if (other.unique) res.unique += 1;
				res.fitting = 0;
				if (fitting) res.fitting += 1;
				if (other.fitting) res.fitting += 1;
				return res;
			}

		};
		
		// solve the istance, mesure time and validate the solution
		// @param instance - problem instance (see `Instance.hpp`)
		// @param format - format string
		// 
		// avaliable format parameters:
		// {solver name} - name of the solver
		// {time} - solve time in seconds
		// {instance} - words in the problem instance
		// {result} - selected words in the result
		// {instance size} - size of the instance
		// {result size} - size of the result
		// {instance N} - limit N of the instance
		// {result N} - cost N of the result
		// {unique} - unique validation result
		// {fitting} - fitting validation result
		// @param stream - stream where to output
		template <typename... Args>
		inline static statistics run(
			const instance_t& instance,
			std::vector<std::pair<std::string, std::ostream&>> outputs,
			Args... args
		) {

			std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
			solution_t res = Solver::solve(instance, args...);
			std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

			statistics stats;

			stats.time = std::chrono::duration<double>(end - begin).count();

			stats.S = Validator<instance_t, solution_t>::getResultValue(instance, res);
			stats.N = Validator<instance_t, solution_t>::getResultWeights(instance, res);
			stats.unique = Validator<instance_t, solution_t>::validateUnique(instance, res);
			stats.fitting = Validator<instance_t, solution_t>::validateFit(instance.limits(), stats.N);

			for (std::pair<std::string, std::ostream&>& output : outputs) {
				size_t pos = output.first.find_first_of('{');
				while (pos != std::string::npos) {
					output.second << output.first.substr(0, pos);
					size_t pos2 = output.first.find_first_of('}');
					std::string keyval = output.first.substr(pos + 1, pos2 - pos - 1);
					if (keyval == "solver name") output.second << Solver::name;
					else if (keyval == "instance name") output.second << instance.name;
					else if (keyval == "time") output.second << stats.time;
					else if (keyval == "instance") output.second << instance.wordSet;
					else if (keyval == "result") {
						for (size_t i : res) {
							output.second << i << ": " << instance.wordSet[i] << "\n";
						}
					}
					else if (keyval == "instance size") output.second << instance.wordSet.size();
					else if (keyval == "result size") output.second << stats.S;
					else if (keyval == "instance N") output.second << instance.n;
					else if (keyval == "result N") output.second << stats.N;
					else if (keyval == "unique") output.second << (stats.unique ? "true" : "false");
					else if (keyval == "fitting") output.second << (stats.fitting ? "true" : "false");
					output.first = output.first.substr(pos2 + 1);
					pos = output.first.find_first_of('{');
				}
				output.second << output.first;
			}

			return stats;
		}

		template <typename... Args>
		inline static statistics run(
			const instance_t& instance,
			std::string format,
			std::ostream& stream = std::cout,
			Args... args
		) {
			return run<Args...>(instance, { {format, stream} }, args...);
		}

		template <typename... Args>
		inline static avg_stats run(
			const std::vector<instance_t>& instances,
			std::vector<std::pair<std::string, std::ostream&>> outputs,
			std::vector<std::pair<std::string, std::ostream&>> avgOutputs,
			Args... args
		) {
			sum_stats resultSumStats;
			sum_stats instanceSumStats;

			// solve all instances
			for (const instance_t& instance : instances) {
				resultSumStats += run<Args...>(instance, outputs, args...);
				instanceSumStats.N += instance.n;
				instanceSumStats.S += instance.wordSet.size();
			}

			// average results
			avg_stats avgResultStats = resultSumStats / instances.size();
			avg_stats avgInstanceStats = instanceSumStats / instances.size();

			// output
			for (std::pair<std::string, std::ostream&> avgOutput : avgOutputs) {
				size_t pos = avgOutput.first.find_first_of('{');
				while (pos != std::string::npos) {
					avgOutput.second << avgOutput.first.substr(0, pos);
					size_t pos2 = avgOutput.first.find_first_of('}');
					std::string keyval = avgOutput.first.substr(pos + 1, pos2 - pos - 1);
					if (keyval == "solver name") avgOutput.second << Solver::name;
					else if (keyval == "time") avgOutput.second << avgResultStats.time;
					else if (keyval == "instance size") avgOutput.second << avgInstanceStats.S;
					else if (keyval == "result size") avgOutput.second << avgResultStats.S;
					else if (keyval == "instance N") avgOutput.second << avgInstanceStats.N;
					else if (keyval == "result N") avgOutput.second << avgResultStats.N;
					else if (keyval == "unique") avgOutput.second << avgResultStats.unique;
					else if (keyval == "fitting") avgOutput.second << avgResultStats.fitting;
					avgOutput.first = avgOutput.first.substr(pos2 + 1);
					pos = avgOutput.first.find_first_of('{');
				}
				avgOutput.second << avgOutput.first;
			}

			return avgResultStats;
		}

		inline static avg_stats run(
			const std::vector<instance_t>& instances,
			std::string format,
			std::ostream& stream = std::cout
		) {
			return run(instances, { {format, stream} });
		}
	};
}
