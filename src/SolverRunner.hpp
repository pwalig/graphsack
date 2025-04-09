#pragma once
#include <chrono>
#include <iostream>
#include <vector>

#include "Validator.hpp"
#include "stats.hpp"

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
		// {structure} - structure validation result
		// {fitting} - fitting validation result
		// @param stream - stream where to output
		template <typename... Args>
		inline static typename stats<instance_t>::single run(
			const instance_t& instance,
			std::vector<std::pair<std::string, std::ostream&>> outputs,
			Args... args
		) {

			std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
			solution_t res = Solver::solve(instance, args...);
			std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

			typename stats<instance_t>::single stats;

			stats.time = std::chrono::duration<double>(end - begin).count();

			stats.S = Validator<instance_t, solution_t>::getResultValue(instance, res);
			stats.N = Validator<instance_t, solution_t>::getResultWeights(instance, res);
			stats.structure = Validator<instance_t, solution_t>::validateStructure(instance, res);
			stats.fitting = Validator<instance_t, solution_t>::validateFit(instance.limits(), stats.N);

			for (std::pair<std::string, std::ostream&>& output : outputs) {
				size_t pos = output.first.find_first_of('{');
				while (pos != std::string::npos) {
					output.second << output.first.substr(0, pos);
					size_t pos2 = output.first.find_first_of('}');
					std::string keyval = output.first.substr(pos + 1, pos2 - pos - 1);
					if (keyval == "solver name") output.second << Solver::name;
					//else if (keyval == "instance name") output.second << instance.name;
					else if (keyval == "time") output.second << stats.time;
					else if (keyval == "instance") output.second << instance;
					else if (keyval == "result") output.second << res;
					else if (keyval == "instance size") output.second << instance.size();
					else if (keyval == "result value") output.second << stats.S;
					else if (keyval == "limits") for (auto elem : instance.limits()) output.second << elem << " ";
					else if (keyval == "result weights") for (auto elem : stats.N) output.second << elem << " ";
					else if (keyval == "structure valid") output.second << (stats.structure ? "true" : "false");
					else if (keyval == "fitting") output.second << (stats.fitting ? "true" : "false");
					output.first = output.first.substr(pos2 + 1);
					pos = output.first.find_first_of('{');
				}
				output.second << output.first;
			}

			return stats;
		}

		template <typename... Args>
		inline static typename stats<instance_t>::single run(
			const instance_t& instance,
			std::string format,
			std::ostream& stream = std::cout,
			Args... args
		) {
			return run<Args...>(instance, { {format, stream} }, args...);
		}

		template <typename... Args>
		inline static typename stats<instance_t>::avg run(
			const std::vector<instance_t>& instances,
			std::vector<std::pair<std::string, std::ostream&>> outputs,
			std::vector<std::pair<std::string, std::ostream&>> avgOutputs,
			Args... args
		) {
			typename stats<instance_t>::sum resultSumStats;
			typename stats<instance_t>::sum  instanceSumStats;

			// solve all instances
			for (const instance_t& instance : instances) {
				resultSumStats += run<Args...>(instance, outputs, args...);
				instanceSumStats.N += instance.n;
				instanceSumStats.S += instance.wordSet.size();
			}

			// average results
			typename stats<instance_t>::avg avgResultStats = resultSumStats / instances.size();
			typename stats<instance_t>::avg avgInstanceStats = instanceSumStats / instances.size();

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
					else if (keyval == "structure") avgOutput.second << avgResultStats.structure;
					else if (keyval == "fitting") avgOutput.second << avgResultStats.fitting;
					avgOutput.first = avgOutput.first.substr(pos2 + 1);
					pos = avgOutput.first.find_first_of('{');
				}
				avgOutput.second << avgOutput.first;
			}

			return avgResultStats;
		}

		inline static typename stats<instance_t>::avg run(
			const std::vector<instance_t>& instances,
			std::string format,
			std::ostream& stream = std::cout
		) {
			return run(instances, { {format, stream} });
		}
	};
}
