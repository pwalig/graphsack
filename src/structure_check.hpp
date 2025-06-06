#pragma once
#include <vector>
#include <stack>
#include <cassert>

namespace gs {
	template <typename instance_t, typename solution_t, typename indexT = typename instance_t::index_type>
	inline bool is_cycle_possible_DFS(
		const instance_t& instance,
		const solution_t& selected,
		std::vector<bool>& visited,
		const std::vector<typename instance_t::weight_type>& _remaining_space,
		indexT current, indexT start
	) {
		for (indexT next : instance.nexts(current)) {
			if (visited[next]) continue; // next item has to be new (not visited yet)

			// fit check
			std::vector<typename instance_t::weight_type> new_remaining_space(_remaining_space);
			indexT j = 0;
			for (; j < instance.dim(); ++j) {
				if (new_remaining_space[j] >= instance.weight(next, j))
					new_remaining_space[j] -= instance.weight(next, j);
				else break;
			}
			if (j != instance.dim()) continue;

			visited[next] = true;

			if (instance.has_connection_to(next, start)) {
				// completness check
				indexT i = 0;
				for (; i < instance.size(); ++i) {
					if (selected.has(i) && !visited[i]) break;
				}
				if (i == instance.size()) return true; // cycle found
			}

			// DFS call
			if (is_cycle_possible_DFS<instance_t, solution_t, indexT>(
				instance, selected, visited, new_remaining_space, next, start
			)) return true; // cycle found later

			visited[next] = false;
		}
		return false; // cycle not found
	}

	template <typename instance_t, typename solution_t, typename indexT = typename instance_t::index_type>
	inline bool is_cycle_possible(
		const instance_t& instance,
		const solution_t& selected
	) {
		assert(selected.size() == instance.size());
		std::vector<bool> visited(selected.size(), false);
		std::vector<typename instance_t::weight_type> _remaining_space(instance.dim());

		for (indexT i = 0; i < instance.size(); ++i) {

			// fit check
			typename instance_t::size_type wid = 0;
			for (; wid < instance.dim(); ++wid) {
				if (instance.limit(wid) >= instance.weight(i, wid))
					_remaining_space[wid] = instance.limit(wid) - instance.weight(i, wid);
				else break;
			}
			if (wid != instance.dim()) continue;

			visited[i] = true;

			if (instance.has_connection_to(i, i)) { // is it a cycle (closed path)
				// completness check
				indexT j = 0;
				for (; j < instance.size(); ++j) {
					if (selected.has(j) && j != i) break;
				}
				if (j == instance.size()) return true; // cycle found
			}

			// DFS call
			if (is_cycle_possible_DFS<instance_t, solution_t, indexT>(
				instance, selected, visited, _remaining_space, i, i
			)) return true; // cycle found somewhere

			visited[i] = false;
		}
		return false;
	}

	template <typename instance_t, typename solution_t, typename indexT = typename instance_t::index_type>
	inline bool is_path_possible_DFS(
		const instance_t& instance,
		const solution_t& selected, 
		std::vector<bool>& visited,
		const std::vector<typename instance_t::weight_type>& _remaining_space,
		const indexT& current
	) {
		for (indexT next : instance.nexts(current)) {
			if (visited[next]) continue; // next item has to be new (not visited yet)

			// fit check
			std::vector<typename instance_t::weight_type> new_remaining_space(_remaining_space);
			indexT j = 0;
			for (; j < instance.dim(); ++j) {
				if (new_remaining_space[j] >= instance.weight(next, j))
					new_remaining_space[j] -= instance.weight(next, j);
				else break;
			}
			if (j != instance.dim()) continue;

			visited[next] = true;

			// completness check
			indexT i = 0;
			for (; i < instance.size(); ++i) {
				if (selected.has(i) && !visited[i]) break;
			}
			if (i == instance.size()) return true; // cycle found

			// DFS call
			if (is_path_possible_DFS<instance_t, solution_t, indexT>(
				instance, selected, visited, new_remaining_space, next
			)) return true; // path found later

			visited[next] = false;
		}
		return false; // cycle not found
	}

	template <typename instance_t, typename solution_t, typename indexT = typename instance_t::index_type>
	inline bool is_path_possible(const instance_t& instance, const solution_t& selected) {
		assert(selected.size() == instance.size());
		std::vector<bool> visited(selected.size(), false);
		std::vector<typename instance_t::weight_type> _remaining_space(instance.dim());

		for (indexT i = 0; i < instance.size(); ++i) {

			// fit check
			typename instance_t::size_type wid = 0;
			for (; wid < instance.dim(); ++wid) {
				if (instance.limit(wid) >= instance.weight(i, wid))
					_remaining_space[wid] = instance.limit(wid) - instance.weight(i, wid);
				else break;
			}
			if (wid != instance.dim()) continue;

			visited[i] = true;

			// completness check
			indexT j = 0;
			for (; j < instance.size(); ++j) {
				if (selected.has(j) && j != i) break;
			}
			if (j == instance.size()) return true; // cycle found

			// DFS call
			if (is_path_possible_DFS<instance_t, solution_t, indexT>(
				instance, selected, visited, _remaining_space, i)
			) return true; // cycle found somewhere

			visited[i] = false;
		}
		return false;
	}

	template <typename instance_t, typename solution_t, typename indexT = typename instance_t::index_type>
	inline bool is_path_DFS(
		const instance_t& problem,
		const solution_t& selected,
		std::vector<bool>& visited,
		indexT current, indexT length, indexT depth
	) {
		for (auto next : problem.nexts(current)) {
			if (selected.has(next) && !visited[next]){ // next item has to be selected and new
				visited[next] = true;
				if (depth == length) return true; // path found
				if (depth > length) return false; // path would have to be to long
				if (is_path_DFS<instance_t, solution_t, indexT>(problem, selected, visited, next, length, depth + 1)) return true; // path found later
				visited[next] = false;
			}
		}
		return false; // path not found
	}

	template <typename instance_t, typename solution_t, typename indexT = typename instance_t::index_type>
	inline bool is_path(const instance_t& instance, const solution_t& selected) {
		assert(selected.size() == instance.size());

		// calculate whats the length of the path
		indexT length = 0;
		for (indexT i = 0; i < selected.size(); ++i)
			if (selected.has(i)) ++length;
		
		if (length <= 1) return true;
		
		// check from each starting point
		std::vector<bool> visited(selected.size(), false);
		for (indexT i = 0; i < selected.size(); ++i) {
			if (selected.has(i)){
				visited[i] = true;
				if (is_path_DFS<instance_t, solution_t, indexT>(instance, selected, visited, i, length, 2)) return true; // path found somewhere
				visited[i] = false;
			}
		}
		return false;
	}

	template <typename instance_t, typename solution_t, typename indexT = typename instance_t::index_type>
	inline bool is_cycle_DFS(
		const instance_t& instance,
		const solution_t& selected,
		std::vector<bool>& visited,
		indexT current, indexT start, indexT length, indexT depth
	) {
		for (indexT next : instance.nexts(current)) {
			if (selected.has(next) && !visited[next]){ // next item has to be selected and new
				visited[next] = true;
				if (depth == length && instance.has_connection_to(next, start)) return true; // cycle found
				if (depth > length) return false; // cycle would have to be to long
				if (is_cycle_DFS<instance_t, solution_t, indexT>(instance, selected, visited, next, start, length, depth + 1)) return true; // cycle found later
				visited[next] = false;
			}
		}
		return false; // cycle not found
	}

	template <typename instance_t, typename solution_t, typename indexT = typename instance_t::index_type>
	inline bool is_cycle_recursive(
		const instance_t& instance,
		const solution_t& selected
	) {
		assert(selected.size() == instance.size());

		// calculate whats the length of the cycle
		indexT length = 0;
		for (indexT i = 0; i < selected.size(); ++i)
			if (selected.has(i)) ++length;
		
		if (length == 0) return true;
		
		// check from each starting point
		std::vector<bool> visited(selected.size(), false);
		for (indexT i = 0; i < selected.size(); ++i) {
			if (selected.has(i)){
				if (length == 1) return instance.has_connection_to(i, i);
				visited[i] = true;
				if (is_cycle_DFS<instance_t, solution_t, indexT>(instance, selected, visited, i, i, length, 2)) return true; // cycle found somewhere
				visited[i] = false;
			}
		}
		return false;
	}

	template <typename instance_t, typename solution_t, typename indexT = typename instance_t::index_type>
	inline bool is_cycle_iterative(
		const instance_t& instance,
		const solution_t& selected
	) {
		// calculate whats the length of the cycle
		uint32_t length = 0;
		for (indexT i = 0; i < instance.size(); ++i)
			if (selected.has(i)) length++;
			
		if (length == 0) return true;

		struct entry {
			indexT item;
			indexT lastViewed;
		};
		std::stack<entry> stack;

		// check from each starting point
		solution_t visited(instance.size());
		uint32_t visitedCount = 0;
		for (indexT start = 0; start < instance.size(); ++start) {
			if (selected.has(start)) {
				indexT prev = start;
				while (true) {

					// visiting for the first time
					if (!visited.has(prev)) {
						visited.add(prev);
						++visitedCount;
						if (visitedCount == length && instance.has_connection_to(prev, start)) {
							return true;
						}
						if (visitedCount > length) stack.push({ prev, static_cast<indexT>(instance.size()) });
						else stack.push({ prev, 0 });
					}

					assert(prev == stack.top().item);
					indexT next = stack.top().lastViewed + 1;
					stack.pop();
					for (; next < instance.size(); ++next) {
						if (!instance.has_connection_to(prev, next)) continue;
						if (selected.has(next) && !visited.has(next)) {
							stack.push({ prev, next });
							prev = next;
							break;
						}
					}

					// visited all nexts
					if (next == instance.size()) {
						visited.remove(prev);
						--visitedCount;
						if (stack.empty()) break;
						prev = stack.top().item;
					}
				}
			}
		}
		return false;
	}
}
