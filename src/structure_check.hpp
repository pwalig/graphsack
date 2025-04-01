#pragma once
#include <vector>
#include <cassert>

namespace gs {
    template <typename instance_t, typename solution_t, typename indexT = size_t>
    bool is_cycle_possible_DFS(
        const instance_t& instance,
        const solution_t& selected,
        std::vector<bool>& visited,
        const std::vector<typename instance_t::weight_type>& _remaining_space,
        indexT current, indexT start
    ) {
        for (indexT next : instance.nexts(current)) {
            if (visited[next]) continue; // next item has to be new (not visited yet)

            bool fit = true;
            std::vector<typename instance_t::weight_type> new_remaining_space(_remaining_space);
            for (int j = 0; j < new_remaining_space.size(); ++j) {
                if (new_remaining_space[j] >= instance.items[next].weights[j]) new_remaining_space[j] -= instance.items[next].weights[j];
                else { fit = false; break; }
            }
            if (!fit) continue;

            visited[next] = true;
            if (instance.connection(next, start)) { // is it a cycle (closed path)
                bool _found = true; // found some cycle lets check if it has all selected vertices
                for (indexT i = 0; i < selected.size(); ++i) {
                    if (selected[i] && !visited[i]) {
                        _found = false;
                        break;
                    }
                }
                if (_found) return true; // it has - cycle found
            }
            if (is_cycle_possible_DFS<instance_t, solution_t, indexT>(
                instance, selected, visited, new_remaining_space, next, start)
			) return true; // cycle found later
            visited[next] = false;
        }
        return false; // cycle not found
    }

    template <typename instance_t, typename solution_t, typename indexT = size_t>
    bool is_cycle_possible(const instance_t& instance, const solution_t& selected) {
        assert(selected.size() == instance.items.size());
        std::vector<bool> visited(selected.size(), false);
        for (int i = 0; i < selected.size(); ++i) {

            bool fit = true;
            std::vector<typename instance_t::weight_type> _remaining_space = instance.knapsack_sizes;
            for (int j = 0; j < _remaining_space.size(); ++j) {
                if (_remaining_space[j] >= instance.items[i].weights[j]) _remaining_space[j] -= instance.items[i].weights[j];
                else { fit = false; break; }
            }
            if (!fit) continue;

            visited[i] = true;
            if (instance.connection(i, i)) { // is it a cycle (closed path)
                bool _found = true; // found some cycle lets check if it has all selected vertices
                for (int j = 0; j < this->selected.size(); ++j) {
                    if (this->selected[j] && j != i) {
                        _found = false;
                        break;
                    }
                }
                if (_found) return true; // it has - cycle found
            }
            if (is_cycle_possible_DFS<instance_t, solution_t, indexT>(
                instance, selected, visited, _remaining_space, i, i)
            ) return true; // cycle found somewhere
            visited[i] = false;
        }
        return false;
    }

    template <typename instance_t, typename solution_t, typename indexT = size_t>
    bool is_path_possible_DFS(
        const instance_t& instance,
        const solution_t& selected, 
        std::vector<bool>& visited,
        std::vector<typename instance_t::weight_type> _remaining_space,
        const indexT& current
    ) {
        for (int next : instance.nexts(current)) {
            if (visited[next]) continue; // next item has to be new (not visited yet)

            bool fit = true;
            for (indexT j = 0; j < _remaining_space.size(); ++j) {
                if (_remaining_space[j] >= instance.weight(next, j)) _remaining_space[j] -= instance.weight(next, j);
                else { fit = false; break; }
            }
            if (!fit) { continue; }

            visited[next] = true;
            // found some path lets check if it has all selected vertices
            bool _found = true;
            for (indexT i = 0; i < selected.size(); ++i) {
                if (selected[i] && !visited[i]) {
                    _found = false;
                    break;
                }
            }
            if (_found) { return true; } // it has - path found
            if (is_path_possible_DFS<instance_t, solution_t, indexT>(
                instance, selected, visited, _remaining_space, next)
			) return true; // path found later
            visited[next] = false;
        }
        return false; // cycle not found
    }

    template <typename instance_t, typename solution_t, typename indexT = size_t>
    bool is_path_possible(const instance_t& instance, const solution_t& selected) {
        assert(selected.size() == instance.size());
        std::vector<bool> visited(selected.size(), false);

        for (indexT i = 0; i < instance.size(); ++i) {

            bool fit = true;
            std::vector<typename instance_t::weight_type> _remaining_space(instance.dim());
            memcpy(_remaining_space.data(), instance.limits().data(), instance.dim() * sizeof(typename instance_t::weight_type));
            for (int j = 0; j < _remaining_space.size(); ++j) {
                if (_remaining_space[j] >= instance.weight(i, j)) _remaining_space[j] -= instance.weight(i, j);
                else { fit = false; break; }
            }
            if (!fit) continue;

            visited[i] = true;
            // found some path lets check if it has all selected vertices
            bool _found = true;
            for (indexT j = 0; j < selected.size(); ++j) {
                if (selected[j] && j != i) {
                    _found = false;
                    break;
                }
            }
            if (_found) { return true; } // it has - path found
            if (is_path_possible_DFS<instance_t, solution_t, indexT>(
                instance, selected, visited, _remaining_space, i)
			) return true; // cycle found somewhere
            visited[i] = false;
        }
        return false;
    }
}
