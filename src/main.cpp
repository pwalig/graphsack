#include <iostream>

#include "weight_value_vector.hpp"

void main(int argc, char** argv) {
	const const int a = 5;
	std::cout << a << " A\n";

	gs::weight_value_vector<> inst(
		{ 11, 12, 13 },
		{ { 21, { 1, 2, 3}}, {22, { 4, 5, 6 }} }
	);

	std::cout << inst;

	std::cout << gs::weight_value_vector<>("instances/test.txt");
}