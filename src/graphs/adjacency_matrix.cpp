#include "adjacency_matrix.hpp"

const gs::graphs::representation gs::graphs::adjacency_matrix::representation = graphs::representation::adjacency;

gs::graphs::adjacency_matrix::adjacency_matrix(std::initializer_list<std::initializer_list<bool>> init)
	: storage(init.size()* init.begin()->size()) {
	size_type i = 0;
	for (const auto& w : init) {
		assert(w.size() == init.size());
		for (bool c : w) {
			storage[i] = c;
			++i;
		}
	}
}

gs::graphs::adjacency_matrix::size_type gs::graphs::adjacency_matrix::ones()
{
	size_type res = 0;
	size_type n = size();
	for (size_type i = 0; i < n; ++i) {
		for (size_type j = 0; j < n; ++j) {
			if (at(i, j)) ++res;
		}
	}
	return res;
}

gs::graphs::adjacency_matrix gs::graphs::adjacency_matrix::from_graph6(const std::string& buff) {
	char bit = 32;
	typename std::string::size_type poz = 0;
	size_type n = 0;
	if (buff[0] == '~') {
		if (buff[1] == '~') {
			poz = 2;
			while (poz < 8) {
				n <<= 6;
				n += buff[poz++] - 63;
			}
		} else {
			poz = 1;
			while (poz < 4) {
				n <<= 6;
				n += buff[poz++] - 63;
			}
		}
	}
	else {
		n = buff[poz++] - 63;
	}
	adjacency_matrix res(n);
	for (size_type i = 1; i < n; i++) {
		for (size_type j = 0; j < i; j++)
		{
			if (bit == 0) { bit = 32;  poz++; }
			if ((buff[poz] - 63) & bit)
				res.at(i, j) = res.at(j, i) = true;
			else
				res.at(i, j) = res.at(j, i) = false;
			bit >>= 1;
		}
	}
	return res;
}

std::string gs::graphs::adjacency_matrix::graph6() {
	size_type n = size();
	std::string result;

	if (n <= 62) {
		result += (char)(n + 63);
	}
	else if (n <= 258047) {
		result += '~';
		for (size_type i = 1 << 12; i > 0; i >>= 6) {
			result += (char)(((n / i) % 64) + 63);
		}
	}
	else if (n <= 68719476735) {
		result += "~~";
		for (size_type i = 1 << 12; i > 0; i >>= 6) {
			result += (char)(((n / i) % 64) + 63);
		}
	}
	else throw std::logic_error("graph size is to large for graph6 representation");

	std::vector<bool> bitVector;
	for (size_type j = 1; j < n; ++j) {
		for (size_type i = 0; i < j; ++i) {
			bitVector.push_back(at(i, j));
		}
	}

	// Make the length a multiple of 6 by padding with 0s
	size_type len = bitVector.size();
	while (len % 6 != 0) {
		bitVector.push_back(false);
		len++;
	}

	// Convert every 6 bits to a base64 character
	for (size_type i = 0; i < len; i += 6) {
		unsigned char val = 0;
		for (size_type j = 0; j < 6; ++j) {
			val = val * 2 + (bitVector[i + j] ? 1 : 0);
		}
		result += (char)(val + 63);
	}

	return result;
}

#ifndef NDEBUG
#include <fstream>
void gs::graphs::adjacency_matrix::test::all()
{
	// initializer
	initializer_list_constructable();

	// gnp
	std::random_device randomDevice;
	std::knuth_b gen(randomDevice());
	from_gnp(24, 0.5, gen, false, false);
	from_gnp(16, 1.0, gen, true, false);
	from_gnp(8, 0.0, gen, false, true);

	// g6
	g6_converter_consistency("SeaLsGRWR{TjcoJYK`hqCYRz@FfnMuhSG");
	g6_converter_consistency_from_file("tests/adjacency_matrix/1.g6");
}

void gs::graphs::adjacency_matrix::test::initializer_list_constructable()
{
	graphs::adjacency_matrix am({
		{true, true, false},
		{true, true, false},
		{true, true, false}
		});

	assert(am.size() == 3);
	assert(am[0][0] == true);
	assert(am.at(0, 1) == true);
	assert(am.at(0)[2] == false);
	assert(am.at(1).at(0) == true);
	assert(am[2].at(2) == false);
}

void gs::graphs::adjacency_matrix::test::g6_converter_consistency_from_file(const std::string& filename)
{
	std::ifstream fin(filename);
	if (fin.is_open()) {
		std::string graph6_string;
		fin >> graph6_string;
		g6_converter_consistency(graph6_string);
	}
}

void gs::graphs::adjacency_matrix::test::g6_converter_consistency(const std::string& graph6)
{
	adjacency_matrix am = graphs::adjacency_matrix::from_graph6(graph6);
	assert(am.graph6() == graph6);
}
#endif
