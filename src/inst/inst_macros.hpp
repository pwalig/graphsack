#pragma once
#include "../weight_treatment.hpp"
#include "../structure.hpp"

#define GS_INST_WEIGHT_TREATMENT_MEMBER gs::weight_treatment weightTreatment;
#define GS_INST_STRUCTURE_MEMBER gs::structure structureToFind;

#define GS_INST_WEIGHT_TREATMENT_AND_STRUCTURE_MEMBERS \
GS_INST_WEIGHT_TREATMENT_MEMBER GS_INST_STRUCTURE_MEMBER


#define GS_INST_WEIGHT_TREATMENT_ACCESS \
inline gs::weight_treatment& weight_treatment() { return weightTreatment; } \
inline const gs::weight_treatment& weight_treatment() const { return weightTreatment; }

#define GS_INST_STRUCTURE_ACCESS \
inline gs::structure& structure_to_find() { return structureToFind; } \
inline const gs::structure& structure_to_find() const { return structureToFind; }

#define GS_INST_WEIGHT_TREATMENT_AND_STRUCTURE_ACCESS \
GS_INST_WEIGHT_TREATMENT_ACCESS GS_INST_STRUCTURE_ACCESS

#define GS_INST_WEIGHT_TREATMENT_FIRST_ONLY_ASSERT assert(weight_treatment() == gs::weight_treatment::first_only);

#define GS_INST_WEIGHT_TREATMENT_FIRST_ONLY_ACCESS \
inline weight_type& limit() { \
	GS_INST_WEIGHT_TREATMENT_FIRST_ONLY_ASSERT \
	return limit(0); \
} \
inline const weight_type& limit() const { \
	GS_INST_WEIGHT_TREATMENT_FIRST_ONLY_ASSERT \
	return limit(0); \
} \
inline weight_type& weight(size_type itemId) { \
	GS_INST_WEIGHT_TREATMENT_FIRST_ONLY_ASSERT \
	return weight(itemId, 0); \
} \
inline const weight_type& weight(size_type itemId) const { \
	GS_INST_WEIGHT_TREATMENT_FIRST_ONLY_ASSERT \
	return weight(itemId, 0); \
}

#define GS_INST_WEIGHT_TREATMENT_AND_STRUCTURE_ACCESS_ALL \
GS_INST_WEIGHT_TREATMENT_AND_STRUCTURE_ACCESS GS_INST_WEIGHT_TREATMENT_FIRST_ONLY_ACCESS
