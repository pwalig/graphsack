#pragma once

namespace gs {
	enum class structure {
		none, path, cycle, tree, connected
	};

	enum class weight_treatment {
		ignore, first_only, as_ones, full
	};
}
