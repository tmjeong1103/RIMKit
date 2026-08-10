#include "signed_distance_bindings.hpp"

#include "signed_distance.hpp"

void bind_signed_distance(nb::module_& m) {
    m.def(
        "signed_distance_arrays",
        &signed_distance_arrays,
        "Compute top-k signed geom distances as compact NumPy arrays."
    );
}
