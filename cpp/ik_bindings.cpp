#include "ik_bindings.hpp"

#include "ik.hpp"

void bind_ik(nb::module_& m) {
    m.def(
        "solve_body_position_ik",
        &solve_body_position_ik,
        "Solve BodyPositionIKSolverRevPriBase loop with native code."
    );
    m.def(
        "solve_planar_base_ik",
        &solve_planar_base_ik,
        "Solve PlanarBaseIKSolver loop with native code."
    );
}
