#pragma once

#include <cstdint>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

nb::dict signed_distance_arrays(
    std::uintptr_t model_address,
    std::uintptr_t data_address,
    nb::ndarray<nb::numpy,const int,nb::ndim<2>,nb::c_contig> geom_pairs,
    double distmax,
    int topk,
    bool sort_output,
    bool sanity_check,
    double lower_bound_tol,
    double fromto_dist_tol
);
