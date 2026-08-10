#include "signed_distance.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include <nanobind/ndarray.h>

#include "mujoco_handles.hpp"

namespace {

struct SignedDistanceRecord {
    double dist = 0.0;
    double fromto[6] = {0.0,0.0,0.0,0.0,0.0,0.0};
    double normal[3] = {0.0,0.0,0.0};
    int geom1 = -1;
    int geom2 = -1;
    int body1 = -1;
    int body2 = -1;
    int source = 0;
    int contact_idx = -1;
};

std::uint64_t pair_key(int g1,int g2) {
    const std::uint32_t lo = static_cast<std::uint32_t>(std::min(g1,g2));
    const std::uint32_t hi = static_cast<std::uint32_t>(std::max(g1,g2));
    return (static_cast<std::uint64_t>(lo) << 32) | static_cast<std::uint64_t>(hi);
}

double point_dist(const double* p1,const double* p2) {
    const double dx = p2[0] - p1[0];
    const double dy = p2[1] - p1[1];
    const double dz = p2[2] - p1[2];
    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

bool record_dist_less(const SignedDistanceRecord& a,const SignedDistanceRecord& b) {
    return a.dist < b.dist;
}

template <typename T>
nb::object copy_array(const std::vector<T>& values,std::initializer_list<size_t> shape) {
    auto* buffer = new std::vector<T>(values);
    nb::capsule owner(buffer,[](void* p) noexcept {
        delete static_cast<std::vector<T>*>(p);
    });
    return nb::cast(nb::ndarray<nb::numpy,T>(buffer->data(),shape,owner));
}

void sort_and_truncate(std::vector<SignedDistanceRecord>& records,int topk) {
    std::sort(records.begin(),records.end(),record_dist_less);
    if (topk >= 0 && static_cast<size_t>(topk) < records.size()) {
        records.resize(static_cast<size_t>(topk));
    }
}

void validate_finite_nonnegative(double value,const char* name) {
    if (!std::isfinite(value) || value < 0.0) {
        throw std::runtime_error(std::string(name) + " must be finite and non-negative.");
    }
}

void validate_nonnegative_allow_infinity(double value,const char* name) {
    if (std::isnan(value) || value < 0.0) {
        throw std::runtime_error(
            std::string(name) + " must be non-negative and may be positive infinity."
        );
    }
}

}

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
    ) {
    const mjModel* model = as_model(model_address);
    const mjData* data = as_data(data_address);

    if (geom_pairs.ndim() != 2 || geom_pairs.shape(1) != 2) {
        throw std::runtime_error("geom_pairs must have shape (N, 2).");
    }
    if (geom_pairs.shape(0) > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("geom_pairs is too large.");
    }

    validate_nonnegative_allow_infinity(distmax,"distmax");
    validate_finite_nonnegative(lower_bound_tol,"lower_bound_tol");
    validate_finite_nonnegative(fromto_dist_tol,"fromto_dist_tol");

    const int n_pair = static_cast<int>(geom_pairs.shape(0));
    const int* pair_data = geom_pairs.data();
    if (topk < -1) {
        throw std::runtime_error("topk must be -1 or non-negative.");
    }
    if (topk == 0 || n_pair == 0) {
        nb::dict empty;
        empty["dist"] = copy_array<double>({}, {0});
        empty["fromto"] = copy_array<double>({}, {0,6});
        empty["normal"] = copy_array<double>({}, {0,3});
        empty["geom_pair"] = copy_array<int>({}, {0,2});
        empty["body_pair"] = copy_array<int>({}, {0,2});
        empty["source"] = copy_array<int>({}, {0});
        empty["contact_idx"] = copy_array<int>({}, {0});
        empty["n_result"] = 0;
        return empty;
    }

    for (int pair_idx = 0; pair_idx < n_pair; ++pair_idx) {
        const int g1 = pair_data[2*pair_idx + 0];
        const int g2 = pair_data[2*pair_idx + 1];
        if (g1 < 0 || g1 >= model->ngeom || g2 < 0 || g2 >= model->ngeom) {
            throw std::runtime_error("geom_pairs contains an out-of-range geom index.");
        }
    }

    auto gil_release = std::make_unique<nb::gil_scoped_release>();
    std::unordered_set<std::uint64_t> candidate_keys;
    candidate_keys.reserve(static_cast<size_t>(n_pair)*2);
    for (int pair_idx = 0; pair_idx < n_pair; ++pair_idx) {
        const int g1 = pair_data[2*pair_idx + 0];
        const int g2 = pair_data[2*pair_idx + 1];
        candidate_keys.insert(pair_key(g1,g2));
    }

    std::vector<SignedDistanceRecord> records;
    records.reserve(topk > 0 ? static_cast<size_t>(topk) : 16);

    std::unordered_set<std::uint64_t> contact_keys;
    contact_keys.reserve(static_cast<size_t>(std::max(1,data->ncon))*2);

    for (int c_idx = 0; c_idx < data->ncon; ++c_idx) {
        const mjContact& contact = data->contact[c_idx];
        const int g1 = contact.geom1;
        const int g2 = contact.geom2;
        if (g1 < 0 || g2 < 0) {
            continue;
        }

        const std::uint64_t key = pair_key(g1,g2);
        if (candidate_keys.find(key) == candidate_keys.end()) {
            continue;
        }
        if (contact_keys.find(key) != contact_keys.end()) {
            continue;
        }

        SignedDistanceRecord rec;
        rec.dist = static_cast<double>(contact.dist);
        rec.geom1 = g1;
        rec.geom2 = g2;
        rec.body1 = static_cast<int>(model->geom_bodyid[g1]);
        rec.body2 = static_cast<int>(model->geom_bodyid[g2]);
        rec.source = 1;
        rec.contact_idx = c_idx;
        for (int axis = 0; axis < 3; ++axis) {
            rec.normal[axis] = static_cast<double>(contact.frame[axis]);
            rec.fromto[axis] = (
                static_cast<double>(contact.pos[axis])
                - 0.5*rec.dist*rec.normal[axis]
            );
            rec.fromto[3 + axis] = (
                static_cast<double>(contact.pos[axis])
                + 0.5*rec.dist*rec.normal[axis]
            );
        }
        records.push_back(rec);
        contact_keys.insert(key);
    }

    double kth_best_dist = distmax;
    if (topk >= 0 && static_cast<int>(records.size()) >= topk) {
        sort_and_truncate(records,topk);
        kth_best_dist = std::min(distmax,records.back().dist);
    }

    std::vector<double> lower_bounds(static_cast<size_t>(n_pair));
    for (int pair_idx = 0; pair_idx < n_pair; ++pair_idx) {
        const int g1 = pair_data[2*pair_idx + 0];
        const int g2 = pair_data[2*pair_idx + 1];
        const double dx = data->geom_xpos[3*g2 + 0] - data->geom_xpos[3*g1 + 0];
        const double dy = data->geom_xpos[3*g2 + 1] - data->geom_xpos[3*g1 + 1];
        const double dz = data->geom_xpos[3*g2 + 2] - data->geom_xpos[3*g1 + 2];
        lower_bounds[static_cast<size_t>(pair_idx)] = (
            std::sqrt(dx*dx + dy*dy + dz*dz)
            - model->geom_rbound[g1]
            - model->geom_rbound[g2]
        );
    }

    std::vector<int> pair_order(static_cast<size_t>(n_pair));
    for (int pair_idx = 0; pair_idx < n_pair; ++pair_idx) {
        pair_order[static_cast<size_t>(pair_idx)] = pair_idx;
    }
    if (topk >= 0) {
        std::sort(
            pair_order.begin(),
            pair_order.end(),
            [&](int a,int b) {
                return lower_bounds[static_cast<size_t>(a)] < lower_bounds[static_cast<size_t>(b)];
            }
        );
    }

    double fromto[6] = {0.0,0.0,0.0,0.0,0.0,0.0};
    for (int pair_idx : pair_order) {
        if (
            topk >= 0 &&
            static_cast<int>(records.size()) >= topk &&
            lower_bounds[static_cast<size_t>(pair_idx)] > kth_best_dist
        ) {
            break;
        }

        const int g1 = pair_data[2*pair_idx + 0];
        const int g2 = pair_data[2*pair_idx + 1];
        if (contact_keys.find(pair_key(g1,g2)) != contact_keys.end()) {
            continue;
        }

        for (double& val : fromto) {
            val = 0.0;
        }
        const double query_distmax = topk >= 0 ? kth_best_dist : distmax;
        const double dist = mj_geomDistance(model,data,g1,g2,query_distmax,fromto);

        if (sanity_check) {
            if (dist < lower_bounds[static_cast<size_t>(pair_idx)] - lower_bound_tol) {
                continue;
            }
            bool finite = std::isfinite(dist);
            for (double val : fromto) {
                finite = finite && std::isfinite(val);
            }
            if (!finite) {
                continue;
            }
            const double fromto_dist = point_dist(fromto,fromto+3);
            if (dist <= 1e-12 && fromto_dist <= 1e-12) {
                continue;
            }
            if (std::fabs(fromto_dist - dist) > fromto_dist_tol) {
                continue;
            }
        }

        SignedDistanceRecord rec;
        rec.dist = dist;
        rec.geom1 = g1;
        rec.geom2 = g2;
        rec.body1 = static_cast<int>(model->geom_bodyid[g1]);
        rec.body2 = static_cast<int>(model->geom_bodyid[g2]);
        rec.source = 0;
        rec.contact_idx = -1;
        std::memcpy(rec.fromto,fromto,sizeof(double)*6);
        records.push_back(rec);

        if (topk >= 0) {
            sort_and_truncate(records,topk);
            if (static_cast<int>(records.size()) >= topk) {
                kth_best_dist = std::min(distmax,records.back().dist);
            }
        }
    }

    if (sort_output || topk >= 0) {
        sort_and_truncate(records,topk);
    }

    gil_release.reset();
    const size_t n_result = records.size();
    std::vector<double> dist_values(n_result);
    std::vector<double> fromto_values(n_result*6);
    std::vector<double> normal_values(n_result*3);
    std::vector<int> geom_pair_values(n_result*2);
    std::vector<int> body_pair_values(n_result*2);
    std::vector<int> source_values(n_result);
    std::vector<int> contact_idx_values(n_result);

    for (size_t idx = 0; idx < n_result; ++idx) {
        const SignedDistanceRecord& rec = records[idx];
        dist_values[idx] = rec.dist;
        source_values[idx] = rec.source;
        contact_idx_values[idx] = rec.contact_idx;
        geom_pair_values[2*idx + 0] = rec.geom1;
        geom_pair_values[2*idx + 1] = rec.geom2;
        body_pair_values[2*idx + 0] = rec.body1;
        body_pair_values[2*idx + 1] = rec.body2;
        for (int axis = 0; axis < 6; ++axis) {
            fromto_values[6*idx + axis] = rec.fromto[axis];
        }
        for (int axis = 0; axis < 3; ++axis) {
            normal_values[3*idx + axis] = rec.normal[axis];
        }
    }

    nb::dict out;
    out["dist"] = copy_array<double>(dist_values,{n_result});
    out["fromto"] = copy_array<double>(fromto_values,{n_result,6});
    out["normal"] = copy_array<double>(normal_values,{n_result,3});
    out["geom_pair"] = copy_array<int>(geom_pair_values,{n_result,2});
    out["body_pair"] = copy_array<int>(body_pair_values,{n_result,2});
    out["source"] = copy_array<int>(source_values,{n_result});
    out["contact_idx"] = copy_array<int>(contact_idx_values,{n_result});
    out["n_result"] = static_cast<int>(n_result);
    return out;
}
