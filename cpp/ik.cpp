#include "ik.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <nanobind/ndarray.h>

#include "mujoco_handles.hpp"

namespace {

template <typename T>
nb::object copy_array(const std::vector<T>& values,std::initializer_list<size_t> shape) {
    auto* buffer = new std::vector<T>(values);
    nb::capsule owner(buffer,[](void* p) noexcept {
        delete static_cast<std::vector<T>*>(p);
    });
    return nb::cast(nb::ndarray<nb::numpy,T>(buffer->data(),shape,owner));
}

bool all_finite(const std::vector<double>& values) {
    for (double value : values) {
        if (!std::isfinite(value)) {
            return false;
        }
    }
    return true;
}

double max_abs(const std::vector<double>& values) {
    double out = 0.0;
    for (double value : values) {
        out = std::max(out,std::fabs(value));
    }
    return out;
}

double norm2(const std::vector<double>& values) {
    double sum = 0.0;
    for (double value : values) {
        sum += value*value;
    }
    return std::sqrt(sum);
}

double sign_value(double value) {
    if (value > 0.0) {
        return 1.0;
    }
    if (value < 0.0) {
        return -1.0;
    }
    return 0.0;
}

void trim_scale_in_place(std::vector<double>& values,double th) {
    const double scale = max_abs(values);
    if (scale > th && scale > 0.0) {
        const double ratio = th/scale;
        for (double& value : values) {
            value *= ratio;
        }
    }
}

double eval_quartic(double x,const double* coef,double x0,double y0) {
    const double xb = x - x0;
    return (
        y0 + coef[0] + coef[1]*xb + coef[2]*xb*xb +
        coef[3]*xb*xb*xb + coef[4]*xb*xb*xb*xb
    );
}

double eval_quartic_derivative(double x,const double* coef,double x0) {
    const double xb = x - x0;
    return coef[1] + 2.0*coef[2]*xb + 3.0*coef[3]*xb*xb + 4.0*coef[4]*xb*xb*xb;
}

std::vector<double> solve_linear(std::vector<double> A,std::vector<double> b,int n) {
    std::vector<double> x(static_cast<size_t>(n),0.0);
    if (n <= 0) {
        return x;
    }

    for (int col = 0; col < n; ++col) {
        int pivot = col;
        double pivot_abs = std::fabs(A[static_cast<size_t>(col*n + col)]);
        for (int row = col + 1; row < n; ++row) {
            const double value_abs = std::fabs(A[static_cast<size_t>(row*n + col)]);
            if (value_abs > pivot_abs) {
                pivot = row;
                pivot_abs = value_abs;
            }
        }
        if (!std::isfinite(pivot_abs) || pivot_abs < 1e-18) {
            return x;
        }
        if (pivot != col) {
            for (int k = col; k < n; ++k) {
                std::swap(A[static_cast<size_t>(col*n + k)],A[static_cast<size_t>(pivot*n + k)]);
            }
            std::swap(b[static_cast<size_t>(col)],b[static_cast<size_t>(pivot)]);
        }

        const double diag = A[static_cast<size_t>(col*n + col)];
        for (int row = col + 1; row < n; ++row) {
            const double factor = A[static_cast<size_t>(row*n + col)]/diag;
            A[static_cast<size_t>(row*n + col)] = 0.0;
            for (int k = col + 1; k < n; ++k) {
                A[static_cast<size_t>(row*n + k)] -= factor*A[static_cast<size_t>(col*n + k)];
            }
            b[static_cast<size_t>(row)] -= factor*b[static_cast<size_t>(col)];
        }
    }

    for (int row = n - 1; row >= 0; --row) {
        double rhs = b[static_cast<size_t>(row)];
        for (int col = row + 1; col < n; ++col) {
            rhs -= A[static_cast<size_t>(row*n + col)]*x[static_cast<size_t>(col)];
        }
        const double diag = A[static_cast<size_t>(row*n + row)];
        if (!std::isfinite(diag) || std::fabs(diag) < 1e-18) {
            return std::vector<double>(static_cast<size_t>(n),0.0);
        }
        x[static_cast<size_t>(row)] = rhs/diag;
    }

    if (!all_finite(x)) {
        return std::vector<double>(static_cast<size_t>(n),0.0);
    }
    return x;
}

std::vector<double> solve_dls(
        const std::vector<double>& J,
        int m,
        int n,
        const std::vector<double>& e,
        const std::vector<double>& lam
    ) {
    std::vector<double> out(static_cast<size_t>(n),0.0);
    if (m <= 0 || n <= 0) {
        return out;
    }
    if (!all_finite(J) || !all_finite(e) || !all_finite(lam)) {
        return out;
    }

    const double s = max_abs(J);
    if (!std::isfinite(s) || s <= 0.0 || s < 1e-12) {
        return out;
    }

    std::vector<double> H(static_cast<size_t>(n*n),0.0);
    std::vector<double> rhs(static_cast<size_t>(n),0.0);
    const double inv_s = 1.0/s;
    for (int row = 0; row < m; ++row) {
        const double es = e[static_cast<size_t>(row)]*inv_s;
        for (int a = 0; a < n; ++a) {
            const double Ja = J[static_cast<size_t>(row*n + a)]*inv_s;
            rhs[static_cast<size_t>(a)] += Ja*es;
            for (int b = 0; b < n; ++b) {
                const double Jb = J[static_cast<size_t>(row*n + b)]*inv_s;
                H[static_cast<size_t>(a*n + b)] += Ja*Jb;
            }
        }
    }
    for (int a = 0; a < n; ++a) {
        double lam_s = lam[static_cast<size_t>(a)]/(s*s);
        if (!std::isfinite(lam_s) || lam_s < 1e-12) {
            lam_s = 1e-12;
        }
        H[static_cast<size_t>(a*n + a)] += lam_s;
    }
    return solve_linear(H,rhs,n);
}

std::vector<double> project_dq_null_damped(
        const std::vector<double>& J,
        int m,
        int n,
        const std::vector<double>& dq_null,
        const std::vector<double>& lam
    ) {
    std::vector<double> out(static_cast<size_t>(n),0.0);
    if (m <= 0 || n <= 0) {
        return out;
    }
    if (!all_finite(J) || !all_finite(dq_null) || !all_finite(lam)) {
        return out;
    }

    const double s = max_abs(J);
    if (!std::isfinite(s) || s <= 0.0 || s < 1e-12) {
        return dq_null;
    }

    std::vector<double> H(static_cast<size_t>(n*n),0.0);
    std::vector<double> rhs(static_cast<size_t>(n),0.0);
    const double inv_s = 1.0/s;
    for (int row = 0; row < m; ++row) {
        double jdq = 0.0;
        for (int a = 0; a < n; ++a) {
            jdq += J[static_cast<size_t>(row*n + a)]*inv_s*dq_null[static_cast<size_t>(a)];
        }
        for (int a = 0; a < n; ++a) {
            const double Ja = J[static_cast<size_t>(row*n + a)]*inv_s;
            rhs[static_cast<size_t>(a)] += Ja*jdq;
            for (int b = 0; b < n; ++b) {
                const double Jb = J[static_cast<size_t>(row*n + b)]*inv_s;
                H[static_cast<size_t>(a*n + b)] += Ja*Jb;
            }
        }
    }
    for (int a = 0; a < n; ++a) {
        double lam_s = lam[static_cast<size_t>(a)]/(s*s);
        if (!std::isfinite(lam_s) || lam_s < 1e-12) {
            lam_s = 1e-12;
        }
        H[static_cast<size_t>(a*n + a)] += lam_s;
    }

    const std::vector<double> corr = solve_linear(H,rhs,n);
    for (int a = 0; a < n; ++a) {
        out[static_cast<size_t>(a)] = dq_null[static_cast<size_t>(a)] - corr[static_cast<size_t>(a)];
    }
    if (!all_finite(out)) {
        return std::vector<double>(static_cast<size_t>(n),0.0);
    }
    return out;
}

void mask_task_relevant(
        const std::vector<double>& J,
        int m,
        int n,
        std::vector<double>& dq_null
    ) {
    std::vector<double> col_norm(static_cast<size_t>(n),0.0);
    for (int col = 0; col < n; ++col) {
        double sum = 0.0;
        for (int row = 0; row < m; ++row) {
            const double value = J[static_cast<size_t>(row*n + col)];
            sum += value*value;
        }
        col_norm[static_cast<size_t>(col)] = std::sqrt(sum);
    }
    const double max_norm = max_abs(col_norm);
    const double th = std::max(1e-12,1e-6*max_norm);
    for (int col = 0; col < n; ++col) {
        if (col_norm[static_cast<size_t>(col)] <= th) {
            dq_null[static_cast<size_t>(col)] = 0.0;
        }
    }
}

void point_world_from_local(
        const mjData* data,
        int body_id,
        const double* p_local,
        double* p_world
    ) {
    const double* R = data->xmat + 9*body_id;
    const double* p = data->xpos + 3*body_id;
    for (int r = 0; r < 3; ++r) {
        p_world[r] = p[r];
        for (int c = 0; c < 3; ++c) {
            p_world[r] += R[3*r + c]*p_local[c];
        }
    }
}

void build_weighted_Je(
        const mjModel* model,
        mjData* data,
        int n_target,
        const int* body_idxs,
        const double* p_local,
        const double* p_trgt,
        const double* weights,
        std::vector<double>& J,
        std::vector<double>& e,
        std::vector<double>& p_err
    ) {
    const int nv = model->nv;
    const int m = 3*n_target;
    J.assign(static_cast<size_t>(m*nv),0.0);
    e.assign(static_cast<size_t>(m),0.0);
    p_err.assign(static_cast<size_t>(m),0.0);

    std::vector<double> jacp(static_cast<size_t>(3*nv),0.0);
    std::vector<double> jacr(static_cast<size_t>(3*nv),0.0);
    double p_world[3] = {0.0,0.0,0.0};

    for (int i = 0; i < n_target; ++i) {
        const int body_id = body_idxs[i];
        point_world_from_local(data,body_id,p_local + 3*i,p_world);

        std::fill(jacp.begin(),jacp.end(),0.0);
        std::fill(jacr.begin(),jacr.end(),0.0);
        mj_jac(model,data,jacp.data(),jacr.data(),p_world,body_id);

        const double w = weights[i];
        const double ws = std::sqrt(w);
        for (int axis = 0; axis < 3; ++axis) {
            const int row = 3*i + axis;
            const double err = p_trgt[3*i + axis] - p_world[axis];
            p_err[static_cast<size_t>(row)] = err;
            e[static_cast<size_t>(row)] = ws*err;
            for (int col = 0; col < nv; ++col) {
                J[static_cast<size_t>(row*nv + col)] = ws*jacp[static_cast<size_t>(axis*nv + col)];
            }
        }
    }
}

std::vector<double> gather_q_rev(
        const mjData* data,
        const int* qpos_idxs,
        int n_rev
    ) {
    std::vector<double> q(static_cast<size_t>(n_rev),0.0);
    for (int i = 0; i < n_rev; ++i) {
        q[static_cast<size_t>(i)] = data->qpos[qpos_idxs[i]];
    }
    return q;
}

std::vector<double> compute_dq_rev_pri(
        const mjModel* model,
        const mjData* data,
        const std::vector<double>& J_full,
        const std::vector<double>& e,
        int task_rows,
        const int* idxs_jac_rev_pri,
        int n_rev,
        const int* idxs_use,
        int n_use,
        const int* is_rev,
        const double* q_mins,
        const double* q_maxs,
        const int* qpos_idxs,
        const double* q_home,
        int q_home_len,
        const double* lam_use_data,
        const int* coupling_passive,
        const int* coupling_active,
        const double* coupling_coefs,
        const double* coupling_x0,
        int n_coupling,
        double ik_stepsize_rev,
        double ik_stepsize_pri,
        double max_probe_rev,
        double max_probe_pri,
        double k_null,
        bool joint_limit_handle_flag,
        bool nullspace_control_flag,
        bool task_null_mask_flag
    ) {
    const int nv = model->nv;
    std::vector<double> J_rev(static_cast<size_t>(task_rows*n_rev),0.0);
    for (int row = 0; row < task_rows; ++row) {
        for (int col = 0; col < n_rev; ++col) {
            J_rev[static_cast<size_t>(row*n_rev + col)] =
                J_full[static_cast<size_t>(row*nv + idxs_jac_rev_pri[col])];
        }
    }

    const std::vector<double> q_rev = gather_q_rev(data,qpos_idxs,n_rev);
    for (int c = 0; c < n_coupling; ++c) {
        const int idx_passive = coupling_passive[c];
        const int idx_active = coupling_active[c];
        const double x = q_rev[static_cast<size_t>(idx_active)];
        const double dfdx = eval_quartic_derivative(
            x,
            coupling_coefs + 5*c,
            coupling_x0[c]
        );
        for (int row = 0; row < task_rows; ++row) {
            J_rev[static_cast<size_t>(row*n_rev + idx_active)] +=
                J_rev[static_cast<size_t>(row*n_rev + idx_passive)]*dfdx;
            J_rev[static_cast<size_t>(row*n_rev + idx_passive)] = 0.0;
        }
    }

    std::vector<double> J_use(static_cast<size_t>(task_rows*n_use),0.0);
    std::vector<double> lam_use(static_cast<size_t>(n_use),1e-6);
    std::vector<double> steps_use(static_cast<size_t>(n_use),0.0);
    for (int col = 0; col < n_use; ++col) {
        const int rev_idx = idxs_use[col];
        lam_use[static_cast<size_t>(col)] = lam_use_data[col];
        steps_use[static_cast<size_t>(col)] = is_rev[rev_idx] ? ik_stepsize_rev : ik_stepsize_pri;
        for (int row = 0; row < task_rows; ++row) {
            J_use[static_cast<size_t>(row*n_use + col)] =
                J_rev[static_cast<size_t>(row*n_rev + rev_idx)];
        }
    }

    std::vector<double> dq_use = solve_dls(J_use,task_rows,n_use,e,lam_use);
    for (int i = 0; i < n_use; ++i) {
        dq_use[static_cast<size_t>(i)] *= steps_use[static_cast<size_t>(i)];
    }
    if (!all_finite(dq_use)) {
        dq_use.assign(static_cast<size_t>(n_use),0.0);
    }

    std::vector<int> valid_local(static_cast<size_t>(n_use),0);
    for (int i = 0; i < n_use; ++i) {
        valid_local[static_cast<size_t>(i)] = i;
    }

    if (joint_limit_handle_flag) {
        std::vector<int> valid;
        valid.reserve(static_cast<size_t>(n_use));
        for (int local = 0; local < n_use; ++local) {
            const int rev_idx = idxs_use[local];
            const double q = q_rev[static_cast<size_t>(rev_idx)];
            const double probe = is_rev[rev_idx] ? max_probe_rev : max_probe_pri;
            const double dq = dq_use[static_cast<size_t>(local)];
            const double q_probe = q + probe*sign_value(dq);
            const bool hit_max = (q_probe > q_maxs[rev_idx]) && (dq > 0.0);
            const bool hit_min = (q_probe < q_mins[rev_idx]) && (dq < 0.0);
            if (!hit_max && !hit_min) {
                valid.push_back(local);
            }
        }

        if (static_cast<int>(valid.size()) < n_use) {
            valid_local = valid;
            dq_use.assign(static_cast<size_t>(n_use),0.0);
            const int n_valid = static_cast<int>(valid_local.size());
            if (n_valid > 0) {
                std::vector<double> J_use2(static_cast<size_t>(task_rows*n_valid),0.0);
                std::vector<double> lam_use2(static_cast<size_t>(n_valid),0.0);
                for (int col = 0; col < n_valid; ++col) {
                    const int local = valid_local[static_cast<size_t>(col)];
                    lam_use2[static_cast<size_t>(col)] = lam_use[static_cast<size_t>(local)];
                    for (int row = 0; row < task_rows; ++row) {
                        J_use2[static_cast<size_t>(row*n_valid + col)] =
                            J_use[static_cast<size_t>(row*n_use + local)];
                    }
                }
                std::vector<double> dq_use2 = solve_dls(J_use2,task_rows,n_valid,e,lam_use2);
                for (int col = 0; col < n_valid; ++col) {
                    const int local = valid_local[static_cast<size_t>(col)];
                    dq_use[static_cast<size_t>(local)] =
                        dq_use2[static_cast<size_t>(col)]*steps_use[static_cast<size_t>(local)];
                }
            }
        }
    }

    if (!all_finite(dq_use)) {
        dq_use.assign(static_cast<size_t>(n_use),0.0);
    }

    if (nullspace_control_flag && q_home_len == n_rev && !valid_local.empty()) {
        const int n_valid = static_cast<int>(valid_local.size());
        std::vector<double> J_valid(static_cast<size_t>(task_rows*n_valid),0.0);
        std::vector<double> lam_valid(static_cast<size_t>(n_valid),0.0);
        std::vector<double> dq_null(static_cast<size_t>(n_valid),0.0);
        for (int col = 0; col < n_valid; ++col) {
            const int local = valid_local[static_cast<size_t>(col)];
            const int rev_idx = idxs_use[local];
            lam_valid[static_cast<size_t>(col)] = lam_use[static_cast<size_t>(local)];
            dq_null[static_cast<size_t>(col)] =
                k_null*(q_home[rev_idx] - q_rev[static_cast<size_t>(rev_idx)]);
            for (int row = 0; row < task_rows; ++row) {
                J_valid[static_cast<size_t>(row*n_valid + col)] =
                    J_use[static_cast<size_t>(row*n_use + local)];
            }
        }
        if (task_null_mask_flag) {
            mask_task_relevant(J_valid,task_rows,n_valid,dq_null);
        }
        std::vector<double> dq_add =
            project_dq_null_damped(J_valid,task_rows,n_valid,dq_null,lam_valid);
        for (int col = 0; col < n_valid; ++col) {
            const int local = valid_local[static_cast<size_t>(col)];
            dq_use[static_cast<size_t>(local)] +=
                dq_add[static_cast<size_t>(col)]*steps_use[static_cast<size_t>(local)];
        }
    }

    std::vector<double> dq_rev(static_cast<size_t>(n_rev),0.0);
    for (int local = 0; local < n_use; ++local) {
        dq_rev[static_cast<size_t>(idxs_use[local])] = dq_use[static_cast<size_t>(local)];
    }
    if (!all_finite(dq_rev)) {
        dq_rev.assign(static_cast<size_t>(n_rev),0.0);
    }
    return dq_rev;
}

std::vector<double> dq_to_q_next(
        const mjData* data,
        int n_rev,
        const int* qpos_idxs,
        const int* idxs_active,
        int n_active,
        const int* is_rev,
        const double* q_mins,
        const double* q_maxs,
        const std::vector<double>& dq_rev,
        const int* coupling_passive,
        const int* coupling_active,
        const double* coupling_coefs,
        const double* coupling_x0,
        const double* coupling_y0,
        int n_coupling,
        double ik_update_th_rev,
        double ik_update_th_pri
    ) {
    std::vector<double> q_curr = gather_q_rev(data,qpos_idxs,n_rev);
    std::vector<double> q_next = q_curr;
    std::vector<double> dq_active(static_cast<size_t>(n_active),0.0);

    std::vector<int> rev_locals;
    std::vector<int> pri_locals;
    for (int i = 0; i < n_active; ++i) {
        const int rev_idx = idxs_active[i];
        dq_active[static_cast<size_t>(i)] = dq_rev[static_cast<size_t>(rev_idx)];
        if (is_rev[rev_idx]) {
            rev_locals.push_back(i);
        } else {
            pri_locals.push_back(i);
        }
    }

    if (!rev_locals.empty()) {
        std::vector<double> values;
        values.reserve(rev_locals.size());
        for (int local : rev_locals) {
            values.push_back(dq_active[static_cast<size_t>(local)]);
        }
        trim_scale_in_place(values,ik_update_th_rev);
        for (size_t i = 0; i < rev_locals.size(); ++i) {
            dq_active[static_cast<size_t>(rev_locals[i])] = values[i];
        }
    }
    if (!pri_locals.empty()) {
        std::vector<double> values;
        values.reserve(pri_locals.size());
        for (int local : pri_locals) {
            values.push_back(dq_active[static_cast<size_t>(local)]);
        }
        trim_scale_in_place(values,ik_update_th_pri);
        for (size_t i = 0; i < pri_locals.size(); ++i) {
            dq_active[static_cast<size_t>(pri_locals[i])] = values[i];
        }
    }

    for (int i = 0; i < n_active; ++i) {
        const int rev_idx = idxs_active[i];
        q_next[static_cast<size_t>(rev_idx)] =
            q_curr[static_cast<size_t>(rev_idx)] + dq_active[static_cast<size_t>(i)];
    }

    for (int c = 0; c < n_coupling; ++c) {
        const int idx_passive = coupling_passive[c];
        const int idx_active = coupling_active[c];
        const double x = q_next[static_cast<size_t>(idx_active)];
        q_next[static_cast<size_t>(idx_passive)] = eval_quartic(
            x,
            coupling_coefs + 5*c,
            coupling_x0[c],
            coupling_y0[c]
        );
    }

    for (int i = 0; i < n_rev; ++i) {
        q_next[static_cast<size_t>(i)] =
            std::min(std::max(q_next[static_cast<size_t>(i)],q_mins[i]),q_maxs[i]);
    }
    return q_next;
}

void apply_q_rev(
        const mjModel* model,
        mjData* data,
        int n_rev,
        const int* qpos_idxs,
        const std::vector<double>& q_rev
    ) {
    for (int i = 0; i < n_rev; ++i) {
        data->qpos[qpos_idxs[i]] = q_rev[static_cast<size_t>(i)];
    }
    mj_forward(model,data);
}

void validate_shape(bool condition,const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void validate_index_array(
        const int* values,
        size_t count,
        int upper_bound,
        const char* name
    ) {
    for (size_t idx = 0; idx < count; ++idx) {
        if (values[idx] < 0 || values[idx] >= upper_bound) {
            throw std::runtime_error(std::string(name) + " contains an out-of-range index.");
        }
    }
}

void validate_binary_array(const int* values,size_t count,const char* name) {
    for (size_t idx = 0; idx < count; ++idx) {
        if (values[idx] != 0 && values[idx] != 1) {
            throw std::runtime_error(std::string(name) + " must contain only 0 or 1.");
        }
    }
}

void validate_finite_array(const double* values,size_t count,const char* name) {
    for (size_t idx = 0; idx < count; ++idx) {
        if (!std::isfinite(values[idx])) {
            throw std::runtime_error(std::string(name) + " must contain only finite values.");
        }
    }
}

void validate_not_nan_array(const double* values,size_t count,const char* name) {
    for (size_t idx = 0; idx < count; ++idx) {
        if (std::isnan(values[idx])) {
            throw std::runtime_error(std::string(name) + " must not contain NaN values.");
        }
    }
}

void validate_nonnegative_array(const double* values,size_t count,const char* name) {
    validate_finite_array(values,count,name);
    for (size_t idx = 0; idx < count; ++idx) {
        if (values[idx] < 0.0) {
            throw std::runtime_error(std::string(name) + " must be non-negative.");
        }
    }
}

void validate_finite_scalar(double value,const char* name) {
    if (!std::isfinite(value)) {
        throw std::runtime_error(std::string(name) + " must be finite.");
    }
}

void validate_nonnegative_scalar(double value,const char* name) {
    validate_finite_scalar(value,name);
    if (value < 0.0) {
        throw std::runtime_error(std::string(name) + " must be non-negative.");
    }
}

std::vector<double> compute_dq_free_base(
        const mjModel* model,
        const std::vector<double>& J_full,
        const std::vector<double>& e,
        int task_rows,
        const int* base_dof_idxs,
        double dls_damping,
        double base_stepsize,
        double base_pos_th,
        double base_rot_th
    ) {
    constexpr int n_base = 6;
    std::vector<double> J_base(static_cast<size_t>(task_rows*n_base),0.0);
    for (int row = 0; row < task_rows; ++row) {
        for (int col = 0; col < n_base; ++col) {
            J_base[static_cast<size_t>(row*n_base + col)] =
                J_full[static_cast<size_t>(row*model->nv + base_dof_idxs[col])];
        }
    }

    const double lam_value = std::max(dls_damping,1e-12);
    const std::vector<double> lam(static_cast<size_t>(n_base),lam_value);
    std::vector<double> dq6 = solve_dls(J_base,task_rows,n_base,e,lam);
    for (double& value : dq6) {
        value *= base_stepsize;
    }
    if (!all_finite(dq6)) {
        return std::vector<double>(static_cast<size_t>(n_base),0.0);
    }

    std::vector<double> dp(dq6.begin(),dq6.begin() + 3);
    std::vector<double> dr(dq6.begin() + 3,dq6.end());
    trim_scale_in_place(dp,base_pos_th);
    trim_scale_in_place(dr,base_rot_th);
    for (int axis = 0; axis < 3; ++axis) {
        dq6[static_cast<size_t>(axis)] = dp[static_cast<size_t>(axis)];
        dq6[static_cast<size_t>(3 + axis)] = dr[static_cast<size_t>(axis)];
    }
    if (!all_finite(dq6)) {
        return std::vector<double>(static_cast<size_t>(n_base),0.0);
    }
    return dq6;
}

void apply_free_base_delta(
        const mjModel* model,
        mjData* data,
        const int* base_dof_idxs,
        const std::vector<double>& dq6
    ) {
    std::vector<double> dq_full(static_cast<size_t>(model->nv),0.0);
    for (int axis = 0; axis < 6; ++axis) {
        dq_full[static_cast<size_t>(base_dof_idxs[axis])] = dq6[static_cast<size_t>(axis)];
    }
    mj_integratePos(model,data->qpos,dq_full.data(),1.0);
    mj_forward(model,data);
}

double get_base_yaw(const mjData* data,int base_body_id) {
    const double* R = data->xmat + 9*base_body_id;
    return std::atan2(R[3],R[0]);
}

std::vector<double> get_base_planar_q(const mjData* data,int base_body_id,int base_qposadr) {
    return {
        data->qpos[base_qposadr + 0],
        data->qpos[base_qposadr + 1],
        get_base_yaw(data,base_body_id),
    };
}

void sanitize_planar_base_pose(
        const mjModel* model,
        mjData* data,
        int base_body_id,
        int base_qposadr,
        double base_z_home
    ) {
    mj_forward(model,data);
    const double yaw = get_base_yaw(data,base_body_id);
    data->qpos[base_qposadr + 2] = base_z_home;
    data->qpos[base_qposadr + 3] = std::cos(0.5*yaw);
    data->qpos[base_qposadr + 4] = 0.0;
    data->qpos[base_qposadr + 5] = 0.0;
    data->qpos[base_qposadr + 6] = std::sin(0.5*yaw);
    mj_forward(model,data);
}

std::vector<double> build_base_motion_matrix(
        const mjData* data,
        int base_body_id,
        const int* base_motion_codes,
        int n_base_dof,
        const double* base_x_local
    ) {
    const double yaw = get_base_yaw(data,base_body_id);
    const double cy = std::cos(yaw);
    const double sy = std::sin(yaw);
    const double bx = base_x_local[0];
    const double by = base_x_local[1];
    const double base_x_world[2] = {cy*bx - sy*by, sy*bx + cy*by};

    std::vector<double> M(static_cast<size_t>(6*n_base_dof),0.0);
    for (int dof_idx = 0; dof_idx < n_base_dof; ++dof_idx) {
        const int code = base_motion_codes[dof_idx];
        if (code == 0) {
            M[static_cast<size_t>(0*n_base_dof + dof_idx)] = base_x_world[0];
            M[static_cast<size_t>(1*n_base_dof + dof_idx)] = base_x_world[1];
        } else if (code == 1) {
            M[static_cast<size_t>(0*n_base_dof + dof_idx)] = 1.0;
        } else if (code == 2) {
            M[static_cast<size_t>(1*n_base_dof + dof_idx)] = 1.0;
        } else if (code == 3) {
            M[static_cast<size_t>(5*n_base_dof + dof_idx)] = 1.0;
        }
    }
    return M;
}

std::vector<double> compute_dq_planar_base(
        const mjModel* model,
        const mjData* data,
        const std::vector<double>& J_full,
        const std::vector<double>& e,
        int task_rows,
        const int* base_dof_idxs,
        const int* base_motion_codes,
        int n_base_dof,
        int base_body_id,
        const double* base_x_local,
        double base_stepsize,
        double base_pos_th,
        double base_yaw_th,
        const double* dls_damping_use
    ) {
    const int nv = model->nv;
    const std::vector<double> M = build_base_motion_matrix(
        data,base_body_id,base_motion_codes,n_base_dof,base_x_local
    );
    std::vector<double> J_planar(static_cast<size_t>(task_rows*n_base_dof),0.0);
    for (int row = 0; row < task_rows; ++row) {
        for (int col = 0; col < n_base_dof; ++col) {
            double value = 0.0;
            for (int k = 0; k < 6; ++k) {
                value += (
                    J_full[static_cast<size_t>(row*nv + base_dof_idxs[k])]*
                    M[static_cast<size_t>(k*n_base_dof + col)]
                );
            }
            J_planar[static_cast<size_t>(row*n_base_dof + col)] = value;
        }
    }

    std::vector<double> lam(
        dls_damping_use,
        dls_damping_use + static_cast<size_t>(n_base_dof)
    );
    std::vector<double> dq = solve_dls(J_planar,task_rows,n_base_dof,e,lam);
    for (double& value : dq) {
        value *= base_stepsize;
    }
    if (!all_finite(dq)) {
        return std::vector<double>(static_cast<size_t>(n_base_dof),0.0);
    }

    std::vector<double> dq_trans;
    std::vector<int> trans_idxs;
    std::vector<double> dq_yaw;
    std::vector<int> yaw_idxs;
    for (int i = 0; i < n_base_dof; ++i) {
        const int code = base_motion_codes[i];
        if (code == 0 || code == 1 || code == 2) {
            trans_idxs.push_back(i);
            dq_trans.push_back(dq[static_cast<size_t>(i)]);
        } else if (code == 3) {
            yaw_idxs.push_back(i);
            dq_yaw.push_back(dq[static_cast<size_t>(i)]);
        }
    }
    if (!dq_trans.empty()) {
        trim_scale_in_place(dq_trans,base_pos_th);
        for (size_t i = 0; i < trans_idxs.size(); ++i) {
            dq[static_cast<size_t>(trans_idxs[i])] = dq_trans[i];
        }
    }
    if (!dq_yaw.empty()) {
        trim_scale_in_place(dq_yaw,base_yaw_th);
        for (size_t i = 0; i < yaw_idxs.size(); ++i) {
            dq[static_cast<size_t>(yaw_idxs[i])] = dq_yaw[i];
        }
    }
    if (!all_finite(dq)) {
        return std::vector<double>(static_cast<size_t>(n_base_dof),0.0);
    }
    return dq;
}

void apply_planar_base_delta(
        const mjModel* model,
        mjData* data,
        const int* base_dof_idxs,
        const int* base_motion_codes,
        int n_base_dof,
        int base_body_id,
        int base_qposadr,
        const double* base_x_local,
        const std::vector<double>& dq_planar,
        double base_z_home
    ) {
    const std::vector<double> M = build_base_motion_matrix(
        data,base_body_id,base_motion_codes,n_base_dof,base_x_local
    );
    std::vector<double> dq_base6(6,0.0);
    for (int row = 0; row < 6; ++row) {
        for (int col = 0; col < n_base_dof; ++col) {
            dq_base6[static_cast<size_t>(row)] += (
                M[static_cast<size_t>(row*n_base_dof + col)]*
                dq_planar[static_cast<size_t>(col)]
            );
        }
    }

    std::vector<double> dq_full(static_cast<size_t>(model->nv),0.0);
    for (int k = 0; k < 6; ++k) {
        dq_full[static_cast<size_t>(base_dof_idxs[k])] = dq_base6[static_cast<size_t>(k)];
    }
    mj_integratePos(model,data->qpos,dq_full.data(),1.0);
    sanitize_planar_base_pose(model,data,base_body_id,base_qposadr,base_z_home);
}

}

nb::dict solve_body_position_ik(
        std::uintptr_t model_address,
        std::uintptr_t data_address,
        nb::ndarray<nb::numpy,const int,nb::ndim<1>,nb::c_contig> body_idxs,
        nb::ndarray<nb::numpy,const double,nb::ndim<2>,nb::c_contig> p_local,
        nb::ndarray<nb::numpy,const double,nb::ndim<2>,nb::c_contig> p_trgt,
        nb::ndarray<nb::numpy,const double,nb::ndim<1>,nb::c_contig> weights,
        nb::ndarray<nb::numpy,const int,nb::ndim<1>,nb::c_contig> rev_pri_qpos_idxs,
        nb::ndarray<nb::numpy,const int,nb::ndim<1>,nb::c_contig> idxs_jac_rev_pri,
        nb::ndarray<nb::numpy,const int,nb::ndim<1>,nb::c_contig> idxs_rev_pri_use,
        nb::ndarray<nb::numpy,const int,nb::ndim<1>,nb::c_contig> idxs_rev_pri_active,
        nb::ndarray<nb::numpy,const int,nb::ndim<1>,nb::c_contig> is_rev_in_rev_pri,
        nb::ndarray<nb::numpy,const double,nb::ndim<1>,nb::c_contig> rev_pri_joint_mins,
        nb::ndarray<nb::numpy,const double,nb::ndim<1>,nb::c_contig> rev_pri_joint_maxs,
        nb::ndarray<nb::numpy,const double,nb::ndim<1>,nb::c_contig> q_home_rev_pri,
        nb::ndarray<nb::numpy,const double,nb::ndim<1>,nb::c_contig> dls_damping_use,
        nb::ndarray<nb::numpy,const int,nb::ndim<1>,nb::c_contig> coupling_passive_idxs,
        nb::ndarray<nb::numpy,const int,nb::ndim<1>,nb::c_contig> coupling_active_idxs,
        nb::ndarray<nb::numpy,const double,nb::ndim<2>,nb::c_contig> coupling_coefs,
        nb::ndarray<nb::numpy,const double,nb::ndim<1>,nb::c_contig> coupling_x0,
        nb::ndarray<nb::numpy,const double,nb::ndim<1>,nb::c_contig> coupling_y0,
        nb::ndarray<nb::numpy,const int,nb::ndim<1>,nb::c_contig> base_dof_idxs,
        bool base_control_flag,
        double base_dls_damping,
        double base_stepsize,
        double base_pos_th,
        double base_rot_th,
        int max_ik_tick,
        double ik_stepsize_rev,
        double ik_stepsize_pri,
        double ik_update_th_rev,
        double ik_update_th_pri,
        double max_probe_rev,
        double max_probe_pri,
        double k_null,
        bool joint_limit_handle_flag,
        bool nullspace_control_flag,
        bool task_null_mask_flag
    ) {
    const mjModel* model = as_model(model_address);
    mjData* data = as_data_mutable(data_address);

    validate_shape(body_idxs.shape(0) <= static_cast<size_t>(std::numeric_limits<int>::max()/3),
        "body_idxs is too large.");
    validate_shape(rev_pri_qpos_idxs.shape(0) <= static_cast<size_t>(std::numeric_limits<int>::max()),
        "rev_pri_qpos_idxs is too large.");
    validate_shape(idxs_rev_pri_use.shape(0) <= static_cast<size_t>(std::numeric_limits<int>::max()),
        "idxs_rev_pri_use is too large.");
    validate_shape(idxs_rev_pri_active.shape(0) <= static_cast<size_t>(std::numeric_limits<int>::max()),
        "idxs_rev_pri_active is too large.");
    validate_shape(coupling_passive_idxs.shape(0) <= static_cast<size_t>(std::numeric_limits<int>::max()),
        "coupling_passive_idxs is too large.");

    const int n_target = static_cast<int>(body_idxs.shape(0));
    const int n_rev = static_cast<int>(rev_pri_qpos_idxs.shape(0));
    const int n_use = static_cast<int>(idxs_rev_pri_use.shape(0));
    const int n_active = static_cast<int>(idxs_rev_pri_active.shape(0));
    const int n_coupling = static_cast<int>(coupling_passive_idxs.shape(0));

    validate_shape(p_local.shape(0) == static_cast<size_t>(n_target) && p_local.shape(1) == 3,
        "p_local must have shape (n_target, 3).");
    validate_shape(p_trgt.shape(0) == static_cast<size_t>(n_target) && p_trgt.shape(1) == 3,
        "p_trgt must have shape (n_target, 3).");
    validate_shape(weights.shape(0) == static_cast<size_t>(n_target),
        "weights must have shape (n_target,).");
    validate_shape(idxs_jac_rev_pri.shape(0) == static_cast<size_t>(n_rev),
        "idxs_jac_rev_pri must match rev_pri_qpos_idxs.");
    validate_shape(is_rev_in_rev_pri.shape(0) == static_cast<size_t>(n_rev),
        "is_rev_in_rev_pri must match rev_pri_qpos_idxs.");
    validate_shape(rev_pri_joint_mins.shape(0) == static_cast<size_t>(n_rev),
        "rev_pri_joint_mins must match rev_pri_qpos_idxs.");
    validate_shape(rev_pri_joint_maxs.shape(0) == static_cast<size_t>(n_rev),
        "rev_pri_joint_maxs must match rev_pri_qpos_idxs.");
    validate_shape(dls_damping_use.shape(0) == static_cast<size_t>(n_use),
        "dls_damping_use must match idxs_rev_pri_use.");
    validate_shape(coupling_active_idxs.shape(0) == static_cast<size_t>(n_coupling),
        "coupling_active_idxs must match coupling_passive_idxs.");
    validate_shape(coupling_coefs.shape(0) == static_cast<size_t>(n_coupling) && coupling_coefs.shape(1) == 5,
        "coupling_coefs must have shape (n_coupling, 5).");
    validate_shape(coupling_x0.shape(0) == static_cast<size_t>(n_coupling),
        "coupling_x0 must match coupling_passive_idxs.");
    validate_shape(coupling_y0.shape(0) == static_cast<size_t>(n_coupling),
        "coupling_y0 must match coupling_passive_idxs.");
    validate_shape((!base_control_flag) || base_dof_idxs.shape(0) == 6,
        "base_dof_idxs must have shape (6,) when base_control_flag is true.");

    validate_shape(max_ik_tick >= 0,"max_ik_tick must be non-negative.");
    validate_shape(
        q_home_rev_pri.shape(0) == 0 || q_home_rev_pri.shape(0) == static_cast<size_t>(n_rev),
        "q_home_rev_pri must be empty or match rev_pri_qpos_idxs."
    );
    validate_index_array(body_idxs.data(),body_idxs.shape(0),model->nbody,"body_idxs");
    validate_index_array(
        rev_pri_qpos_idxs.data(),rev_pri_qpos_idxs.shape(0),model->nq,"rev_pri_qpos_idxs"
    );
    validate_index_array(
        idxs_jac_rev_pri.data(),idxs_jac_rev_pri.shape(0),model->nv,"idxs_jac_rev_pri"
    );
    validate_index_array(
        idxs_rev_pri_use.data(),idxs_rev_pri_use.shape(0),n_rev,"idxs_rev_pri_use"
    );
    validate_index_array(
        idxs_rev_pri_active.data(),idxs_rev_pri_active.shape(0),n_rev,"idxs_rev_pri_active"
    );
    validate_binary_array(
        is_rev_in_rev_pri.data(),is_rev_in_rev_pri.shape(0),"is_rev_in_rev_pri"
    );
    validate_index_array(
        coupling_passive_idxs.data(),coupling_passive_idxs.shape(0),n_rev,
        "coupling_passive_idxs"
    );
    validate_index_array(
        coupling_active_idxs.data(),coupling_active_idxs.shape(0),n_rev,
        "coupling_active_idxs"
    );

    validate_finite_array(p_local.data(),body_idxs.shape(0)*3,"p_local");
    validate_finite_array(p_trgt.data(),body_idxs.shape(0)*3,"p_trgt");
    validate_nonnegative_array(weights.data(),weights.shape(0),"weights");
    validate_not_nan_array(
        rev_pri_joint_mins.data(),rev_pri_joint_mins.shape(0),"rev_pri_joint_mins"
    );
    validate_not_nan_array(
        rev_pri_joint_maxs.data(),rev_pri_joint_maxs.shape(0),"rev_pri_joint_maxs"
    );
    for (int rev_idx = 0; rev_idx < n_rev; ++rev_idx) {
        validate_shape(
            rev_pri_joint_mins.data()[rev_idx] <= rev_pri_joint_maxs.data()[rev_idx],
            "rev_pri_joint_mins must not exceed rev_pri_joint_maxs."
        );
    }
    validate_finite_array(q_home_rev_pri.data(),q_home_rev_pri.shape(0),"q_home_rev_pri");
    validate_nonnegative_array(
        dls_damping_use.data(),dls_damping_use.shape(0),"dls_damping_use"
    );
    validate_finite_array(
        coupling_coefs.data(),coupling_passive_idxs.shape(0)*5,"coupling_coefs"
    );
    validate_finite_array(coupling_x0.data(),coupling_x0.shape(0),"coupling_x0");
    validate_finite_array(coupling_y0.data(),coupling_y0.shape(0),"coupling_y0");

    validate_nonnegative_scalar(base_dls_damping,"base_dls_damping");
    validate_nonnegative_scalar(base_stepsize,"base_stepsize");
    validate_nonnegative_scalar(base_pos_th,"base_pos_th");
    validate_nonnegative_scalar(base_rot_th,"base_rot_th");
    validate_nonnegative_scalar(ik_stepsize_rev,"ik_stepsize_rev");
    validate_nonnegative_scalar(ik_stepsize_pri,"ik_stepsize_pri");
    validate_nonnegative_scalar(ik_update_th_rev,"ik_update_th_rev");
    validate_nonnegative_scalar(ik_update_th_pri,"ik_update_th_pri");
    validate_nonnegative_scalar(max_probe_rev,"max_probe_rev");
    validate_nonnegative_scalar(max_probe_pri,"max_probe_pri");
    validate_nonnegative_scalar(k_null,"k_null");

    if (base_control_flag) {
        for (int axis = 0; axis < 6; ++axis) {
            const int dof_idx = base_dof_idxs.data()[axis];
            validate_shape(dof_idx >= 0 && dof_idx < model->nv,
                "base_dof_idxs contains an out-of-range index.");
        }
    }

    if (n_target == 0 || n_rev == 0 || n_use == 0) {
        nb::dict empty;
        empty["q_rev_pri_best"] = copy_array<double>({}, {0});
        empty["ik_err_array"] = copy_array<double>({0.0}, {1});
        empty["idx_best"] = 0;
        empty["ik_err_best"] = 0.0;
        empty["elapsed_time"] = 0.0;
        empty["qpos_full_best"] = copy_array<double>(
            std::vector<double>(data->qpos,data->qpos + model->nq),
            {static_cast<size_t>(model->nq)}
        );
        empty["qpos_used_best"] = copy_array<double>({}, {0});
        return empty;
    }

    auto gil_release = std::make_unique<nb::gil_scoped_release>();
    const auto t0 = std::chrono::steady_clock::now();
    const int task_rows = 3*n_target;
    std::vector<double> J;
    std::vector<double> e;
    std::vector<double> p_err;
    std::vector<double> ik_err_array(static_cast<size_t>(max_ik_tick + 1),0.0);

    build_weighted_Je(
        model,data,n_target,body_idxs.data(),p_local.data(),p_trgt.data(),weights.data(),J,e,p_err
    );
    ik_err_array[0] = norm2(p_err);

    std::vector<double> q_rev_best = gather_q_rev(data,rev_pri_qpos_idxs.data(),n_rev);
    std::vector<double> qpos_full_best(data->qpos,data->qpos + model->nq);
    int idx_best = 0;
    double ik_err_best = ik_err_array[0];

    for (int ik_tick = 0; ik_tick < max_ik_tick; ++ik_tick) {
        build_weighted_Je(
            model,data,n_target,body_idxs.data(),p_local.data(),p_trgt.data(),weights.data(),J,e,p_err
        );

        if (base_control_flag) {
            const std::vector<double> dq6 = compute_dq_free_base(
                model,
                J,
                e,
                task_rows,
                base_dof_idxs.data(),
                base_dls_damping,
                base_stepsize,
                base_pos_th,
                base_rot_th
            );
            apply_free_base_delta(model,data,base_dof_idxs.data(),dq6);
            build_weighted_Je(
                model,data,n_target,body_idxs.data(),p_local.data(),p_trgt.data(),weights.data(),J,e,p_err
            );
        }

        const std::vector<double> dq_rev = compute_dq_rev_pri(
            model,
            data,
            J,
            e,
            task_rows,
            idxs_jac_rev_pri.data(),
            n_rev,
            idxs_rev_pri_use.data(),
            n_use,
            is_rev_in_rev_pri.data(),
            rev_pri_joint_mins.data(),
            rev_pri_joint_maxs.data(),
            rev_pri_qpos_idxs.data(),
            q_home_rev_pri.data(),
            static_cast<int>(q_home_rev_pri.shape(0)),
            dls_damping_use.data(),
            coupling_passive_idxs.data(),
            coupling_active_idxs.data(),
            coupling_coefs.data(),
            coupling_x0.data(),
            n_coupling,
            ik_stepsize_rev,
            ik_stepsize_pri,
            max_probe_rev,
            max_probe_pri,
            k_null,
            joint_limit_handle_flag,
            nullspace_control_flag,
            task_null_mask_flag
        );

        const std::vector<double> q_next = dq_to_q_next(
            data,
            n_rev,
            rev_pri_qpos_idxs.data(),
            idxs_rev_pri_active.data(),
            n_active,
            is_rev_in_rev_pri.data(),
            rev_pri_joint_mins.data(),
            rev_pri_joint_maxs.data(),
            dq_rev,
            coupling_passive_idxs.data(),
            coupling_active_idxs.data(),
            coupling_coefs.data(),
            coupling_x0.data(),
            coupling_y0.data(),
            n_coupling,
            ik_update_th_rev,
            ik_update_th_pri
        );
        apply_q_rev(model,data,n_rev,rev_pri_qpos_idxs.data(),q_next);

        build_weighted_Je(
            model,data,n_target,body_idxs.data(),p_local.data(),p_trgt.data(),weights.data(),J,e,p_err
        );
        const double err = norm2(p_err);
        ik_err_array[static_cast<size_t>(ik_tick + 1)] = err;
        if (err < ik_err_best) {
            ik_err_best = err;
            idx_best = ik_tick + 1;
            q_rev_best = gather_q_rev(data,rev_pri_qpos_idxs.data(),n_rev);
            qpos_full_best.assign(data->qpos,data->qpos + model->nq);
        }
    }

    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed =
        std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();

    std::vector<double> qpos_used_best(static_cast<size_t>(n_use),0.0);
    for (int i = 0; i < n_use; ++i) {
        const int rev_idx = idxs_rev_pri_use.data()[i];
        const int qpos_idx = rev_pri_qpos_idxs.data()[rev_idx];
        qpos_used_best[static_cast<size_t>(i)] = qpos_full_best[static_cast<size_t>(qpos_idx)];
    }

    gil_release.reset();
    nb::dict out;
    out["q_rev_pri_best"] = copy_array<double>(q_rev_best,{static_cast<size_t>(n_rev)});
    out["ik_err_array"] = copy_array<double>(ik_err_array,{static_cast<size_t>(max_ik_tick + 1)});
    out["idx_best"] = idx_best;
    out["ik_err_best"] = ik_err_best;
    out["elapsed_time"] = elapsed;
    out["qpos_full_best"] = copy_array<double>(qpos_full_best,{static_cast<size_t>(model->nq)});
    out["qpos_used_best"] = copy_array<double>(qpos_used_best,{static_cast<size_t>(n_use)});
    return out;
}

nb::dict solve_planar_base_ik(
        std::uintptr_t model_address,
        std::uintptr_t data_address,
        nb::ndarray<nb::numpy,const int,nb::ndim<1>,nb::c_contig> body_idxs,
        nb::ndarray<nb::numpy,const double,nb::ndim<2>,nb::c_contig> p_local,
        nb::ndarray<nb::numpy,const double,nb::ndim<2>,nb::c_contig> p_trgt,
        nb::ndarray<nb::numpy,const double,nb::ndim<1>,nb::c_contig> weights,
        nb::ndarray<nb::numpy,const int,nb::ndim<1>,nb::c_contig> base_dof_idxs,
        nb::ndarray<nb::numpy,const int,nb::ndim<1>,nb::c_contig> base_motion_codes,
        nb::ndarray<nb::numpy,const double,nb::ndim<1>,nb::c_contig> base_x_local,
        int base_body_id,
        int base_qposadr,
        int max_ik_tick,
        double base_stepsize,
        double base_pos_th,
        double base_yaw_th,
        nb::ndarray<nb::numpy,const double,nb::ndim<1>,nb::c_contig> dls_damping_use,
        double base_z_home
    ) {
    const mjModel* model = as_model(model_address);
    mjData* data = as_data_mutable(data_address);

    validate_shape(body_idxs.shape(0) <= static_cast<size_t>(std::numeric_limits<int>::max()/3),
        "body_idxs is too large.");
    validate_shape(base_motion_codes.shape(0) <= static_cast<size_t>(std::numeric_limits<int>::max()),
        "base_motion_codes is too large.");

    const int n_target = static_cast<int>(body_idxs.shape(0));
    const int n_base_dof = static_cast<int>(base_motion_codes.shape(0));

    validate_shape(p_local.shape(0) == static_cast<size_t>(n_target) && p_local.shape(1) == 3,
        "p_local must have shape (n_target, 3).");
    validate_shape(p_trgt.shape(0) == static_cast<size_t>(n_target) && p_trgt.shape(1) == 3,
        "p_trgt must have shape (n_target, 3).");
    validate_shape(weights.shape(0) == static_cast<size_t>(n_target),
        "weights must have shape (n_target,).");
    validate_shape(base_dof_idxs.shape(0) == 6,
        "base_dof_idxs must have shape (6,).");
    validate_shape(base_x_local.shape(0) == 2,
        "base_x_local must have shape (2,).");
    validate_shape(dls_damping_use.shape(0) == static_cast<size_t>(n_base_dof),
        "dls_damping_use must match base_motion_codes.");
    validate_shape(n_base_dof > 0,
        "base_motion_codes must not be empty.");

    validate_shape(max_ik_tick >= 0,"max_ik_tick must be non-negative.");
    validate_shape(base_body_id >= 0 && base_body_id < model->nbody,
        "base_body_id is out of range.");
    validate_shape(base_qposadr >= 0 && base_qposadr + 6 < model->nq,
        "base_qposadr must address a seven-value free-joint pose.");
    validate_index_array(body_idxs.data(),body_idxs.shape(0),model->nbody,"body_idxs");
    validate_index_array(
        base_dof_idxs.data(),base_dof_idxs.shape(0),model->nv,"base_dof_idxs"
    );
    for (size_t idx = 0; idx < base_motion_codes.shape(0); ++idx) {
        const int code = base_motion_codes.data()[idx];
        validate_shape(code >= 0 && code <= 3,
            "base_motion_codes must contain only 0, 1, 2, or 3.");
    }
    validate_finite_array(p_local.data(),body_idxs.shape(0)*3,"p_local");
    validate_finite_array(p_trgt.data(),body_idxs.shape(0)*3,"p_trgt");
    validate_nonnegative_array(weights.data(),weights.shape(0),"weights");
    validate_finite_array(base_x_local.data(),base_x_local.shape(0),"base_x_local");
    validate_nonnegative_array(
        dls_damping_use.data(),dls_damping_use.shape(0),"dls_damping_use"
    );
    validate_nonnegative_scalar(base_stepsize,"base_stepsize");
    validate_nonnegative_scalar(base_pos_th,"base_pos_th");
    validate_nonnegative_scalar(base_yaw_th,"base_yaw_th");
    validate_finite_scalar(base_z_home,"base_z_home");

    sanitize_planar_base_pose(model,data,base_body_id,base_qposadr,base_z_home);

    if (n_target == 0) {
        nb::dict empty;
        empty["ik_err_array"] = copy_array<double>({0.0},{1});
        empty["idx_best"] = 0;
        empty["ik_err_best"] = 0.0;
        empty["elapsed_time"] = 0.0;
        empty["qpos_full_best"] = copy_array<double>(
            std::vector<double>(data->qpos,data->qpos + model->nq),
            {static_cast<size_t>(model->nq)}
        );
        empty["base_planar_best"] = copy_array<double>(
            get_base_planar_q(data,base_body_id,base_qposadr),
            {3}
        );
        return empty;
    }

    auto gil_release = std::make_unique<nb::gil_scoped_release>();
    const auto t0 = std::chrono::steady_clock::now();
    const int task_rows = 3*n_target;
    std::vector<double> J;
    std::vector<double> e;
    std::vector<double> p_err;
    std::vector<double> ik_err_array(static_cast<size_t>(max_ik_tick + 1),0.0);

    build_weighted_Je(
        model,data,n_target,body_idxs.data(),p_local.data(),p_trgt.data(),weights.data(),J,e,p_err
    );
    ik_err_array[0] = norm2(p_err);

    std::vector<double> qpos_full_best(data->qpos,data->qpos + model->nq);
    std::vector<double> base_planar_best = get_base_planar_q(data,base_body_id,base_qposadr);
    int idx_best = 0;
    double ik_err_best = ik_err_array[0];

    for (int ik_tick = 0; ik_tick < max_ik_tick; ++ik_tick) {
        build_weighted_Je(
            model,data,n_target,body_idxs.data(),p_local.data(),p_trgt.data(),weights.data(),J,e,p_err
        );

        const std::vector<double> dq_planar = compute_dq_planar_base(
            model,
            data,
            J,
            e,
            task_rows,
            base_dof_idxs.data(),
            base_motion_codes.data(),
            n_base_dof,
            base_body_id,
            base_x_local.data(),
            base_stepsize,
            base_pos_th,
            base_yaw_th,
            dls_damping_use.data()
        );
        apply_planar_base_delta(
            model,
            data,
            base_dof_idxs.data(),
            base_motion_codes.data(),
            n_base_dof,
            base_body_id,
            base_qposadr,
            base_x_local.data(),
            dq_planar,
            base_z_home
        );

        build_weighted_Je(
            model,data,n_target,body_idxs.data(),p_local.data(),p_trgt.data(),weights.data(),J,e,p_err
        );
        const double err = norm2(p_err);
        ik_err_array[static_cast<size_t>(ik_tick + 1)] = err;
        if (err < ik_err_best) {
            ik_err_best = err;
            idx_best = ik_tick + 1;
            qpos_full_best.assign(data->qpos,data->qpos + model->nq);
            base_planar_best = get_base_planar_q(data,base_body_id,base_qposadr);
        }
    }

    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed =
        std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();

    gil_release.reset();
    nb::dict out;
    out["ik_err_array"] = copy_array<double>(ik_err_array,{static_cast<size_t>(max_ik_tick + 1)});
    out["idx_best"] = idx_best;
    out["ik_err_best"] = ik_err_best;
    out["elapsed_time"] = elapsed;
    out["qpos_full_best"] = copy_array<double>(qpos_full_best,{static_cast<size_t>(model->nq)});
    out["qpos_used_best"] = copy_array<double>(base_planar_best,{3});
    out["base_planar_best"] = copy_array<double>(base_planar_best,{3});
    return out;
}
