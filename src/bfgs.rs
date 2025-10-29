/// bfgs/bfgs.jl in rs
use super::bfgs_types::*;
use nalgebra::{DMatrix, DVector};
use std::f64;
use std::time::{Duration, Instant};

// -----------------------------------------------------------------------------
// printing utilities
// -----------------------------------------------------------------------------

fn print_header_bfgs(format_parts: &[&str], data: &[String]) {
    // Julia:
    //  "─────────────────..."
    //  @printf row of headers using a format string based on `format`
    //  "─────────────────..."

    let bar = "\n─────────────────────────────────────────────────────────────────────────────────────────────────\n";
    print!("{bar}");

    let fmt = format_parts.join(" ") + "\n";

    // we interpret each entry in `data` as already formatted string
    // so we just print them aligned using the given fmt
    // example: format_parts = ["%13s", "%14s", "%14s", "%14s"]
    // data = ["Iteration", "f", "||∇f||", "Time"]
    //
    // Rust's formatting syntax is different from printf's ("%13s" etc).
    // Easiest path: just left-pad manually.

    // We'll emulate printf-style padding:
    let rendered_line = format_line_printf_style(format_parts, data);
    print!("{rendered_line}\n");

    print!("{bar}");
}

fn print_iter(format_parts: &[&str], data: &[String]) {
    // one row each iteration
    let rendered_line = format_line_printf_style(format_parts, data);
    println!("{rendered_line}");
}

// helper to emulate printf-style width formats like "%13s", "%14.3e", etc.
fn format_line_printf_style(format_parts: &[&str], data: &[String]) -> String {
    let mut out_segments = Vec::new();

    for (spec, val) in format_parts.iter().zip(data.iter()) {
        // spec looks like "%13s" or "%14.3e" etc.
        // We’ll do a cheap parse:
        // width = number after '%', before any '.' or type char
        // precision if any after '.'
        // type char last char, e.g. 's','e','f'
        let mut width: Option<usize> = None;
        let mut precision: Option<usize> = None;
        let mut ty: Option<char> = None;

        if let Some(stripped) = spec.strip_prefix('%') {
            // find first non-digit or '.'
            let mut i = 0;
            // read width digits
            let mut wdigits = String::new();
            while i < stripped.len() {
                let c = stripped.as_bytes()[i] as char;
                if c.is_ascii_digit() {
                    wdigits.push(c);
                    i += 1;
                } else {
                    break;
                }
            }
            if !wdigits.is_empty() {
                width = wdigits.parse::<usize>().ok();
            }

            // read optional .precision
            if i < stripped.len() && stripped.as_bytes()[i] as char == '.' {
                i += 1;
                let mut pdigits = String::new();
                while i < stripped.len() {
                    let c = stripped.as_bytes()[i] as char;
                    if c.is_ascii_digit() {
                        pdigits.push(c);
                        i += 1;
                    } else {
                        break;
                    }
                }
                if !pdigits.is_empty() {
                    precision = pdigits.parse::<usize>().ok();
                }
            }

            // last char = type (s, e, f, etc.)
            if i < stripped.len() {
                ty = (stripped.as_bytes()[i] as char).into();
            }
        }

        // we already pre-formatted numeric values into strings on call site,
        // so at this point we just pad to width.
        let mut seg = val.clone();

        if let Some(w) = width {
            if seg.len() < w {
                let pad = " ".repeat(w - seg.len());
                seg = format!("{pad}{seg}");
            }
        }

        out_segments.push(seg);
    }

    out_segments.join(" ")
}

// Julia:
//   function print_headers(::BFGSSolver, ::BFGSOptions)
// just returns ["Iteration", "f", "||∇f||", "Time"]
fn print_headers() -> [&'static str; 4] {
    ["Iteration", "f", "||∇f||", "Time"]
}

// Julia header_format
// return ["%13s", "%14s", "%14s", "%14s"]
fn header_format() -> [&'static str; 4] {
    ["%13s", "%14s", "%14s", "%14s"]
}

// Julia iter_format
// return ["%13s", "%14.3e", "%14.3e", "%13.3f"]
fn iter_format() -> [&'static str; 4] {
    ["%13s", "%14.3e", "%14.3e", "%13.3f"]
}

// -----------------------------------------------------------------------------
// BFGS helpers
// -----------------------------------------------------------------------------

// compute_search_direction!(state::BFGSState)
// pk = - Hk * gk
pub fn compute_search_direction_bfgs(state: &mut BFGSState) {
    // pk = Hk * gk
    state.pk = &state.hk * &state.gk;
    // pk = -pk
    state.pk *= -1.0;
}

// update_Hk!(state::BFGSState)
// classic BFGS inverse Hessian update
pub fn update_hk_bfgs(state: &mut BFGSState) {
    let sk = state.sk.clone(); // step
    let yk = state.yk.clone(); // grad diff
    let hk_old = state.hk.clone();

    let denom = yk.dot(&sk);
    if !denom.is_finite() || denom == 0.0 {
        return;
    }
    let rho = 1.0 / denom;

    // I - ρ s y^T
    let n = state.hk.nrows();
    let i_mat = DMatrix::<f64>::identity(n, n);

    // outer products
    let s_yT = DMatrix::<f64>::from_fn(n, n, |i, j| sk[i] * yk[j]);
    let y_sT = DMatrix::<f64>::from_fn(n, n, |i, j| yk[i] * sk[j]);
    let s_sT = DMatrix::<f64>::from_fn(n, n, |i, j| sk[i] * sk[j]);

    // A = (I - rho * s y^T)
    let a = &i_mat - (rho * &s_yT);
    // B = (I - rho * y s^T)
    let b = &i_mat - (rho * &y_sT);

    // Hk = A * Hk * B + rho * s s^T
    state.hk = a * hk_old * b + rho * s_sT;
}

// scale_H0!(state::BFGSState)
// scale initial Hk by c = (y⋅s)/(y⋅y)
pub fn scale_h0_bfgs(state: &mut BFGSState) {
    let sk = &state.sk;
    let yk = &state.yk;

    let ys = yk.dot(sk);
    let yy = yk.dot(yk);

    let mut c = ys / yy;
    if !c.is_finite() {
        c = 1.0;
    }

    state.hk *= c;
}

// -----------------------------------------------------------------------------
// LBFGS helpers
// -----------------------------------------------------------------------------

// compute_search_direction!(state::LBFGSState)
// This is two-loop recursion to compute pk = -Hk * gk without storing Hk
pub fn compute_search_direction_lbfgs(state: &mut LBFGSState) {
    let m = state.sks.len();
    let n = state.xk.len();

    // Julia does circular buffer logic using "ind" and mod arithmetic.
    // We'll mirror the logic:
    // reverse_inds: go backward over last m updates
    // forward_inds: go forward in same wraparound order

    // our `ind[0]` is current position in [0, m-1]
    let ind = state.ind[0]; // 0-based in Rust

    // build reverse order indices
    let mut reverse_inds = Vec::with_capacity(m);
    for k in 0..m {
        // (ind-1, ind-2, ..., ind-m) mod m
        let idx = (ind + m - 1 - k) % m;
        reverse_inds.push(idx);
    }

    // build forward order
    let mut forward_inds = Vec::with_capacity(m);
    for k in 0..m {
        // (ind-m, ..., ind-1) mod m
        let idx = (ind + m - k - 1) % m;
        forward_inds.insert(0, idx); // reverse push to get forward order
    }

    // pk = gk
    state.pk.copy_from(&state.gk);

    // first loop: α_i = ρ_i * s_i⋅pk ; pk -= α_i * y_i
    for &i in &reverse_inds {
        let rho_i = state.rhos[i];
        if !rho_i.is_finite() || rho_i == 0.0 {
            state.alphas[i] = 0.0;
            continue;
        }
        let alpha_i = rho_i * state.sks[i].dot(&state.pk);
        state.alphas[i] = alpha_i;

        // pk -= α_i * y_i
        for j in 0..n {
            state.pk[j] -= alpha_i * state.yks[i][j];
        }
    }

    // pk = γk * pk, where γk is scalar stored in gamma_k[0]
    let gamma = state.gamma_k[0];
    state.pk *= gamma;

    // second loop: β_i = ρ_i * y_i⋅pk; pk += (α_i - β_i)*s_i
    for &i in &forward_inds {
        let rho_i = state.rhos[i];
        if !rho_i.is_finite() || rho_i == 0.0 {
            continue;
        }
        let beta_i = rho_i * state.yks[i].dot(&state.pk);
        state.betas[i] = beta_i;

        let corr = state.alphas[i] - beta_i;
        for j in 0..n {
            state.pk[j] += corr * state.sks[i][j];
        }
    }

    // final: pk = -pk
    state.pk *= -1.0;
}

// update_Hk!(state::LBFGSState)
// updates history sks/yks, gamma_k, and rho
pub fn update_hk_lbfgs(state: &mut LBFGSState) {
    let m = state.sks.len();
    let ind = state.ind[0]; // current slot

    // previous index in ring buffer
    let prev_ind = if ind == 0 { m - 1 } else { ind - 1 };

    // gamma_k[0] = (yᵀ s)/(yᵀ y) at prev_ind
    let ys = state.yks[prev_ind].dot(&state.sks[prev_ind]);
    let yy = state.yks[prev_ind].dot(&state.yks[prev_ind]);
    let mut gamma = ys / yy;
    if !gamma.is_finite() {
        gamma = 1.0;
    }
    state.gamma_k[0] = gamma;

    // write new sk, yk into current slot ind
    state.sks[ind].copy_from(&state.sk);
    state.yks[ind].copy_from(&state.yk);

    // ρ[ind] = 1 / (yᵀ s)
    let ys_curr = state.yks[ind].dot(&state.sks[ind]);
    state.rhos[ind] = 1.0 / ys_curr;

    // advance ring index
    state.ind[0] = (ind + 1) % m;
}

// scale_H0!(::LBFGSState) is a no-op in Julia
pub fn scale_h0_lbfgs(_state: &mut LBFGSState) {
    // nothing
}

// -----------------------------------------------------------------------------
// line search
// -----------------------------------------------------------------------------

// line_search(solver, f∇f!, p, fxk, gxk)
//
// This is specialized logic based on weak Wolfe conditions.
// We'll keep signature close but make types explicit.
//
// f_grad_f is a callback: given (grad_out, x, p_data) -> f(x)
// - it must fill grad_out with ∇f(x)
// - return f(x)
pub fn line_search(
    solver: &mut BFGSSolver,
    p_data: &mut dyn FGradF,
    fxk: f64,
    gxk: &DVector<f64>,
) -> f64 {
    // unpack aliases to state vectors
    let c1 = solver.c1;
    let c2 = solver.c2;
    let n = solver.n;

    // we need mutable access to internals
    let (xk_ref, pk_ref, xnext_ref, vn_ref) = match &mut solver.state {
        SolverStateEnum::Bfgs(st) => (&mut st.xk, &mut st.pk, &mut st.xnext, &mut st.vn),
        SolverStateEnum::Lbfgs(st) => (&mut st.xk, &mut st.pk, &mut st.xnext, &mut st.vn),
    };
    // rename for brevity
    let xk = xk_ref;
    let pk = pk_ref;
    let xnext = xnext_ref;
    let vn = vn_ref;

    // bounds on step alpha
    let lb0 = 0.0;
    // ub0 from feasibility: x + α p >= 0 => α <= -x[i]/p[i] for p[i] < 0
    let mut tmax_x = f64::INFINITY;
    for i in 0..n {
        if pk[i] < 0.0 {
            let cand = -xk[i] / pk[i];
            if cand < tmax_x {
                tmax_x = cand;
            }
        }
    }
    let ub0 = tmax_x - f64::EPSILON.sqrt();

    let mut lb = lb0;
    let mut ub = ub0;

    // directional derivative at 0: dh_0 = gᵀ p
    let mut dh0 = 0.0;
    for i in 0..n {
        dh0 += gxk[i] * pk[i];
    }

    // closure (hdh!) from Julia computes:
    // xnext = xk + α pk
    // check feasibility xnext >= sqrt(eps)
    // fnext = f_grad_f(vn, xnext, p_data)
    // h = fnext - fxk
    // dh = vn ⋅ pk
    // returns (h, dh)
    let mut hdh = |alpha: f64| -> (f64, f64) {
        // xnext = xk + α pk
        for i in 0..n {
            xnext[i] = xk[i] + alpha * pk[i];
        }

        // feasibility: xnext[i] >= sqrt(eps)
        let min_allowed = f64::EPSILON.sqrt();
        if xnext.iter().any(|&xi| xi < min_allowed) {
            return (f64::INFINITY, f64::INFINITY);
        }

        // vn <- grad at xnext, return f(xnext)
        let fnext = p_data.f_grad_f(vn, xnext);

        // h = f(xk+αp) - f(xk)
        let h = fnext - fxk;

        // dh = grad(xnext) ⋅ pk
        let mut dh_val = 0.0;
        for i in 0..n {
            dh_val += vn[i] * pk[i];
        }

        (h, dh_val)
    };

    // initial alpha guess
    let mut alpha_k = ub0.min(1.0);

    for _ in 0..20 {
        let (h, dh_val) = hdh(alpha_k);

        // Wolfe condition checks
        // if h >= c1 * dh0 * αk  => upper bound
        if h >= c1 * dh0 * alpha_k {
            ub = alpha_k;
        } else if dh_val <= c2 * dh0 {
            // else if dh(αk) <= c2 * dh(0) => increase lower bound
            lb = alpha_k;
        } else {
            // satisfies weak Wolfe, break
            break;
        }

        // next alpha
        alpha_k = if ub.is_finite() {
            0.5 * (lb + ub)
        } else {
            2.0 * lb
        };
    }

    alpha_k
}

// -----------------------------------------------------------------------------
// state update steps
// -----------------------------------------------------------------------------

pub fn update_x(solver: &mut BFGSSolver, alpha: f64) {
    match &mut solver.state {
        SolverStateEnum::Bfgs(st) => {
            // xprev = xk
            st.xprev.copy_from(&st.xk);

            // xk += α pk
            st.xk.axpy(alpha, &st.pk, 1.0);

            // sk = xk - xprev
            st.sk = &st.xk - &st.xprev;
        }
        SolverStateEnum::Lbfgs(st) => {
            st.xprev.copy_from(&st.xk);
            st.xk.axpy(alpha, &st.pk, 1.0);
            st.sk = &st.xk - &st.xprev;
        }
    }
}

pub fn converged(solver: &BFGSSolver, opts: &BFGSOptions) -> bool {
    solver.g_norm < opts.eps_g_norm
}

// -----------------------------------------------------------------------------
// solve loop (skeleton)
// -----------------------------------------------------------------------------

#[derive(Debug)]
pub enum SolverStatus {
    Optimal,
    IterationLimit,
    TimeLimit,
}

pub struct SolveResult {
    pub status: SolverStatus,
    pub obj_val: f64,
    pub g_norm: f64,
    pub x: DVector<f64>,
    pub log: BFGSLog,
}

trait FGradF {
    /// use trait to skip passing `Any` and a function to op on `Any` like in julia
    fn f_grad_f(&mut self, gk: &mut DVector<f64>, xk: &DVector<f64>) -> f64;
}

// solve!(...) in Julia. We'll call it `solve` and make it return `SolveResult`.
pub fn solve(
    solver: &mut BFGSSolver,
    p_data: &mut dyn FGradF,
    opts: &BFGSOptions,
    x0: Option<&DVector<f64>>,
    h0: Option<&DMatrix<f64>>,
    reset_solver_flag: bool,
) -> SolveResult {
    let n = solver.n;

    // logging buffer
    let mut tmp_log = if opts.logging {
        BFGSLog::temp_for_solver(solver, opts.max_iters + 1)
    } else {
        BFGSLog::empty(0, 0.0)
    };

    // print header
    if opts.verbose {
        let headers = print_headers();
        let header_fmt = header_format();
        let headers_owned: Vec<String> = headers.iter().map(|s| s.to_string()).collect();
        print_header_bfgs(&header_fmt, &headers_owned);
    }

    // init solver state
    if reset_solver_flag {
        solver.reset_solver();
    }

    if x0.is_some() || h0.is_some() {
        solver.initialize(x0, h0);
    } else {
        // xk .= one(T)  => fill with 1.0
        match &mut solver.state {
            SolverStateEnum::Bfgs(st) => st.xk.fill(1.0),
            SolverStateEnum::Lbfgs(st) => st.xk.fill(1.0),
        }
    }

    // timer start
    let t_start = Instant::now();

    // eval at x0
    let (xk, gk, gnext) = match &mut solver.state {
        SolverStateEnum::Bfgs(st) => (&mut st.xk, &mut st.gk, &mut st.gnext),
        SolverStateEnum::Lbfgs(st) => (&mut st.xk, &mut st.gk, &mut st.gnext),
    };

    solver.obj_val = p_data.f_grad_f(gk, xk);
    solver.g_norm = gk.norm();

    let t_now = t_start.elapsed().as_secs_f64();
    if opts.logging {
        tmp_log.populate(0, t_now, solver.obj_val, solver.g_norm, xk);
    }

    if opts.verbose {
        let iter_fmt = iter_format();
        let row = vec![
            format!("{}", 0),
            format!("{:.3e}", solver.obj_val),
            format!("{:.3e}", solver.g_norm),
            format!("{:.3}", t_now),
        ];
        print_iter(&iter_fmt, &row);
    }

    let mut k = 0usize;
    let mut status = SolverStatus::Optimal;

    while k < opts.max_iters && t_start.elapsed() < opts.max_time {
        k += 1;

        // direction
        match &mut solver.state {
            SolverStateEnum::Bfgs(st) => compute_search_direction_bfgs(st),
            SolverStateEnum::Lbfgs(st) => compute_search_direction_lbfgs(st),
        }

        // line search
        let alpha_k = {
            // need gxk = current gradient gk
            let gxk_clone = match &solver.state {
                SolverStateEnum::Bfgs(st) => st.gk.clone(),
                SolverStateEnum::Lbfgs(st) => st.gk.clone(),
            };
            line_search(solver, p_data, solver.obj_val, &gxk_clone)
        };

        // update x
        update_x(solver, alpha_k);

        // compute new obj + grad at updated x
        {
            let (xk2, gnext2) = match &mut solver.state {
                SolverStateEnum::Bfgs(st) => (&mut st.xk, &mut st.gnext),
                SolverStateEnum::Lbfgs(st) => (&mut st.xk, &mut st.gnext),
            };
            solver.obj_val = p_data.f_grad_f(gnext2, xk2);
        }

        // gradient bookkeeping
        match &mut solver.state {
            SolverStateEnum::Bfgs(st) => {
                // yk = gnext - gk
                st.yk = &st.gnext - &st.gk;
                // gk = gnext
                st.gk.copy_from(&st.gnext);
            }
            SolverStateEnum::Lbfgs(st) => {
                st.yk = &st.gnext - &st.gk;
                st.gk.copy_from(&st.gnext);
            }
        }

        // update norms
        let g_now = match &solver.state {
            SolverStateEnum::Bfgs(st) => &st.gk,
            SolverStateEnum::Lbfgs(st) => &st.gk,
        };
        solver.g_norm = g_now.norm();

        // scaling and Hessian/buffer update
        if k == 1 && h0.is_none() {
            match &mut solver.state {
                SolverStateEnum::Bfgs(st) => scale_h0_bfgs(st),
                SolverStateEnum::Lbfgs(st) => scale_h0_lbfgs(st),
            }
        }
        match &mut solver.state {
            SolverStateEnum::Bfgs(st) => update_hk_bfgs(st),
            SolverStateEnum::Lbfgs(st) => update_hk_lbfgs(st),
        }

        // log + print
        let t_now = t_start.elapsed().as_secs_f64();

        if opts.logging {
            tmp_log.populate(
                k,
                t_now,
                solver.obj_val,
                solver.g_norm,
                match &solver.state {
                    SolverStateEnum::Bfgs(st) => &st.xk,
                    SolverStateEnum::Lbfgs(st) => &st.xk,
                },
            );
        }

        if opts.verbose && (k == 1 || k % opts.print_iter == 0) {
            let iter_fmt = iter_format();
            let row = vec![
                format!("{}", k),
                format!("{:.3e}", solver.obj_val),
                format!("{:.3e}", solver.g_norm),
                format!("{:.3}", t_now),
            ];
            print_iter(&iter_fmt, &row);
        }

        // convergence?
        if converged(solver, opts) {
            status = SolverStatus::Optimal;
            break;
        }

        // stopping?
        if k >= opts.max_iters {
            status = SolverStatus::IterationLimit;
            break;
        }
        if t_start.elapsed() >= opts.max_time {
            status = SolverStatus::TimeLimit;
            break;
        }
    }

    let solve_time_sec = t_start.elapsed().as_secs_f64();

    // finalize log
    tmp_log.solve_time = solve_time_sec;
    tmp_log.num_iters = k;

    // final result x
    let x_final = solver.state.xk().clone();

    SolveResult {
        status,
        obj_val: solver.obj_val,
        g_norm: solver.g_norm,
        x: x_final,
        log: tmp_log,
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use nalgebra::{DMatrix, DVector};
    use rand::SeedableRng;
    use rand::prelude::*;

    fn is_optimal(res: &SolveResult, eps_g_norm: f64) -> bool {
        matches!(res.status, SolverStatus::Optimal) && res.g_norm < eps_g_norm
    }

    #[test]
    fn test_quadratic() {
        // parameters
        let g_norm_tol = 1e-6;
        let options = BFGSOptions {
            eps_g_norm: g_norm_tol,
            verbose: false,
            logging: false,
            max_iters: 1000,
            max_time: std::time::Duration::from_secs_f64(60.0),
            print_iter: 10,
            num_threads: num_cpus::get(),
            mu: 10.0,
            final_print: true,
        };

        // f(x) = ||x - v||² + cᵀx
        // ∇f(x) = 2(x - v) + c
        //
        // We'll generate random x*, c, and define v = x* + c/2,
        // so that grad=0 at x*.
        let n = 10;
        let mut rng = StdRng::seed_from_u64(1);

        let xstar_vec: Vec<f64> = (0..n).map(|_| 10.0 * rng.random::<f64>()).collect();
        let c_vec: Vec<f64> = (0..n).map(|_| 10.0 * rng.random::<f64>()).collect();

        let xstar = DVector::from_vec(xstar_vec.clone());
        let c = DVector::from_vec(c_vec.clone());

        let mut v_data = Vec::with_capacity(n);
        for i in 0..n {
            v_data.push(xstar[i] + c[i] / 2.0);
        }
        let v = DVector::from_vec(v_data);

        // pack p1 = (v, c) into a struct so we can hand it as &mut dyn Any
        struct P1 {
            v: DVector<f64>,
            c: DVector<f64>,
        }
        let mut p1 = P1 {
            v: v.clone(),
            c: c.clone(),
        };

        // f1∇f1!(g, x, p)
        // g = 2(x - v) + c
        // f = ||x - v||^2 + cᵀ x
        impl FGradF for P1 {
            fn f_grad_f(&mut self, g: &mut DVector<f64>, x: &DVector<f64>) -> f64 {
                let n = x.len();

                for i in 0..n {
                    g[i] = 2.0 * (x[i] - self.v[i]) + self.c[i];
                }

                // ||x - v||^2
                let mut fval = 0.0;
                for i in 0..n {
                    let d = x[i] - self.v[i];
                    fval += d * d;
                }
                // + cᵀ x
                fval += self.c.dot(x);

                fval
            }
        }

        // initial x0 = zeros(n)
        let x0 = DVector::zeros(n);

        // BFGS solver
        {
            let mut solver1 = BFGSSolver::new(
                n,
                Method::Bfgs,
                1e-4, // c1
                0.9,  // c2
            );
            let res1 = solve(&mut solver1, &mut p1, &options, Some(&x0), None, true);

            // @test isapprox(xstar, res1.x; atol=1e-6)
            for i in 0..n {
                assert!(
                    (res1.x[i] - xstar[i]).abs() <= 1e-6,
                    "x mismatch at {i}: got {}, expect {}",
                    res1.x[i],
                    xstar[i]
                );
            }

            // @test is_optimal(res1; eps_g_norm=g_norm_tol)
            assert!(
                is_optimal(&res1, g_norm_tol),
                "res1 not optimal: status {:?}, g_norm {}, tol {}",
                res1.status,
                res1.g_norm,
                g_norm_tol
            );
        }

        // L-BFGS solver
        {
            let mut solver_lbfgs = BFGSSolver::new(n, Method::Lbfgs { m: 10 }, 1e-4, 0.9);
            let res1_lbfgs = solve(&mut solver_lbfgs, &mut p1, &options, Some(&x0), None, true);

            assert!(
                is_optimal(&res1_lbfgs, g_norm_tol),
                "res1_lbfgs not optimal: status {:?}, g_norm {}, tol {}",
                res1_lbfgs.status,
                res1_lbfgs.g_norm,
                g_norm_tol
            );
        }
    }

    #[test]
    fn test_rosenbrock() {
        // parameters
        let g_norm_tol = 1e-6;
        let options = BFGSOptions {
            eps_g_norm: g_norm_tol,
            verbose: false,
            logging: false,
            max_iters: 1000,
            max_time: std::time::Duration::from_secs_f64(60.0),
            print_iter: 10,
            num_threads: num_cpus::get(),
            mu: 10.0,
            final_print: true,
        };

        // Rosenbrock:
        // f(x) = 100 (x2 - x1^2)^2 + (1 - x1)^2
        // grad:
        // g1 = -200 (x2 - x1^2) * 2 x1 - 2(1 - x1)
        // g2 =  200 (x2 - x1^2)

        struct Rosenbrock {}
        impl FGradF for Rosenbrock {
            fn f_grad_f(&mut self, g: &mut DVector<f64>, x: &DVector<f64>) -> f64 {
                assert!(x.len() == 2, "rosenbrock expects n=2");
                let x1 = x[0];
                let x2 = x[1];
                let t = x2 - x1 * x1;

                // gradient
                g[0] = -200.0 * t * (2.0 * x1) - 2.0 * (1.0 - x1);
                g[1] = 200.0 * t;

                // function
                100.0 * t * t + (1.0 - x1) * (1.0 - x1)
            }
        }

        let n = 2;
        let x0 = DVector::from_vec(vec![0.1, 0.1]);

        // BFGS
        {
            let mut solver2 = BFGSSolver::new(n, Method::Bfgs, 1e-4, 0.9);
            let res2 = solve(
                &mut solver2,
                &mut Rosenbrock {},
                &options,
                Some(&x0),
                None,
                true,
            );

            assert!(
                is_optimal(&res2, g_norm_tol),
                "res2 not optimal: status {:?}, g_norm {}, tol {}",
                res2.status,
                res2.g_norm,
                g_norm_tol
            );
        }

        // LBFGS
        {
            let mut solver2_lbfgs = BFGSSolver::new(n, Method::Lbfgs { m: 10 }, 1e-4, 0.9);
            let res2_lbfgs = solve(
                &mut solver2_lbfgs,
                &mut Rosenbrock {},
                &options,
                Some(&x0),
                None,
                true,
            );

            assert!(
                is_optimal(&res2_lbfgs, g_norm_tol),
                "res2_lbfgs not optimal: status {:?}, g_norm {}, tol {}",
                res2_lbfgs.status,
                res2_lbfgs.g_norm,
                g_norm_tol
            );
        }

        // iteration limit test
        {
            let options_iter_limit = BFGSOptions {
                max_iters: 2,
                verbose: false,
                logging: false,
                eps_g_norm: g_norm_tol,
                max_time: std::time::Duration::from_secs_f64(60.0),
                print_iter: 10,
                num_threads: num_cpus::get(),
                mu: 10.0,
                final_print: true,
            };

            let mut solver2_short = BFGSSolver::new(n, Method::Bfgs, 1e-4, 0.9);
            let res_limit = solve(
                &mut solver2_short,
                &mut Rosenbrock {},
                &options_iter_limit,
                Some(&x0),
                None,
                true,
            );

            match res_limit.status {
                SolverStatus::IterationLimit => {}
                other => panic!("expected IterationLimit, got {:?}", other),
            }
        }
    }
}
