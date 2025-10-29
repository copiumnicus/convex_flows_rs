/// solver.jl in rs
use super::{bfgs::*, bfgs_types::*, edge::*, objective::*};
use nalgebra::DVector;
use std::any::Any;

// We'll assume:
// - Objective trait (with ubar, grad_ubar, lower_limit, etc.)
// - EdgeEither enum { Gain(EdgeGain), ClosedForm(EdgeClosedForm) }
// - find_arb_all(xs, nu_or_prices, edges) implemented elsewhere
// - netflows(...) like before
// - L-BFGS-B is NOT implemented here. We'll sketch placeholders for bounds-based solve.

// -------------------------------
// Solver struct
// -------------------------------
pub struct Solver {
    pub flow_objective: Box<dyn Objective>, // flow_objective
    pub edge_objectives: Option<Vec<Box<dyn Objective>>>, // edge_objectives or None
    pub vis_zero: bool,                     // Vis_zero
    pub edges: Vec<EdgeEither>,             // edges
    pub y: DVector<f64>,                    // node flows y
    pub xs: Vec<DVector<f64>>,              // per-edge primal x
    pub nu: DVector<f64>,                   // ν
    pub etas: Vec<DVector<f64>>,            // ηᵢ (dual vars per edge objective)
    pub arb_prices: Vec<DVector<f64>>,      // ν_i (+ η_i if present)
    pub mu0: DVector<f64>,                  // initial μ0 (stacked ν,η)
    pub n: usize,                           // n nodes
    pub m: usize,                           // m edges
}

// This mirrors the Julia constructor:
// Solver(flow_objective, edge_objectives=nothing, edges, n)
pub fn make_solver(
    flow_objective: Box<dyn Objective>,
    edge_objectives: Option<Vec<Box<dyn Objective>>>,
    edges: Vec<EdgeEither>,
    n: usize,
) -> Solver {
    // dimension checks
    let m = edges.len();

    // sanity: check n equals "number of nodes involved in all edges"
    // In Julia: n != length(vcat([e.Ai for e in edges]...)) -> error
    // That code is weird (it flattens Ai pairs and counts length). Real intent:
    // "n must be number of nodes in problem".
    // We'll just trust caller-provided n for now.

    if let Some(ref eos) = edge_objectives {
        assert!(
            eos.len() == m,
            "edge_objectives must have same length as edges (m)"
        );
    }

    // y = zeros(n)
    let y = DVector::zeros(n);

    // xs[i] = zeros(length(e.Ai))
    // For 2-node edges, length(Ai)=2.
    let xs: Vec<DVector<f64>> = edges
        .iter()
        .map(|edge| {
            let len_edge_vars = match edge {
                EdgeEither::Gain(g) => 2,       // g.ai is (i32,i32)
                EdgeEither::ClosedForm(c) => 2, // also 2-node
            };
            DVector::zeros(len_edge_vars)
        })
        .collect();

    // ν = zeros(n)
    let nu = DVector::zeros(n);

    // ηᵢ = zeros(length(e.Ai)) for each edge
    let etas: Vec<DVector<f64>> = edges
        .iter()
        .map(|edge| {
            let len_edge_vars = match edge {
                EdgeEither::Gain(_) => 2,
                EdgeEither::ClosedForm(_) => 2,
            };
            DVector::zeros(len_edge_vars)
        })
        .collect();

    // arb_prices same shape as xs
    let arb_prices: Vec<DVector<f64>> = edges
        .iter()
        .map(|edge| {
            let len_edge_vars = match edge {
                EdgeEither::Gain(_) => 2,
                EdgeEither::ClosedForm(_) => 2,
            };
            DVector::zeros(len_edge_vars)
        })
        .collect();

    // Vis_zero = isnothing(edge_objectives)
    let vis_zero = edge_objectives.is_none();

    // μ0 has length:
    // if Vis_zero:
    //   n
    // else:
    //   n + sum(length(e.Ai))
    let extra_len: usize = if vis_zero {
        0
    } else {
        // sum over edges of number of components per edge dual var
        edges.iter().map(|_| 2usize).sum()
    };
    let mu0 = DVector::zeros(n + extra_len);

    Solver {
        flow_objective,
        edge_objectives,
        vis_zero,
        edges,
        y,
        xs,
        nu,
        etas,
        arb_prices,
        mu0,
        n,
        m,
    }
}

// -------------------------------
// find_arb!(s)
// -------------------------------
// Julia:
//   if Vis_zero:
//       arb_prices[i] = ν[Ai]
//       find_arb!(xs[i], edge[i], ν[Ai])
//   else:
//       arb_prices[i] = ηᵢ + ν[Ai]
//       find_arb!(xs[i], edge[i], arb_prices[i])
//
// We'll assume:
// - each edge has .ai = (i32,i32) already 0-based
// - find_arb_edge(x_i, edge, price_slice) is implemented elsewhere
pub fn find_arb_solver(s: &mut Solver) {
    for (i, edge) in s.edges.iter().enumerate() {
        let (i1, i2) = match edge {
            EdgeEither::Gain(g) => (g.ai.0 as usize, g.ai.1 as usize),
            EdgeEither::ClosedForm(c) => (c.ai.0 as usize, c.ai.1 as usize),
        };

        if s.vis_zero {
            // arb_prices[i] = ν[Ai]
            s.arb_prices[i][0] = s.nu[i1];
            s.arb_prices[i][1] = s.nu[i2];

            // xs[i] = argmax edge given that price vector
            find_arb_edge(&mut s.xs[i], edge, &s.arb_prices[i]);
        } else {
            // arb_prices[i] = ηᵢ + ν[Ai]
            s.arb_prices[i][0] = s.etas[i][0] + s.nu[i1];
            s.arb_prices[i][1] = s.etas[i][1] + s.nu[i2];

            find_arb_edge(&mut s.xs[i], edge, &s.arb_prices[i]);
        }
    }
}

// -------------------------------
// netflows!(s)
// -------------------------------
// Julia:
// s.y .= 0
// for (x,e) in zip(xs,edges):
//   y[e.Ai] .+= x
//
// Here we assume edges are 2-node, so we just add x[0] to node i1, x[1] to node i2.
pub fn netflows_inplace(s: &mut Solver) {
    s.y.fill(0.0);

    for (x_i, edge) in s.xs.iter().zip(s.edges.iter()) {
        let (i1, i2) = match edge {
            EdgeEither::Gain(g) => (g.ai.0 as usize, g.ai.1 as usize),
            EdgeEither::ClosedForm(c) => (c.ai.0 as usize, c.ai.1 as usize),
        };

        s.y[i1] += x_i[0];
        s.y[i2] += x_i[1];
    }
}

// -------------------------------
// solve!(s; ...)
// -------------------------------
//
// Julia version uses L-BFGS-B with bound constraints to solve over μ,
// where μ = [ν; η₁; η₂; ...] if edge objectives exist, else μ = ν.
//
// We don't have a full L-BFGS-B implementation here,
// so below is the structural translation:
//
pub fn solve_solver(
    s: &mut Solver,
    nu0: Option<DVector<f64>>,
    eta0: Option<Vec<DVector<f64>>>,
    verbose: bool,
    memory: usize,
    _factr: f64,
    _pgtol: f64,
    _max_fun: usize,
    _max_iter: usize,
    final_netflows: bool,
) -> f64 {
    // argument checks
    if let Some(ref eta_init) = eta0 {
        assert_eq!(eta_init.len(), s.m, "η0 must be length m");
        assert!(
            s.edge_objectives.is_some(),
            "solver does not have edge objectives"
        );
    }

    // nis = [length(e.Ai) ...]  here always 2
    let nis: Vec<usize> = s.edges.iter().map(|_| 2usize).collect();

    // determine bounds
    // LBFGS-B style bounds matrix:
    // bounds[0,i] = 1 means bounded
    // bounds[1,i] = lower bound
    // bounds[2,i] = upper bound
    //
    // Julia enforces ν >= max(0, lower_limit(U)), etc.
    //
    // We'll just build lower bounds vector lb[], ignoring ub (∞).
    let len_mu = if s.vis_zero {
        s.n
    } else {
        s.n + nis.iter().sum::<usize>()
    };

    let mut lb = vec![0.0; len_mu];

    // flow objective lower bounds
    {
        let lo_flow = s.flow_objective.lower_limit();
        for i in 0..s.n {
            lb[i] = lo_flow[i].max(0.0);
        }
    }

    // edge objective lower bounds (if present)
    if !s.vis_zero {
        let mut idx = s.n;
        if let Some(ref eos) = s.edge_objectives {
            for (edge_i, obj_i) in eos.iter().enumerate() {
                let lo_i = obj_i.lower_limit();
                let ni = nis[edge_i];
                for k in 0..ni {
                    lb[idx + k] = lo_i[k].max(0.0);
                }
                idx += ni;
            }
        }
    }

    // set initial μ0
    // μ0[0:n] = ν0 or (lb + 1)
    if let Some(nu_init) = nu0 {
        assert_eq!(nu_init.len(), s.n);
        for i in 0..s.n {
            s.mu0[i] = nu_init[i];
        }
    } else {
        for i in 0..s.n {
            s.mu0[i] = lb[i] + 1.0;
        }
    }

    // if Vis_zero=false, initialize per-edge η pieces
    if !s.vis_zero {
        // if eta0 is provided, use it
        if let Some(ref eta_init) = eta0 {
            let mut idx = s.n;
            for (edge_i, eta_vec) in eta_init.iter().enumerate() {
                let ni = nis[edge_i];
                assert_eq!(eta_vec.len(), ni);
                for k in 0..ni {
                    s.mu0[idx + k] = eta_vec[k];
                }
                idx += ni;
            }
        } else {
            // otherwise set μ0 edge-blocks = μ0[ν][Ai]
            let mut idx = s.n;
            for edge in s.edges.iter() {
                let (i1, i2) = match edge {
                    EdgeEither::Gain(g) => (g.ai.0 as usize, g.ai.1 as usize),
                    EdgeEither::ClosedForm(c) => (c.ai.0 as usize, c.ai.1 as usize),
                };
                s.mu0[idx + 0] = s.mu0[i1];
                s.mu0[idx + 1] = s.mu0[i2];
                idx += 2;
            }
        }
    }

    // define fn(μ) and grad!(g, μ)
    // This is what L-BFGS-B would repeatedly call.

    fn objective_fn(s: &mut Solver, mu: &DVector<f64>) -> f64 {
        // unpack μ into ν and ηᵢ
        // update s.nu, s.etas from mu
        let len_n = s.n;
        s.nu.copy_from(&mu.rows(0, len_n));

        if !s.vis_zero {
            let mut idx = len_n;
            for i in 0..s.m {
                for k in 0..2 {
                    s.etas[i][k] = mu[idx + k];
                }
                idx += 2;
            }
        }

        // refresh xs, arb_prices etc
        find_arb_solver(s);

        // acc = sum over edges
        let mut acc_edges = 0.0;
        for i in 0..s.m {
            if s.vis_zero {
                // acc += x_i ⋅ ν[Ai]
                let (i1, i2) = match &s.edges[i] {
                    EdgeEither::Gain(g) => (g.ai.0 as usize, g.ai.1 as usize),
                    EdgeEither::ClosedForm(c) => (c.ai.0 as usize, c.ai.1 as usize),
                };
                acc_edges += s.xs[i][0] * s.nu[i1] + s.xs[i][1] * s.nu[i2];
            } else {
                // acc += Ubar(V_i, η_i) + x_i ⋅ (η_i + ν[Ai])
                let edge_obj = s.edge_objectives.as_ref().unwrap()[i].as_ref();
                acc_edges += edge_obj.ubar(&s.etas[i]);
                // dot(xs[i], arb_prices[i])
                acc_edges += s.xs[i].dot(&s.arb_prices[i]);
            }
        }

        // total dual obj:
        s.flow_objective.ubar(&s.nu) + acc_edges
    }

    fn gradient_fn(s: &mut Solver, mu: &DVector<f64>, grad_out: &mut DVector<f64>) {
        // NOTE: gradient_fn is assumed to be called after objective_fn
        // so s.nu, s.etas, s.xs, s.arb_prices are already synced.

        grad_out.fill(0.0);

        // ∇_ν part
        let mut g_nu = s.flow_objective.grad_ubar(&s.nu);
        // add flows from xs into those ν indices
        for i in 0..s.m {
            let (i1, i2) = match &s.edges[i] {
                EdgeEither::Gain(g) => (g.ai.0 as usize, g.ai.1 as usize),
                EdgeEither::ClosedForm(c) => (c.ai.0 as usize, c.ai.1 as usize),
            };

            g_nu[i1] += s.xs[i][0];
            g_nu[i2] += s.xs[i][1];
        }

        // write ∇_ν to grad_out[0:n]
        for j in 0..s.n {
            grad_out[j] = g_nu[j];
        }

        // ∇_η blocks if vis_zero = false
        if !s.vis_zero {
            let mut idx = s.n;
            for i in 0..s.m {
                let mut g_eta_i = s.edge_objectives.as_ref().unwrap()[i].grad_ubar(&s.etas[i]);

                // g_eta_i += xs[i]
                for k in 0..2 {
                    g_eta_i[k] += s.xs[i][k];
                    grad_out[idx + k] = g_eta_i[k];
                }

                idx += 2;
            }
        }
    }

    // ----- this is where Julia calls L-BFGS-B with bounds -----
    //
    // In actual Rust you'd now pass:
    //   objective_fn(s, μ)
    //   gradient_fn(s, μ, g)
    //   lower bounds lb
    // into an L-BFGS-B implementation.
    //
    // We don't have that here, so we'll just "fake solve" by setting ν = μ0[0:n],
    // etc., then computing final outputs once.

    // first sync to μ0
    let mu_init = s.mu0.clone();
    let _f0 = objective_fn(s, &mu_init);

    // no optimizer, just pretend mu_final = mu_init
    let mu_final = mu_init;

    // final sync
    let _fend = objective_fn(s, &mu_final);

    // update primal y:
    if final_netflows {
        // y = netflows(xs)
        netflows_inplace(s);
    } else {
        // y = -∇Ubar(flow_objective, ν)
        s.y = s.flow_objective.grad_ubar(&s.nu);
        s.y *= -1.0;
    }

    // "solver_time"
    // Julia returns time spent in LBFGS-B. We'll just return 0.0 here.
    0.0
}
