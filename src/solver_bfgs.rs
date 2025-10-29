use super::{bfgs::*, bfgs_types::*, edge::*, objective::*};
/// solver_bfgs.jl in rs
use nalgebra::DVector;
use std::any::Any;

pub struct ConvexFlowProblemTwoNode {
    pub obj: Box<dyn Objective>, // U in Julia
    pub edges: Vec<EdgeEither>,  // edges::Vector{<: Edge}
    pub y: DVector<f64>,         // primal flow per node
    pub xs: Vec<DVector<f64>>,   // per-edge solution x = [-w, h(w)]
    pub nu: DVector<f64>,        // dual vars ν
    pub n: usize,                // number of nodes
    pub m: usize,                // number of edges
}

// problem(obj=obj, edges=edges)
pub fn make_problem(obj: Box<dyn Objective>, edges: Vec<EdgeEither>) -> ConvexFlowProblemTwoNode {
    let n = obj.len();
    let m = edges.len();

    // y = zeros(n)
    let y = DVector::zeros(n);

    // xs = [zeros(2) for e in edges]
    let xs = (0..m).map(|_| DVector::zeros(2)).collect::<Vec<_>>();

    // ν = zeros(n)
    let nu = DVector::zeros(n);

    ConvexFlowProblemTwoNode {
        obj,
        edges,
        y,
        xs,
        nu,
        n,
        m,
    }
}

// netflows(xs, edges, n)
// sum edge contributions into node flow vector
pub fn netflows(xs: &[DVector<f64>], edges: &[EdgeEither], n: usize) -> DVector<f64> {
    let mut ret = DVector::zeros(n);
    for (x, e) in xs.iter().zip(edges.iter()) {
        let (i1, i2) = match e {
            EdgeEither::Gain(g) => g.ai,
            EdgeEither::ClosedForm(c) => c.ai,
        };
        // Julia is 1-based, Rust is 0-based.
        // The Julia code assumes Ai = (node_i1, node_i2) already valid indices.
        // We assume here we've already converted them to 0-based when building edges.
        ret[i1] += x[0];
        ret[i2] += x[1];
    }
    ret
}

// solve!(problem; options=..., method=:bfgs, memory=10)
pub fn solve_problem(
    problem: &mut ConvexFlowProblemTwoNode,
    maybe_options: Option<BFGSOptions>,
    method: Method, // Method::Bfgs or Method::Lbfgs { m: memory }
) -> SolveResult {
    let n = problem.n;
    let options = maybe_options.unwrap_or_default();

    // problem.ν .= 1.0
    problem.nu.fill(1.0);

    // We need to build the dual objective:
    //
    // f∇f!(g, ν, xs, obj, edges) returns:
    //   acc = Ubar(obj, ν)
    //   g   = ∇Ubar(obj, ν)
    //   find_arb!(xs, ν, edges)   (fills per-edge arbitrage x for given ν)
    //   for each edge i with nodes (i1,i2):
    //       acc += x[i][1]*ν[i1] + x[i][2]*ν[i2]
    //       g[i1] += x[i][1]
    //       g[i2] += x[i][2]
    //
    // We'll capture problem.xs, problem.obj, problem.edges by mutable reference
    // using an adapter struct so we can pass &mut dyn Any to `solve`.

    struct DualProblemCtx<'a> {
        xs: &'a mut [DVector<f64>],
        obj: &'a mut Box<dyn Objective>,
        edges: &'a [EdgeEither],
    }

    // callback that matches: Fn(&mut DVector<f64>, &DVector<f64>, &mut dyn Any) -> f64
    fn dual_obj_grad(
        fa: &FindArb,
        g: &mut DVector<f64>, // gradient output (this is ∇ w.r.t ν)
        nu: &DVector<f64>,    // current ν
        p_any: &mut dyn Any,  // we downcast to DualProblemCtx
    ) -> f64 {
        // downcast
        let p = p_any
            .downcast_mut::<DualProblemCtx>()
            .expect("bad payload for dual_obj_grad");

        let n = nu.len();
        let m = p.edges.len();

        // acc = Ubar(obj, ν)
        let mut acc = p.obj.ubar(nu);

        // g = ∇Ubar(obj, ν)
        let grad_vec = p.obj.grad_ubar(nu);
        g.copy_from(&grad_vec);

        // find_arb!(xs, ν, edges)
        // fills xs[i] = [-w*, h(w*)] given nu ratio etc.
        // (you already implemented find_arb_all in Rust)
        fa.find_all(p.xs, nu, p.edges);

        // sum over edges
        for (edge_idx, edge) in p.edges.iter().enumerate() {
            let (i1, i2) = match edge {
                EdgeEither::Gain(gain) => gain.ai,
                EdgeEither::ClosedForm(cf) => cf.ai,
            };

            let x_i = &p.xs[edge_idx];
            // x_i[0] ↔ xs[i][1] in Julia (1-based); x_i[1] ↔ xs[i][2]
            acc += x_i[0] * nu[i1] + x_i[1] * nu[i2];

            g[i1] += x_i[0];
            g[i2] += x_i[1];
        }

        acc
    }

    // build context for the callback
    let mut ctx = DualProblemCtx {
        xs: &mut problem.xs,
        obj: &mut problem.obj,
        edges: &problem.edges,
    };

    // make solver
    let mut solver = BFGSSolver::new(
        n, method, 1e-4, // c1
        0.9,  // c2
    );

    // run solve(...) on the dual in ν-space
    let mut ctx_any: &mut dyn Any = &mut ctx;
    let result = solve(
        &mut solver,
        dual_obj_grad,
        &mut ctx_any,
        &options,
        Some(&problem.nu),
        None,
        true,
    );

    // update problem.ν .= result.x
    problem.nu.copy_from(&result.x);

    // ∇Ubar!(problem.y, problem.U, problem.ν)
    // in Julia: grad goes into y, then y .= -y
    {
        let grad_nu = problem.obj.grad_ubar(&problem.nu);
        problem.y.copy_from(&grad_nu);
        problem.y *= -1.0;
    }

    // yhat = netflows(problem.xs, problem.edges, problem.n)
    let yhat = netflows(&problem.xs, &problem.edges, problem.n);

    // primal feasibility residual: ||y - yhat||
    let mut diff = &problem.y - &yhat;
    let pres = diff.norm();

    if options.verbose {
        eprintln!("\nDual problem solve status:");
    }
    if options.final_print {
        // mimic display(result): dump summary-ish
        eprintln!(
            "status: {:?}\nobj_val: {:.6e}\ng_norm: {:.6e}\niterations: {}\n",
            result.status, result.obj_val, result.g_norm, result.log.num_iters
        );
    }
    if options.verbose {
        eprintln!("Primal feasibility ||y - Σ Aᵢ xᵢ||: {:.4e}", pres);
    }

    result
}
