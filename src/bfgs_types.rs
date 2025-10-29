/// bfgs/types.jl in rs
use nalgebra::{DMatrix, DVector};
use std::time::Duration;

pub trait SolverState {
    fn reset_state(&mut self);
    fn initialize_state(&mut self, x0: Option<&DVector<f64>>, h0: Option<&DMatrix<f64>>);
    fn xk(&self) -> &DVector<f64>;
    fn xk_mut(&mut self) -> &mut DVector<f64>;
    fn gk(&self) -> &DVector<f64>;
    fn gk_mut(&mut self) -> &mut DVector<f64>;
    fn dim(&self) -> usize;
}

/* ============================
BFGSState
============================ */

pub struct BFGSState {
    pub xk: DVector<f64>,    // current iterate
    pub xprev: DVector<f64>, // previous iterate
    pub xnext: DVector<f64>, // next iterate
    pub pk: DVector<f64>,    // search direction
    pub sk: DVector<f64>,    // step
    pub gk: DVector<f64>,    // gradient at current iterate
    pub gnext: DVector<f64>, // gradient at next iterate
    pub yk: DVector<f64>,    // gradient difference
    pub hk: DMatrix<f64>,    // inverse Hessian approximation
    pub vn: DVector<f64>,    // cache vector
}

impl BFGSState {
    pub fn new(n: usize) -> Self {
        Self {
            xk: DVector::zeros(n),
            xprev: DVector::zeros(n),
            xnext: DVector::zeros(n),
            pk: DVector::zeros(n),
            sk: DVector::zeros(n),
            gk: DVector::zeros(n),
            gnext: DVector::zeros(n),
            yk: DVector::zeros(n),
            hk: DMatrix::identity(n, n),
            vn: DVector::zeros(n),
        }
    }
}

impl SolverState for BFGSState {
    fn reset_state(&mut self) {
        let n = self.dim();

        self.xk.fill(0.0);
        self.xprev.fill(0.0);
        self.pk.fill(0.0);
        self.sk.fill(0.0);

        self.gk.fill(0.0);
        self.gnext.fill(0.0);
        self.yk.fill(0.0);

        self.hk.fill(0.0);
        for i in 0..n {
            self.hk[(i, i)] = 1.0;
        }
    }

    fn initialize_state(&mut self, x0: Option<&DVector<f64>>, h0: Option<&DMatrix<f64>>) {
        if let Some(x0v) = x0 {
            self.xk.copy_from(x0v);
        }
        if let Some(h0m) = h0 {
            self.hk.copy_from(h0m);
        }
    }

    fn xk(&self) -> &DVector<f64> {
        &self.xk
    }
    fn xk_mut(&mut self) -> &mut DVector<f64> {
        &mut self.xk
    }

    fn gk(&self) -> &DVector<f64> {
        &self.gk
    }
    fn gk_mut(&mut self) -> &mut DVector<f64> {
        &mut self.gk
    }

    fn dim(&self) -> usize {
        self.xk.len()
    }
}

/* ============================
LBFGSState
============================ */

pub struct LBFGSState {
    pub ind: Vec<usize>, // index of current iterate (Julia stored [1])
    pub xk: DVector<f64>,
    pub xprev: DVector<f64>,
    pub xnext: DVector<f64>,
    pub pk: DVector<f64>,
    pub sk: DVector<f64>,
    pub gk: DVector<f64>,
    pub gnext: DVector<f64>,
    pub yk: DVector<f64>,
    pub sks: Vec<DVector<f64>>, // history: step
    pub yks: Vec<DVector<f64>>, // history: gradient diff
    pub alphas: DVector<f64>,   // αs cache for Hessian mult
    pub betas: DVector<f64>,    // βs cache for Hessian mult
    pub rhos: DVector<f64>,     // ρs cache for Hessian mult
    pub gamma_k: DVector<f64>,  // scaling factor for Hₖ⁰ (stored as length-1 vec)
    pub vn: DVector<f64>,       // cache vector
}

impl LBFGSState {
    pub fn new(n: usize, m: usize) -> Self {
        Self {
            ind: vec![0], // Julia starts at 1, Rust 0
            xk: DVector::zeros(n),
            xprev: DVector::zeros(n),
            xnext: DVector::zeros(n),
            pk: DVector::zeros(n),
            sk: DVector::zeros(n),
            gk: DVector::zeros(n),
            gnext: DVector::zeros(n),
            yk: DVector::zeros(n),
            sks: (0..m).map(|_| DVector::zeros(n)).collect(),
            yks: (0..m).map(|_| DVector::zeros(n)).collect(),
            alphas: DVector::zeros(m),
            betas: DVector::zeros(m),
            rhos: DVector::zeros(m),
            gamma_k: DVector::from_vec(vec![1.0]),
            vn: DVector::zeros(n),
        }
    }
}

impl SolverState for LBFGSState {
    fn reset_state(&mut self) {
        let n = self.dim();

        self.xk.fill(0.0);
        self.pk.fill(0.0);

        self.gk.fill(0.0);
        self.gnext.fill(0.0);

        for v in &mut self.sks {
            v.fill(0.0);
        }
        for v in &mut self.yks {
            v.fill(0.0);
        }

        self.gamma_k[0] = 1.0;

        // yk, sk, pk, etc are already zeroed or explicitly filled above
        // we don't bother touching alphas/betas/rhos here; Julia didn't either
        // (they're scratch used in two-loop recursion)
        for i in 0..self.alphas.len() {
            self.alphas[i] = 0.0;
            self.betas[i] = 0.0;
            self.rhos[i] = 0.0;
        }

        self.vn.fill(0.0);
        self.sk.fill(0.0);
        self.yk.fill(0.0);
        self.xprev.fill(0.0);
        self.xnext.fill(0.0);
    }

    fn initialize_state(&mut self, x0: Option<&DVector<f64>>, h0: Option<&DMatrix<f64>>) {
        if let Some(x0v) = x0 {
            self.xk.copy_from(x0v);
        }
        if let Some(h0v) = h0 {
            // Julia: state.γk[1] = H0
            // In L-BFGS they store a scalar for initial Hessian scaling.
            // We'll assume `h0` is 1x1 here or a diagonal scale guess.
            // If you want full control, change signature.
            assert!(
                h0v.nrows() == 1 && h0v.ncols() == 1,
                "LBFGS initialize_state!: expected H0 to be 1x1 scaling"
            );
            self.gamma_k[0] = h0v[(0, 0)];
        }
    }

    fn xk(&self) -> &DVector<f64> {
        &self.xk
    }
    fn xk_mut(&mut self) -> &mut DVector<f64> {
        &mut self.xk
    }

    fn gk(&self) -> &DVector<f64> {
        &self.gk
    }
    fn gk_mut(&mut self) -> &mut DVector<f64> {
        &mut self.gk
    }

    fn dim(&self) -> usize {
        self.xk.len()
    }
}

/* ============================
SolverStateEnum
============================ */

pub enum SolverStateEnum {
    Bfgs(BFGSState),
    Lbfgs(LBFGSState),
}

impl SolverStateEnum {
    pub fn reset_state(&mut self) {
        match self {
            SolverStateEnum::Bfgs(s) => s.reset_state(),
            SolverStateEnum::Lbfgs(s) => s.reset_state(),
        }
    }

    pub fn initialize_state(&mut self, x0: Option<&DVector<f64>>, h0: Option<&DMatrix<f64>>) {
        match self {
            SolverStateEnum::Bfgs(s) => s.initialize_state(x0, h0),
            SolverStateEnum::Lbfgs(s) => s.initialize_state(x0, h0),
        }
    }

    pub fn xk(&self) -> &DVector<f64> {
        match self {
            SolverStateEnum::Bfgs(s) => s.xk(),
            SolverStateEnum::Lbfgs(s) => s.xk(),
        }
    }

    pub fn xk_mut(&mut self) -> &mut DVector<f64> {
        match self {
            SolverStateEnum::Bfgs(s) => s.xk_mut(),
            SolverStateEnum::Lbfgs(s) => s.xk_mut(),
        }
    }

    pub fn gk(&self) -> &DVector<f64> {
        match self {
            SolverStateEnum::Bfgs(s) => s.gk(),
            SolverStateEnum::Lbfgs(s) => s.gk(),
        }
    }

    pub fn gk_mut(&mut self) -> &mut DVector<f64> {
        match self {
            SolverStateEnum::Bfgs(s) => s.gk_mut(),
            SolverStateEnum::Lbfgs(s) => s.gk_mut(),
        }
    }

    pub fn dim(&self) -> usize {
        match self {
            SolverStateEnum::Bfgs(s) => s.dim(),
            SolverStateEnum::Lbfgs(s) => s.dim(),
        }
    }
}

/* ============================
BFGSSolver
============================ */

pub struct BFGSSolver {
    pub n: usize,
    pub state: SolverStateEnum,
    pub obj_val: f64,
    pub g_norm: f64,
    pub cs_norm: f64,
    pub c1: f64, // Wolfe condition 1
    pub c2: f64, // Wolfe condition 2
}

pub enum Method {
    Bfgs,
    Lbfgs { m: usize },
}

impl BFGSSolver {
    pub fn new(n: usize, method: Method, c1: f64, c2: f64) -> Self {
        let state = match method {
            Method::Bfgs => SolverStateEnum::Bfgs(BFGSState::new(n)),
            Method::Lbfgs { m } => SolverStateEnum::Lbfgs(LBFGSState::new(n, m)),
        };

        Self {
            n,
            state,
            obj_val: 0.0,
            g_norm: 0.0,
            cs_norm: 0.0,
            c1,
            c2,
        }
    }

    pub fn reset_solver(&mut self) {
        self.state.reset_state();
    }

    pub fn initialize(&mut self, x0: Option<&DVector<f64>>, h0: Option<&DMatrix<f64>>) {
        self.state.initialize_state(x0, h0);
    }
}

/* ============================
BFGSOptions
============================ */

pub struct BFGSOptions {
    pub max_iters: usize,
    pub max_time: Duration,
    pub print_iter: usize,
    pub verbose: bool,
    pub logging: bool,
    pub eps_g_norm: f64,
    pub num_threads: usize,
    pub mu: f64,
    pub final_print: bool,
}

impl Default for BFGSOptions {
    fn default() -> Self {
        Self {
            max_iters: 1000,
            max_time: Duration::from_secs_f64(60.0),
            print_iter: 10,
            verbose: true,
            logging: true,
            eps_g_norm: 1e-6,
            num_threads: num_cpus::get(),
            mu: 10.0,
            final_print: true,
        }
    }
}

/* ============================
BFGSLog
============================ */

pub struct BFGSLog {
    pub fx: Option<DVector<f64>>,
    pub g_norm: Option<DVector<f64>>,
    pub xk: Option<Vec<DVector<f64>>>,
    pub iter_time: Option<DVector<f64>>,
    pub num_iters: usize,
    pub solve_time: f64,
}

impl BFGSLog {
    pub fn empty(num_iters: usize, solve_time: f64) -> Self {
        Self {
            fx: None,
            g_norm: None,
            xk: None,
            iter_time: None,
            num_iters,
            solve_time,
        }
    }

    pub fn temp_for_solver(solver: &BFGSSolver, max_iters: usize) -> Self {
        let n = solver.n;

        Self {
            fx: Some(DVector::zeros(max_iters)),
            g_norm: Some(DVector::zeros(max_iters)),
            xk: Some((0..max_iters).map(|_| DVector::zeros(n)).collect()),
            iter_time: Some(DVector::zeros(max_iters)),
            num_iters: 0,
            solve_time: 0.0,
        }
    }

    pub fn populate(
        &mut self,
        k: usize,
        time_sec: f64,
        obj_val: f64,
        g_norm: f64,
        xk: &DVector<f64>,
    ) {
        if let Some(x_store) = &mut self.xk {
            x_store[k].copy_from(xk);
        }
        if let Some(it) = &mut self.iter_time {
            it[k] = time_sec;
        }
        if let Some(fx) = &mut self.fx {
            fx[k] = obj_val;
        }
        if let Some(gn) = &mut self.g_norm {
            gn[k] = g_norm;
        }
        self.num_iters = k + 1;
    }
}

/* ============================
BFGSResult
============================ */

pub struct BFGSResult {
    pub status: SolverStatus,
    pub obj_val: f64,
    pub g_norm: f64,
    pub x: DVector<f64>,
    pub log: BFGSLog,
}

pub enum SolverStatus {
    Optimal,
    NotConverged,
    Error,
}
