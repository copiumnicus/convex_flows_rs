/// objectives.jl
use nalgebra::{Cholesky, DMatrix, DVector};
use std::f64;

pub trait Objective {
    fn len(&self) -> usize;

    /// Evaluates the net flow utility function `objective` at `y`.
    fn u(&self, y: &DVector<f64>) -> f64;

    /// Returns the gradient of the net flow utility function `objective` at `y`.
    fn grad_u(&self, _y: &DVector<f64>) -> Vec<f64> {
        unimplemented!("grad_U not implemented for this objective")
    }

    /// Evaluates the 'conjugate' of the net flow utility function `objective` at `ν`.
    fn ubar(&self, nu: &DVector<f64>) -> f64;

    fn grad_ubar(&self, nu: &DVector<f64>) -> DVector<f64>;

    /// Componentwise lower bound on argument `ν` for objective [`Ubar`](@ref).  
    /// Returns a vector with length `length(ν)` (number of nodes).
    fn lower_limit(&self) -> Vec<f64> {
        // sqrt(eps()) ~ sqrt(machine epsilon)
        // f64::EPSILON is 2^-52 ≈ 2.22e-16, sqrt ≈ 1.49e-8
        let v = f64::EPSILON.sqrt();
        vec![v; self.len()]
    }

    /// Componentwise upper bound on argument `ν` for objective [`Ubar`](@ref).  
    /// Returns a vector with length `length(ν)` (number of nodes).
    fn upper_limit(&self) -> Vec<f64> {
        vec![f64::INFINITY; self.len()]
    }
}

pub struct NonpositiveQuadratic {
    /// n = dimension
    pub n: usize,
    /// a = scaling vector (a[i] > 0)
    pub a: DVector<f64>,
    /// b = shift vector
    pub b: DVector<f64>,
}

impl NonpositiveQuadratic {
    pub fn new(b: DVector<f64>, a: Option<DVector<f64>>) -> Self {
        let n = b.len();
        let a = a.unwrap_or_else(|| DVector::from_element(n, 1.0));
        Self { n, a, b }
    }
}

impl Objective for NonpositiveQuadratic {
    fn len(&self) -> usize {
        self.n
    }

    // U(y) = -0.5 * sum( max( sqrt(a_i)*b_i - y_i, 0 )^2 )
    fn u(&self, y: &DVector<f64>) -> f64 {
        let mut acc = 0.0;
        for i in 0..self.n {
            let term = self.a[i].sqrt() * self.b[i] - y[i];
            let clipped = term.max(0.0);
            acc += clipped * clipped;
        }
        -0.5 * acc
    }

    // Ubar(ν) = 0.5 * sum((ν_i / sqrt(a_i))^2) - b ⋅ ν
    fn ubar(&self, nu: &DVector<f64>) -> f64 {
        let mut quad = 0.0;
        let mut lin = 0.0;
        for i in 0..self.n {
            let s = nu[i] / self.a[i].sqrt();
            quad += s * s;
            lin += self.b[i] * nu[i];
        }
        0.5 * quad - lin
    }

    // ∇Ubar(ν) = ν ./ a - b
    fn grad_ubar(&self, nu: &DVector<f64>) -> DVector<f64> {
        let mut g = DVector::zeros(self.n);
        for i in 0..self.n {
            g[i] = nu[i] / self.a[i] - self.b[i];
        }
        g
    }
}

pub struct Markowitz {
    /// μ  = expected returns vector
    pub mu: DVector<f64>,
    /// Σ  = covariance matrix
    pub sigma: DMatrix<f64>,
}

impl Markowitz {
    pub fn new(mu: DVector<f64>, sigma: DMatrix<f64>) -> Self {
        Self { mu, sigma }
    }
    // solve Σ x = rhs
    fn solve_sigma(&self, rhs: &DVector<f64>) -> DVector<f64> {
        // Cholesky expects symmetric positive-definite
        let chol = Cholesky::new(self.sigma.clone()).expect("Σ not SPD");
        chol.solve(rhs)
    }
}

impl Objective for Markowitz {
    fn len(&self) -> usize {
        self.mu.len()
    }
    // U(y) = μ⋅y - 0.5 * y' Σ y
    fn u(&self, y: &DVector<f64>) -> f64 {
        self.mu.dot(y) - 0.5 * y.dot(&(&self.sigma * y))
    }
    // Ubar(ν) = 0.5 * (μ-ν)' * ( Σ \ (μ-ν) )
    fn ubar(&self, nu: &DVector<f64>) -> f64 {
        let diff = &self.mu - nu; // μ - ν
        let tmp = self.solve_sigma(&diff); // Σ \ (μ - ν)
        0.5 * diff.dot(&tmp)
    }
    // grad Ubar(ν) = Σ \ (ν - μ)
    fn grad_ubar(&self, nu: &DVector<f64>) -> DVector<f64> {
        let rhs = nu - &self.mu; // ν - μ
        self.solve_sigma(&rhs) // Σ \ (ν - μ)
    }
}

// Linear(c) -> Markowitz(c, sqrt(eps()) * I)
pub fn linear_objective(c: DVector<f64>) -> Markowitz {
    // all elements must be strictly positive
    if c.iter().any(|&x| x < 0.0) {
        panic!("all elements must be strictly positive");
    }

    let n = c.len();
    let tiny = f64::EPSILON.sqrt(); // ≈ 1e-8

    // Σ = tiny * I
    let sigma = DMatrix::<f64>::identity(n, n) * tiny;

    Markowitz { mu: c, sigma }
}

#[cfg(test)]
mod test {
    use super::*;
    use rand::SeedableRng;
    use rand::prelude::*;

    const OBJ_TOL: f64 = 1e-5;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() <= OBJ_TOL
    }

    // helper: check Fenchel-Young style relation
    // U(obj, -g) - ν⋅(-g)  ≈  Ubar(obj, ν)
    fn check_conjugacy<O: Objective>(obj: &O, nu: &DVector<f64>) {
        let g = obj.grad_ubar(nu); // this is ∇Ubar(obj, ν)

        let minus_g = -&g;
        let lhs = obj.u(&minus_g) - nu.dot(&minus_g);
        let rhs = obj.ubar(nu);

        assert!(
            approx_eq(lhs, rhs),
            "conjugacy failed: lhs={} rhs={} tol={}",
            lhs,
            rhs,
            OBJ_TOL
        );
    }

    trait TestU {
        fn check_u(&self, v: &[f64], exp: f64);
    }
    impl<T: Objective> TestU for T {
        fn check_u(&self, v: &[f64], exp: f64) {
            let y = DVector::from_vec(v.to_vec());
            let val = self.u(&y);
            assert!(approx_eq(val, exp), "u({:?}) = {} expected {}", v, val, exp);
        }
    }

    #[test]
    fn nonpositive_quadratic() {
        // obj = NonpositiveQuadratic([1.0, 2.0])
        // note: Julia default a = ones
        let b = DVector::from_vec(vec![1.0, 2.0]);
        let obj = NonpositiveQuadratic::new(b, None);

        // U(obj, [1.0, 2.0]) ≈ 0.0
        obj.check_u(&[1.0, 2.0], 0.0);
        // U(obj, [2.0, 2.0]) ≈ 0.0
        obj.check_u(&[2.0, 2.0], 0.0);
        // U(obj, [0.0, 0.0]) ≈ -2.5
        obj.check_u(&[0.0, 0.0], -2.5);

        // random tests of conjugacy with grad_ubar
        let mut rng = StdRng::seed_from_u64(1);
        for _ in 0..5 {
            let nu = DVector::from_fn(obj.len(), |_, _| rng.random::<f64>());
            check_conjugacy(&obj, &nu);
        }
    }

    #[test]
    fn markowitz() {
        // Σ = 2.0 * I
        // μ = [1.0, 2.0]
        let mu = DVector::from_vec(vec![1.0, 2.0]);
        let sigma = DMatrix::<f64>::identity(2, 2) * 2.0;
        let obj = Markowitz::new(mu.clone(), sigma);

        // U(obj, [1.0, 2.0]) ≈ 0.0
        obj.check_u(&[1.0, 2.0], 0.0);

        // random tests of conjugacy with grad_ubar
        let mut rng = StdRng::seed_from_u64(1);
        for _ in 0..5 {
            let nu = DVector::from_fn(obj.len(), |_, _| rng.random::<f64>());
            check_conjugacy(&obj, &nu);
        }
    }

    #[test]
    fn linear_objective_tests() {
        // obj = Linear([1.0, 2.0] ./ 10)
        // which is Markowitz(μ, sqrt(eps()) * I) with μ = [0.1, 0.2]
        let mu_vec = DVector::from_vec(vec![1.0 / 10.0, 2.0 / 10.0]);
        let obj = linear_objective(mu_vec.clone());

        // U(obj, [3.0, 4.0]) ≈ 1.1
        obj.check_u(&[3.0, 4.0], 1.1);

        // random tests
        let mut rng = StdRng::seed_from_u64(1);
        for _ in 0..5 {
            let nu = DVector::from_fn(obj.len(), |_, _| rng.random::<f64>());

            // in Julia:
            // if any(obj.μ - ν .> 0)
            //     assert abs(U(obj, -g)) > 1/obj_tol && Ubar(obj, ν) > 1/obj_tol
            // else
            //     standard conjugacy check

            let diff_pos = {
                let mut any_pos = false;
                for i in 0..obj.len() {
                    if obj.mu[i] - nu[i] > 0.0 {
                        any_pos = true;
                        break;
                    }
                }
                any_pos
            };

            let g = obj.grad_ubar(&nu);
            let minus_g = -&g;
            let u_minus_g = obj.u(&minus_g);
            let ubar_nu = obj.ubar(&nu);

            if diff_pos {
                // abs(U(obj, -g)) > 1/obj_tol && Ubar(obj, ν) > 1/obj_tol
                assert!(
                    u_minus_g.abs() > 1.0 / OBJ_TOL,
                    "|U(-g)| = {} not > {}",
                    u_minus_g,
                    1.0 / OBJ_TOL
                );
                assert!(
                    ubar_nu > 1.0 / OBJ_TOL,
                    "Ubar(ν) = {} not > {}",
                    ubar_nu,
                    1.0 / OBJ_TOL
                );
            } else {
                let lhs = u_minus_g - nu.dot(&minus_g);
                let rhs = ubar_nu;
                assert!(
                    approx_eq(lhs, rhs),
                    "linear conjugacy failed: lhs={} rhs={} tol={}",
                    lhs,
                    rhs,
                    OBJ_TOL
                );
            }
        }
    }
}
