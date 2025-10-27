/// edges.jl in rs
use rayon::prelude::*;

pub type F64Fun = Box<dyn Fn(f64) -> f64 + Send + Sync>;

pub trait Edge<T> {
    fn ai(&self) -> (usize, usize);
    fn len(&self) -> usize {
        2
    }
}

pub struct EdgeGain {
    pub ai: (usize, usize),
    pub h: F64Fun,
    pub ub: f64,
}

impl Edge<f64> for EdgeGain {
    fn ai(&self) -> (usize, usize) {
        self.ai
    }
}

pub struct EdgeClosedForm {
    pub ai: (usize, usize),
    pub h: F64Fun,
    pub ub: f64,
    pub wstar: F64Fun,
}

impl Edge<f64> for EdgeClosedForm {
    fn ai(&self) -> (usize, usize) {
        self.ai
    }
}

pub enum EdgeEither {
    Gain(EdgeGain),
    ClosedForm(EdgeClosedForm),
}

pub fn edge(inds: (usize, usize), h: F64Fun, ub: f64, wstar: Option<F64Fun>) -> EdgeEither {
    match wstar {
        None => EdgeEither::Gain(EdgeGain { ai: inds, h, ub }),
        Some(w) => EdgeEither::ClosedForm(EdgeClosedForm {
            ai: inds,
            h,
            ub,
            wstar: w,
        }),
    }
}

fn find_arb_closed_form(x: &mut [f64; 2], e: &EdgeClosedForm, ratio: f64) {
    x[0] = -(e.wstar)(ratio);
    x[1] = (e.h)(-x[0]);
}

pub struct FindArb {
    pub delta_abs_break: f64,
    pub first_derivative_eps: f64,
    pub second_derivative_eps: f64,
}

impl Default for FindArb {
    fn default() -> Self {
        Self {
            delta_abs_break: 1e-8,
            first_derivative_eps: 1e-8,
            second_derivative_eps: 1e-5,
        }
    }
}

impl FindArb {
    fn first_derivative(&self, f: &impl Fn(f64) -> f64, x: f64) -> f64 {
        let eps = self.first_derivative_eps;
        (f(x + eps) - f(x - eps)) / (2.0 * eps)
    }

    fn second_derivative(&self, f: &impl Fn(f64) -> f64, x: f64) -> f64 {
        let eps = self.second_derivative_eps;
        (f(x + eps) - 2.0 * f(x) + f(x - eps)) / (eps * eps)
    }
    fn find_arb_gain(&self, x: &mut [f64; 2], e: &EdgeGain, ratio: f64) {
        let p_min = self.first_derivative(&e.h, e.ub);
        let p_max = self.first_derivative(&e.h, 0.0);

        if !ratio.is_finite() || ratio >= p_max {
            x[0] = 0.0;
            x[1] = (e.h)(0.0);
            return;
        } else if ratio <= p_min {
            x[0] = -e.ub;
            x[1] = (e.h)(e.ub);
            return;
        }

        let mut x1 = e.ub * 0.5;
        for _ in 0..20 {
            let dh = self.first_derivative(&e.h, x1);
            let d2h = self.second_derivative(&e.h, x1);
            let delta = (ratio - dh) / d2h;
            x1 += delta;
            if x1 < 0.0 {
                x1 = 0.0;
            } else if x1 > e.ub {
                x1 = e.ub;
            }
            if delta.abs() <= self.delta_abs_break {
                break;
            }
        }

        x[0] = -x1;
        x[1] = (e.h)(-x[0]);
    }
    fn find_arb_ratio(&self, x: &mut [f64; 2], e: &EdgeEither, ratio: f64) {
        // have to say, not a fan of the function sig polymorphism mixed with the edge union types in julia (look .jl code)
        match e {
            EdgeEither::ClosedForm(e) => find_arb_closed_form(x, e, ratio),
            EdgeEither::Gain(e) => self.find_arb_gain(x, e, ratio),
        }
    }
    pub fn find_all(&self, xs: &mut [[f64; 2]], nu: &[f64], edges: &[EdgeEither]) {
        xs.par_iter_mut()
            .zip(edges.par_iter())
            .for_each(|(xslot, e)| {
                let (i1, i2) = match e {
                    EdgeEither::Gain(g) => g.ai,
                    EdgeEither::ClosedForm(c) => c.ai,
                };
                self.find_arb_ratio(xslot, e, nu[i1] / nu[i2]);
            });
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::sync::LazyLock;

    const EDGE_TOL: f64 = 1e-6;
    const FA: LazyLock<FindArb> = LazyLock::new(|| FindArb::default());

    fn assert_properties(
        name: &str,
        e: EdgeGain,
        e_closed: EdgeClosedForm,
        nrat: &[f64],
        ub: f64,
        h: impl Fn(f64) -> f64,
        dh: impl Fn(f64) -> f64,
    ) {
        let mut x = [0.0; 2];
        let mut xc = [0.0; 2];
        for nrat in nrat {
            let nrat = *nrat;
            FA.find_arb_gain(&mut x, &e, nrat);
            find_arb_closed_form(&mut xc, &e_closed, nrat);

            println!("name={} nrat={} {:?} {:?}", name, nrat, x, xc);

            // test that x has form (-w, h(w)) where w ≥ 0
            let (w, wc) = (-x[0], -xc[0]);
            assert!((h(w) - x[1]).abs() <= EDGE_TOL);
            assert!((h(wc) - xc[1]).abs() <= EDGE_TOL);

            // closed form and regular should be same
            for i in 0..2 {
                assert!((x[i] - xc[i]).abs() <= EDGE_TOL);
            }

            // optimality condition (smooth function)
            let (dh_ub, dh_lb) = (dh(ub), dh(0.0));
            if nrat >= dh_lb {
                assert!(w.abs() <= EDGE_TOL);
                assert!(wc.abs() <= EDGE_TOL);
            } else if nrat <= dh_ub {
                assert!((w - ub).abs() <= EDGE_TOL);
                assert!((wc - ub).abs() <= EDGE_TOL);
            } else {
                assert!((dh(w) - nrat).abs() <= EDGE_TOL);
                assert!((dh(wc) - nrat).abs() <= EDGE_TOL);
            }
        }
    }

    #[test]
    fn test_quadratic() {
        let ub = 1.0;
        let h = Box::new(|w: f64| 2.0 * w - w * w);
        let dh = Box::new(|w: f64| 2.0 - 2.0 * w);
        let wstar = Box::new(move |nrat: f64| {
            // capture `ub`
            if nrat >= 2.0 {
                0.0
            } else {
                (1f64 - nrat / 2f64).min(ub)
            }
        });
        let e = EdgeGain {
            ai: (1, 2),
            h: h.clone(),
            ub: ub,
        };
        let e_closed = EdgeClosedForm {
            ai: (1, 2),
            h: h.clone(),
            ub,
            wstar: wstar.clone(),
        };
        let nrat = [0.25, 0.75, 1.25, 1.75, 2.25];
        assert_properties("quad", e, e_closed, &nrat, ub, &h, &dh);
    }

    #[test]
    fn test_general() {
        let ub = 3.0;
        let h = Box::new(|w: f64| 3.0 * w - 16.0 * ((0.25 * w).exp().ln_1p() - 2.0f64.ln()));
        let dh = Box::new(|w: f64| {
            let z = 0.25 * w;
            let logistic = 1.0 / (1.0 + (-z).exp());
            3.0 - 4.0 * logistic
        });
        let wstar = Box::new(move |nrat: f64| {
            if nrat >= 1.0 {
                0.0
            } else {
                let val = 4.0 * ((3.0 - nrat) / (1.0 + nrat)).ln();
                val.min(ub)
            }
        });
        let e = EdgeGain {
            ai: (1, 2),
            h: h.clone(),
            ub: ub,
        };
        let e_closed = EdgeClosedForm {
            ai: (1, 2),
            h: h.clone(),
            ub,
            wstar: wstar.clone(),
        };
        // notice: nrat different to quad
        let nrat = [0.25, 0.5, 0.75, 1.0, 1.25];
        assert_properties("general", e, e_closed, &nrat, ub, &h, &dh);
    }
}
