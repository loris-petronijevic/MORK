//! Implementation of [RK], and a list of Runge-Kutta methods.

use crate::*;

/// [RK] is an implementation of Runge-Kutta methods.
pub struct RK {
    pub s: usize,
    pub nodes: Vec<f64>,
    pub weights: Vec<Vec<f64>>,
    pub computation_order: Vec<SCC>,
    pub error_fraction: f64,
    pub min_iter: u32,
    pub max_iter: u32,
}

impl RK {
    /// [new][RK::new] creates a new instance of [GMORK].
    pub fn new(weights: Vec<Vec<f64>>, nodes: Vec<f64>) -> Self {
        let s = weights.len() - 1;
        let weight_graph = (0..=s)
            .map(|j| {
                (0..=s)
                    .map(|j1| {
                        if j1 == s || weights[j][j1] == 0. {
                            false
                        } else {
                            true
                        }
                    })
                    .collect()
            })
            .collect();
        let computation_order = create_computation_order(&weight_graph);
        let s = nodes.len() - 1;
        RK {
            s,
            nodes,
            weights,
            computation_order,
            error_fraction: ERROR_FRACTION,
            min_iter: MIN_ITER,
            max_iter: MAX_ITER,
        }
    }

    fn picard_iterations(
        threshold: f64,
        min_iter: u32,
        max_iter: u32,
        t: f64,
        h: f64,
        nodes: &Vec<f64>,
        F: &mut Vec<Vec<f64>>,
        J: &Vec<usize>,
        f: &dyn Fn(f64, &Vec<Vec<f64>>) -> Vec<f64>,
        weights: &Vec<Vec<f64>>,
        y0: &Vec<Vec<f64>>,
        y: &mut Vec<Vec<Vec<f64>>>,
    ) {
        let constant = y.clone();
        let mut iter_count = 0;
        let mut d = threshold + 1.;
        let mut f_cache;
        let mut sum;
        while iter_count < min_iter || (d > threshold && iter_count < max_iter) {
            iter_count += 1;
            // update evaluations and calculate difference
            d = 0.;
            for &j in J {
                f_cache = f(t + nodes[j] * h, &y[j]);
                // calculate difference for threshold
                for k in 0..f_cache.len() {
                    d = d.max((f_cache[k] - F[j][k]).abs());
                }
                F[j] = f_cache;
            }
            // add evaluations
            for &j in J {
                for k in 0..y0.len() {
                    for N in (1..y0[k].len()).rev() {
                        sum = 0.;
                        for &j1 in J {
                            sum += weights[j][j1] * y[j1][k][N - 1];
                        }
                        y[j][k][N] = constant[j][k][N] + h * sum;
                    }
                    sum = 0.;
                    for &j1 in J {
                        sum += weights[j][j1] * F[j1][k];
                    }
                    y[j][k][0] = constant[j][k][0] + h * sum;
                }
            }
        }
    }

    /// [approximate_RK][RK::approximate_RK] is an implementation of the approximation of a method.
    pub fn approximate_RK(
        t: f64,
        h: f64,
        f: &dyn Fn(f64, &Vec<Vec<f64>>) -> Vec<f64>,
        y0: &Vec<Vec<f64>>,
        s: usize,
        threshold: f64,
        computation_order: &Vec<SCC>,
        weights: &Vec<Vec<f64>>,
        nodes: &Vec<f64>,
        min_iter: u32,
        max_iter: u32,
    ) -> Vec<Vec<f64>> {
        let mut F: Vec<Vec<f64>> = (0..s).map(|_| vec![0.; y0.len()]).collect();
        let mut y: Vec<Vec<Vec<f64>>> = (0..=s).map(|_| y0.clone()).collect();
        let mut sum;
        for task in computation_order.iter() {
            match task {
                SCC::Explicit(j) => {
                    let j = *j;
                    for k in 0..y0.len() {
                        sum = 0.;
                        for j1 in 0..s {
                            sum += weights[j][j1] * F[j1][k];
                        }
                        y[j][k][0] += h * sum;
                        for N in 1..y0[k].len() {
                            sum = 0.;
                            for j1 in 0..s {
                                sum += weights[j][j1] * y[j1][k][N - 1];
                            }
                            y[j][k][N] += h * sum;
                        }
                    }
                    if j != s {
                        F[j] = f(t + nodes[j] * h, &y[j]);
                    }
                }

                SCC::Implicit(J, comp_J) => {
                    // calculate constant terms
                    for &j in J {
                        for k in 0..y0.len() {
                            sum = 0.;
                            for &j1 in comp_J {
                                sum += weights[j][j1] * F[j1][k];
                            }
                            y[j][k][0] += h * sum;
                            for N in 1..y0[k].len() {
                                sum = 0.;
                                for &j1 in comp_J {
                                    sum += weights[j][j1] * y[j1][k][N - 1];
                                }
                                y[j][k][N] += h * sum;
                            }
                        }
                    }
                    RK::picard_iterations(
                        threshold, min_iter, max_iter, t, h, nodes, &mut F, J, f, weights, y0,
                        &mut y,
                    );
                }
            }
        }
        return y[s].clone();
    }
}

impl Solver for RK {
    fn approximate(
        &mut self,
        t: f64,
        h: f64,
        f: &dyn Fn(f64, &Vec<Vec<f64>>) -> Vec<f64>,
        y0: &Vec<Vec<f64>>,
    ) -> Vec<Vec<f64>> {
        if h == 0. {
            return y0.clone();
        }
        // computes difference threshold for picard iterations
        let mut threshold = y0[0][0].abs();
        for k in 0..y0.len() {
            for N in 0..y0[k].len() {
                if threshold < y0[k][N].abs() {
                    threshold = y0[k][N].abs()
                }
            }
        }
        threshold *= self.error_fraction;
        RK::approximate_RK(
            t,
            h,
            f,
            y0,
            self.s,
            threshold,
            &self.computation_order,
            &self.weights,
            &self.nodes,
            self.min_iter,
            self.max_iter,
        )
    }
}

pub mod list {
    //! A list of Runge-Kutta methods.

    use super::RK;

    pub fn explicit_euler_weights() -> Vec<Vec<f64>> {
        vec![vec![0.], vec![1.]]
    }

    pub fn explicit_euler_nodes() -> Vec<f64> {
        vec![0., 1.]
    }

    pub fn explicit_euler() -> RK {
        RK::new(explicit_euler_weights(), explicit_euler_nodes())
    }

    pub fn explicit_midpoint_weights() -> Vec<Vec<f64>> {
        vec![vec![0., 0.], vec![0.5, 0.], vec![0., 1.]]
    }

    pub fn explicit_midpoint_nodes() -> Vec<f64> {
        vec![0., 0.5, 1.]
    }

    pub fn explicit_midpoint() -> RK {
        RK::new(explicit_midpoint_weights(), explicit_midpoint_nodes())
    }

    pub fn ralston_weights() -> Vec<Vec<f64>> {
        vec![vec![0., 0.], vec![2. / 3., 0.], vec![1. / 4., 3. / 4.]]
    }

    pub fn ralston_nodes() -> Vec<f64> {
        vec![0., 2. / 3., 1.]
    }

    pub fn ralston() -> RK {
        RK::new(ralston_weights(), ralston_nodes())
    }

    pub fn heun_weights() -> Vec<Vec<f64>> {
        vec![
            vec![0., 0., 0.],
            vec![1. / 3., 0., 0.],
            vec![0., 2. / 3., 0.],
            vec![1. / 4., 0., 3. / 4.],
        ]
    }

    pub fn heun_nodes() -> Vec<f64> {
        vec![0., 1. / 3., 2. / 3., 1.]
    }

    pub fn heun() -> RK {
        RK::new(heun_weights(), heun_nodes())
    }

    pub fn RK4_weights() -> Vec<Vec<f64>> {
        {
            vec![
                vec![0., 0., 0., 0.],
                vec![0.5, 0., 0., 0.],
                vec![0., 0.5, 0., 0.],
                vec![0., 0., 1., 0.],
                vec![1. / 6., 1. / 3., 1. / 3., 1. / 6.],
            ]
        }
    }

    pub fn RK4_nodes() -> Vec<f64> {
        vec![0., 0.5, 0.5, 1., 1.]
    }

    pub fn RK4() -> RK {
        RK::new(RK4_weights(), RK4_nodes())
    }

    pub fn RK4b_weights() -> Vec<Vec<f64>> {
        vec![
            vec![0., 0., 0., 0.],
            vec![0.5, 0., 0., 0.],
            vec![0.25, 0.25, 0., 0.],
            vec![0., -1., 2., 0.],
            vec![1. / 6., 0., 2. / 3., 1. / 6.],
        ]
    }

    pub fn RK4b_nodes() -> Vec<f64> {
        vec![0., 0.5, 0.5, 1., 1.]
    }

    pub fn RK4b() -> RK {
        RK::new(RK4b_weights(), RK4b_nodes())
    }

    pub fn implicit_euler_weights() -> Vec<Vec<f64>> {
        vec![vec![1.], vec![1.]]
    }

    pub fn implicit_euler_nodes() -> Vec<f64> {
        vec![1., 1.]
    }

    pub fn implicit_euler() -> RK {
        RK::new(implicit_euler_weights(), implicit_euler_nodes())
    }

    pub fn implicit_midpoint_weights() -> Vec<Vec<f64>> {
        vec![vec![0.5], vec![1.]]
    }

    pub fn implicit_midpoint_nodes() -> Vec<f64> {
        vec![0.5, 1.]
    }

    pub fn implicit_midpoint() -> RK {
        RK::new(implicit_midpoint_weights(), implicit_midpoint_nodes())
    }

    pub fn crank_nicolson_weights() -> Vec<Vec<f64>> {
        vec![vec![0., 0.], vec![0.5, 0.5], vec![0.5, 0.5]]
    }

    pub fn crank_nicolson_nodes() -> Vec<f64> {
        vec![0., 1., 1.]
    }

    pub fn crank_nicolson() -> RK {
        RK::new(crank_nicolson_weights(), crank_nicolson_nodes())
    }

    pub fn CNb_weights() -> Vec<Vec<f64>> {
        vec![vec![0., 0.], vec![1. / 3., 1. / 3.], vec![1. / 4., 3. / 4.]]
    }

    pub fn CNb_nodes() -> Vec<f64> {
        vec![0., 2. / 3., 1.]
    }

    pub fn CNb() -> RK {
        RK::new(CNb_weights(), CNb_nodes())
    }

    pub fn gauss_legendre_weights() -> Vec<Vec<f64>> {
        let sqrt3 = 3_f64.sqrt();
        vec![
            vec![1. / 4., 1. / 4. - sqrt3 / 6.],
            vec![1. / 4. + sqrt3 / 6., 1. / 4.],
            vec![0.5, 0.5],
        ]
    }

    pub fn gauss_legendre_nodes() -> Vec<f64> {
        let sqrt3 = 3_f64.sqrt();
        vec![0.5 - sqrt3 / 6., 0.5 + sqrt3 / 6., 1.]
    }

    pub fn gauss_legendre() -> RK {
        RK::new(gauss_legendre_weights(), gauss_legendre_nodes())
    }
}