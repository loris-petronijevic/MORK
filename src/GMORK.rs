//! Implementation of [GMORK].

use crate::*;

/// [GMORK] is an implementation of general multi-order Runge-Kutta methods.
pub struct GMORK {
    nodes: Vec<f64>,
    main_weights: Vec<Vec<Vec<f64>>>,
    main_weights_f: Box<dyn Fn(u32) -> Vec<Vec<f64>>>,
    secondary_weights: Vec<Vec<Vec<f64>>>,
    secondary_weights_f: Box<dyn Fn(u32) -> Vec<Vec<f64>>>,
    computation_order: Vec<SCC>,
    s: usize,
    pub stored_length: usize,
    implicit_ranks: Vec<Vec<bool>>, // [N-1][j]
    factorial: Vec<f64>,
    h_powers: Vec<f64>,
    h: f64,
    pub error_fraction: f64,
    pub min_iter: u32,
    pub max_iter: u32,
}

impl GMORK {
    /// [new][GMORK::new] creates a new instance of [GMORK].
    pub fn new(
        main_weights_function: Box<dyn Fn(u32) -> Vec<Vec<f64>>>,
        secondary_weights_function: Box<dyn Fn(u32) -> Vec<Vec<f64>>>,
        nodes: Vec<f64>,
        maximum_weight_graph: Vec<Vec<bool>>,
    ) -> Self {
        let main_weights = vec![main_weights_function(1)];
        let secondary_weights = vec![secondary_weights_function(1)];
        let s = main_weights[0].len() - 1;
        let mut implicit_ranks: Vec<Vec<bool>> = vec![vec![false; s]];
        let computation_order = create_computation_order(&maximum_weight_graph);
        for task in computation_order.iter() {
            if let SCC::Implicit(J, _) = task {
                for &j in J {
                    for &j1 in J {
                        if main_weights[0][j][j1] != 0. {
                            implicit_ranks[0][j] = true;
                            break;
                        }
                    }
                }
            }
        }
        GMORK {
            s,
            stored_length: 1,
            nodes,
            main_weights,
            main_weights_f: main_weights_function,
            secondary_weights,
            secondary_weights_f: secondary_weights_function,
            factorial: vec![1., 1.],
            h: 0.,
            h_powers: vec![1., 0.],
            computation_order,
            implicit_ranks,
            error_fraction: ERROR_FRACTION,
            min_iter: MIN_ITER,
            max_iter: MAX_ITER,
        }
    }

    /// [set_minimum_length][GMORK::set_minimum_length] ensures the method stores at least the constant necessary for initial value problems of order up to `n`
    pub fn set_minimum_length(
        n: usize,
        s: usize,
        stored_length: &mut usize,
        factorial: &mut Vec<f64>,
        main_weights: &mut Vec<Vec<Vec<f64>>>,
        main_weights_f: &dyn Fn(u32) -> Vec<Vec<f64>>,
        secondary_weights: &mut Vec<Vec<Vec<f64>>>,
        secondary_weights_f: &dyn Fn(u32) -> Vec<Vec<f64>>,
        h_powers: &mut Vec<f64>,
        h: f64,
        cyclic_derivatives: &mut Vec<Vec<bool>>,
        computation_order: &Vec<SCC>,
    ) {
        if *stored_length >= n {
            return;
        }
        factorial.extend(vec![0.; n - *stored_length]);
        main_weights.extend((*stored_length..n).map(|N| (main_weights_f)(N as u32 + 1)));
        secondary_weights.extend((*stored_length..n).map(|N| (secondary_weights_f)(N as u32 + 1)));
        h_powers.extend((*stored_length..n).map(|N| h.powi(N as i32 + 1)));
        cyclic_derivatives.extend((*stored_length..n).map(|_| vec![false; s]));
        for N in *stored_length..n {
            factorial[N + 1] = factorial[N] * (N as f64 + 1.);
            for task in computation_order.iter() {
                if let SCC::Implicit(J, _) = task {
                    for &j in J {
                        for &j1 in J {
                            if main_weights[N][j][j1] != 0. {
                                cyclic_derivatives[N][j] = true;
                                break;
                            }
                        }
                    }
                }
            }
        }
        *stored_length = n;
    }

    fn add_constant_part(
        J: &Vec<usize>,
        j: usize,
        y: &mut Vec<Vec<Vec<f64>>>,
        y0: &Vec<Vec<f64>>,
        F: &Vec<Vec<f64>>,
        main_weights: &Vec<Vec<Vec<f64>>>,
        secondary_weights: &Vec<Vec<Vec<f64>>>,
        h_powers: &Vec<f64>,
        factorial: &Vec<f64>,
    ) {
        let mut sum;
        for k in 0..y0.len() {
            for N in 0..y0[k].len() {
                for N1 in 0..N {
                    y[j][k][N] += secondary_weights[N][N1][j] * h_powers[N-N1] * y0[k][N1]
                }
                sum = 0.;
                for &j1 in J {
                    sum += main_weights[N][j][j1] * F[j1][k];
                }
                y[j][k][N] += h_powers[N + 1] / factorial[N + 1] * sum;
            }
        }
    }

    fn picard_iterations(
        t: f64,
        h: f64,
        y0: &Vec<Vec<f64>>,
        y: &mut Vec<Vec<Vec<f64>>>,
        F: &mut Vec<Vec<f64>>,
        f: &dyn Fn(f64, &Vec<Vec<f64>>) -> Vec<f64>,
        J: &Vec<usize>,
        threshold: f64,
        min_iter: u32,
        max_iter: u32,
        nodes: &Vec<f64>,
        cyclic_derivatives: &Vec<Vec<bool>>,
        main_weights: &Vec<Vec<Vec<f64>>>,
        h_powers: &Vec<f64>,
        factorial: &Vec<f64>,
    ) {
        let mut sum;
        let constant = y.clone();
        let mut iter_count = 0;
        let mut d = threshold + 1.;
        let mut f_cache: Vec<f64>;
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
            // constant part
            *y = constant.clone();
            // add evaluations and calculate norm difference
            for &j in J {
                for k in 0..y0.len() {
                    for N in (0..y0[k].len()).filter(|&N| cyclic_derivatives[N][j]) {
                        sum = 0.;
                        for &j1 in J {
                            sum += main_weights[N][j][j1] * F[j1][k];
                        }
                        y[j][k][N] += h_powers[N + 1] / factorial[N + 1] * sum;
                    }
                }
            }
        }
    }

    /// [approximate_GMORK][GMORK::approximate_GMORK] is an implementation of the approximation of a method.
    pub fn approximate_GMORK(
        t: f64,
        h: f64,
        f: &dyn Fn(f64, &Vec<Vec<f64>>) -> Vec<f64>,
        y0: &Vec<Vec<f64>>,
        threshold: f64,
        s: usize,
        computation_order: &Vec<SCC>,
        nodes: &Vec<f64>,
        main_weights: &Vec<Vec<Vec<f64>>>,
        secondary_weights: &Vec<Vec<Vec<f64>>>,
        h_powers: &Vec<f64>,
        factorial: &Vec<f64>,
        min_iter: u32,
        max_iter: u32,
        cyclic_derivatives: &Vec<Vec<bool>>,
    ) -> Vec<Vec<f64>> {
        let mut F: Vec<Vec<f64>> = (0..s).map(|_| vec![0.; y0.len()]).collect();
        let mut y: Vec<Vec<Vec<f64>>> = (0..=s).map(|_| y0.clone()).collect();
        for task in computation_order.iter() {
            match task {
                SCC::Explicit(j) => {
                    let j = *j;
                    GMORK::add_constant_part(
                        &(0..s).collect(),
                        j,
                        &mut y,
                        y0,
                        &F,
                        main_weights,
                        secondary_weights,
                        h_powers,
                        factorial,
                    );
                    if j != s {
                        F[j] = f(t + nodes[j] * h, &y[j]);
                    }
                }
                SCC::Implicit(J, comp_J) => {
                    // calculate constant terms
                    for &j in J {
                        GMORK::add_constant_part(
                            comp_J,
                            j,
                            &mut y,
                            y0,
                            &F,
                            main_weights,
                            secondary_weights,
                            h_powers,
                            factorial,
                        );
                    }
                    GMORK::picard_iterations(
                        t,
                        h,
                        y0,
                        &mut y,
                        &mut F,
                        f,
                        J,
                        threshold,
                        min_iter,
                        max_iter,
                        nodes,
                        cyclic_derivatives,
                        main_weights,
                        h_powers,
                        factorial,
                    );
                }
            }
        }
        return y[s].clone();
    }
}

impl Solver for GMORK {
    fn approximate(
        &mut self,
        t: f64,
        h: f64,
        f: &dyn Fn(f64, &Vec<Vec<f64>>) -> Vec<f64>,
        y0: &Vec<Vec<f64>>,
    ) -> Vec<Vec<f64>> {
        if h != self.h {
            self.h = h;
            for N in 1..=self.stored_length {
                self.h_powers[N] = h.powi(N as i32)
            }
        }
        // computes difference threshold for picard iterations
        let mut threshold = y0[0][0].abs();
        for k in 0..y0.len() {
            // verifies the length of the method is enough
            if y0[k].len() > self.stored_length {
                GMORK::set_minimum_length(
                    y0[k].len(),
                    self.s,
                    &mut self.stored_length,
                    &mut self.factorial,
                    &mut self.main_weights,
                    &self.main_weights_f,
                    &mut self.secondary_weights,
                    &self.secondary_weights_f,
                    &mut self.h_powers,
                    h,
                    &mut self.implicit_ranks,
                    &self.computation_order,
                );
            }
            for N in 0..y0[k].len() {
                if threshold < y0[k][N].abs() {
                    threshold = y0[k][N].abs()
                }
            }
        }
        threshold *= self.error_fraction;
        GMORK::approximate_GMORK(
            t,
            h,
            f,
            y0,
            threshold,
            self.s,
            &self.computation_order,
            &self.nodes,
            &self.main_weights,
            &self.secondary_weights,
            &self.h_powers,
            &self.factorial,
            self.min_iter,
            self.max_iter,
            &self.implicit_ranks,
        )
    }
}