//! Implementation of [MORK], and a list of node-determined methods.

use crate::*;

pub enum GMORKType {
    General(Box<dyn Fn(u32) -> Vec<Vec<f64>>>), // Secondary weight function
    NodeDetermined,
}

enum CacheType {
    General(Box<dyn Fn(u32) -> Vec<Vec<f64>>>, Vec<Vec<Vec<f64>>>), // Secondary weight function, secondary weights
    NodeDetermined(Vec<Vec<f64>>),                                  // Coefficients
}

/// [MORK] is an implement of node-determined multi-order Runge-Kutta methods.
pub struct MORK {
    s: usize,
    pub stored_length: usize,
    nodes: Vec<f64>,
    main_weights: Vec<Vec<Vec<f64>>>,
    main_weights_function: Box<dyn Fn(u32) -> Vec<Vec<f64>>>,
    factorial: Vec<f64>,
    h: f64,
    h_powers: Vec<f64>,
    computation_order: Vec<SCC>,
    implicit_ranks: Vec<Vec<bool>>, // [N-1][j]
    pub error_fraction: f64,
    pub min_iter: u32,
    pub max_iter: u32,
    cache: CacheType,
}

impl MORK {
    /// [new][MORK::new] creates a new instance of [MORK]. Requires the nodes, the weight function and the maximum weight digraph of the method.
    pub fn new(
        main_weights_function: Box<dyn Fn(u32) -> Vec<Vec<f64>>>,
        nodes: Vec<f64>,
        maximum_weight_graph: Vec<Vec<bool>>,
        method_type: GMORKType,
    ) -> Self {
        let s = nodes.len() - 1;
        let weights = vec![main_weights_function(1)];
        let computation_order = create_computation_order(&maximum_weight_graph);
        let mut implicit_ranks: Vec<Vec<bool>> = vec![vec![false; s]];
        for task in computation_order.iter() {
            if let SCC::Implicit(J, _) = task {
                for &j in J {
                    for &j1 in J {
                        if weights[0][j][j1] != 0. {
                            implicit_ranks[0][j] = true;
                            break;
                        }
                    }
                }
            }
        }
        let cache = match method_type {
            GMORKType::General(secondary_weight_function) => {
                let weight_1 = secondary_weight_function(1);
                CacheType::General(secondary_weight_function, vec![weight_1])
            }
            GMORKType::NodeDetermined => CacheType::NodeDetermined(vec![vec![1.; s + 1]]),
        };
        MORK {
            s,
            stored_length: 1,
            nodes,
            main_weights: weights,
            main_weights_function,
            factorial: vec![1., 1.],
            h: 0.,
            h_powers: vec![1., 0.],
            computation_order,
            implicit_ranks,
            error_fraction: ERROR_FRACTION,
            min_iter: MIN_ITER,
            max_iter: MAX_ITER,
            cache,
        }
    }

    /// [set_minimum_length][MORK::set_minimum_length] ensures the method stores at least the constant necessary for initial value problems of order up to `n`
    pub fn set_minimum_length(&mut self, h: f64, n: usize) {
        if self.stored_length >= n {
            return;
        }
        self.factorial.extend(vec![0.; n - self.stored_length]);
        self.main_weights
            .extend((self.stored_length..n).map(|N| (self.main_weights_function)(N as u32 + 1)));
        self.h_powers
            .extend((self.stored_length..n).map(|N| h.powi(N as i32 + 1)));
        self.implicit_ranks
            .extend((self.stored_length..n).map(|_| vec![false; self.s]));
        for N in self.stored_length..n {
            self.factorial[N + 1] = self.factorial[N] * (N as f64 + 1.);
            for task in self.computation_order.iter() {
                if let SCC::Implicit(J, _) = task {
                    for &j in J {
                        for &j1 in J {
                            if self.main_weights[N][j][j1] != 0. {
                                self.implicit_ranks[N][j] = true;
                                break;
                            }
                        }
                    }
                }
            }
        }
        match &mut self.cache {
            CacheType::General(secondary_weight_function, secondary_weights) => secondary_weights
                .extend((self.stored_length..n).map(|N| (secondary_weight_function)(N as u32 + 1))),
            CacheType::NodeDetermined(coefficients) => {
                coefficients.extend((self.stored_length..n).map(|N| {
                    (0..self.s + 1)
                        .map(|j| self.nodes[j].powi(N as i32) / self.factorial[N])
                        .collect()
                }));
            }
        }
        self.stored_length = n;
    }

    /// [picard_iterations][MORK::picard_iterations] implements a Picard's iteration algorithm to solve implicit SCCs.
    fn picard_iterations(
        &self,
        t: f64,
        h: f64,
        y0: &Vec<Vec<f64>>,
        y: &mut Vec<Vec<Vec<f64>>>,
        F: &mut Vec<Vec<f64>>,
        f: &dyn Fn(f64, &Vec<Vec<f64>>) -> Vec<f64>,
        J: &Vec<usize>,
        threshold: f64,
    ) {
        let constant = y.clone();
        let mut iter_count = 0;
        let mut d = threshold + 1.;
        let mut f_cache: Vec<f64>;
        let mut sum;
        while iter_count < self.min_iter || (d > threshold && iter_count < self.max_iter) {
            iter_count += 1;
            // update evaluations and calculate difference
            d = 0.;
            for &j in J {
                f_cache = f(t + self.nodes[j] * h, &y[j]);
                // calculate difference for threshold
                for k in 0..f_cache.len() {
                    d = d.max((f_cache[k] - F[j][k]).abs());
                }
                F[j] = f_cache;
            }
            // add evaluations
            for &j in J {
                for k in 0..y0.len() {
                    for N in (0..y0[k].len()).filter(|&N| self.implicit_ranks[N][j]) {
                        sum = 0.;
                        for &j1 in J {
                            sum += self.main_weights[N][j][j1] * F[j1][k];
                        }
                        y[j][k][N] =
                            constant[j][k][N] + self.h_powers[N + 1] / self.factorial[N + 1] * sum;
                    }
                }
            }
        }
    }
}

impl Solver for MORK {
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
        if h != self.h {
            self.h = h;
            for N in 1..=self.stored_length {
                self.h_powers[N] = h.powi(N as i32)
            }
        }

        // computes threshold for picard iterations
        let mut threshold = y0[0][0].abs();
        for k in 0..y0.len() {
            // verifies the length of the method is enough
            if y0[k].len() > self.stored_length {
                self.set_minimum_length(h, y0[k].len());
            }
            for N in 0..y0[k].len() {
                if threshold < y0[k][N].abs() {
                    threshold = y0[k][N].abs()
                }
            }
        }
        threshold *= self.error_fraction;

        let mut F: Vec<Vec<f64>> = (0..self.s).map(|_| vec![0.; y0.len()]).collect();
        let mut y: Vec<Vec<Vec<f64>>> = (0..=self.s).map(|_| y0.clone()).collect();
        let mut sum;

        // Add initial values
        match &self.cache {
            // GMORK version
            CacheType::General(_, secondary_weights) => {
                for j in 0..=self.s {
                    for k in 0..y0.len() {
                        for N in 0..y0[k].len() {
                            for N1 in 0..N {
                                y[j][k][N] +=
                                    secondary_weights[N][N1][j] * self.h_powers[N - N1] * y0[k][N1]
                            }
                        }
                    }
                }
            }
            // NDMORK version
            CacheType::NodeDetermined(coefficients) => {
                for j in 0..=self.s {
                    if self.nodes[j] != 0. {
                        for k in 0..y0.len() {
                            for N in 0..y0[k].len() {
                                for N1 in 0..N {
                                    y[j][k][N] +=
                                        coefficients[N - N1][j] * self.h_powers[N - N1] * y0[k][N1]
                                }
                            }
                        }
                    }
                }
            }
        }

        for task in self.computation_order.iter() {
            match task {
                SCC::Explicit(j) => {
                    let j = *j;
                    for k in 0..y0.len() {
                        for N in 0..y0[k].len() {
                            sum = 0.;
                            for j1 in 0..self.s {
                                sum += self.main_weights[N][j][j1] * F[j1][k];
                            }
                            y[j][k][N] += self.h_powers[N + 1] / self.factorial[N + 1] * sum;
                        }
                    }
                    if j != self.s {
                        F[j] = f(t + self.nodes[j] * h, &y[j]);
                    }
                }

                SCC::Implicit(J, comp_J) => {
                    for &j in J {
                        for k in 0..y0.len() {
                            for N in 0..y0[k].len() {
                                sum = 0.;
                                for &j1 in comp_J {
                                    sum += self.main_weights[N][j][j1] * F[j1][k];
                                }
                                y[j][k][N] += self.h_powers[N + 1] / self.factorial[N + 1] * sum;
                            }
                        }
                    }
                    self.picard_iterations(t, h, y0, &mut y, &mut F, f, J, threshold);
                }
            }
        }
        return y[self.s].clone();
    }
}

pub mod list {
    //! A list of multi-order Runge-Kutta methods.

    use crate::MORK::GMORKType;

    use super::MORK;

    pub fn MO_explicit_euler_weight_function(_N: u32) -> Vec<Vec<f64>> {
        vec![vec![0.], vec![1.]]
    }

    pub fn MO_explicit_euler_nodes() -> Vec<f64> {
        vec![0., 1.]
    }

    pub fn MO_explicit_euler_weight_graph() -> Vec<Vec<bool>> {
        vec![vec![false, false], vec![true, false]]
    }

    pub fn MO_explicit_euler() -> MORK {
        MORK::new(
            Box::new(MO_explicit_euler_weight_function),
            MO_explicit_euler_nodes(),
            MO_explicit_euler_weight_graph(),
            GMORKType::NodeDetermined,
        )
    }

    pub fn MO_explicit_midpoint_weight_function(N: u32) -> Vec<Vec<f64>> {
        vec![
            vec![0., 0.],
            vec![2_f64.powi(-(N as i32)), 0.],
            vec![1. - 2. / (1. + N as f64), 2. / (1. + N as f64)],
        ]
    }

    pub fn MO_explicit_midpoint_nodes() -> Vec<f64> {
        vec![0., 0.5, 1.]
    }

    pub fn MO_explicit_midpoint_weight_graph() -> Vec<Vec<bool>> {
        vec![
            vec![false, false, false],
            vec![true, false, false],
            vec![true, true, false],
        ]
    }

    pub fn MO_explicit_midpoint() -> MORK {
        MORK::new(
            Box::new(MO_explicit_midpoint_weight_function),
            MO_explicit_midpoint_nodes(),
            MO_explicit_midpoint_weight_graph(),
            GMORKType::NodeDetermined,
        )
    }

    pub fn MO_ralston_weight_function(N: u32) -> Vec<Vec<f64>> {
        vec![
            vec![0., 0.],
            vec![(2_f64 / 3_f64).powi(N as i32), 0.],
            vec![
                (2. * N as f64 - 1.) / (2. * (1. + N as f64)),
                3. / (2. * (1. + N as f64)),
            ],
        ]
    }

    pub fn MO_ralston_nodes() -> Vec<f64> {
        vec![0., 2. / 3., 1.]
    }

    pub fn MO_ralston_weight_graph() -> Vec<Vec<bool>> {
        vec![
            vec![false, false, false],
            vec![true, false, false],
            vec![true, true, false],
        ]
    }

    pub fn MO_ralston() -> MORK {
        MORK::new(
            Box::new(MO_ralston_weight_function),
            MO_ralston_nodes(),
            MO_ralston_weight_graph(),
            GMORKType::NodeDetermined,
        )
    }

    pub fn MO_heun_weight_function(N: u32) -> Vec<Vec<f64>> {
        let N = N as i32;
        let Nf = N as f64;
        vec![
            vec![0., 0., 0.],
            vec![3_f64.powi(-N), 0., 0.],
            vec![
                (2. / 3_f64).powi(N) * (Nf - 1.) / (1. + Nf),
                (2. / 3_f64).powi(N) * 2. / (1. + Nf),
                0.,
            ],
            vec![
                1. - 9. * Nf / (2. * (1. + Nf) * (2. + Nf)),
                6. * (Nf - 1.) / ((1. + Nf) * (2. + Nf)),
                3. * (4. - Nf) / (2. * (1. + Nf) * (2. + Nf)),
            ],
        ]
    }

    pub fn MO_heun_nodes() -> Vec<f64> {
        vec![0., 1. / 3., 2. / 3., 1.]
    }

    pub fn MO_heun_weight_graph() -> Vec<Vec<bool>> {
        vec![
            vec![false, false, false, false],
            vec![true, false, false, false],
            vec![true, true, false, false],
            vec![true, true, true, false],
        ]
    }

    pub fn MO_heun() -> MORK {
        MORK::new(
            Box::new(MO_heun_weight_function),
            MO_heun_nodes(),
            MO_heun_weight_graph(),
            GMORKType::NodeDetermined,
        )
    }

    pub fn MORK4_weight_function(N: u32) -> Vec<Vec<f64>> {
        let N = N as i32;
        let Nf = N as f64;
        vec![
            vec![0., 0., 0., 0.],
            vec![2_f64.powi(-N), 0., 0., 0.],
            vec![
                2_f64.powi(-N) * (Nf - 1.) / (1. + Nf),
                2_f64.powi(1 - N) / (1. + Nf),
                0.,
                0.,
            ],
            vec![(Nf - 1.) / (1. + Nf), (1. - Nf) / (1. + Nf), 1., 0.],
            vec![
                Nf.powi(2) / ((1. + Nf) * (2. + Nf)),
                2. * Nf / ((1. + Nf) * (2. + Nf)),
                2. * Nf / ((1. + Nf) * (2. + Nf)),
                (2. - Nf) / ((1. + Nf) * (2. + Nf)),
            ],
        ]
    }

    pub fn MORK4_nodes() -> Vec<f64> {
        vec![0., 0.5, 0.5, 1., 1.]
    }

    pub fn MO_RK4_weight_graph() -> Vec<Vec<bool>> {
        vec![
            vec![false, false, false, false, false],
            vec![true, false, false, false, false],
            vec![true, true, false, false, false],
            vec![true, true, true, false, false],
            vec![true, true, true, true, false],
        ]
    }

    pub fn MORK4() -> MORK {
        MORK::new(
            Box::new(MORK4_weight_function),
            MORK4_nodes(),
            MO_RK4_weight_graph(),
            GMORKType::NodeDetermined,
        )
    }

    pub fn MORK4b_weight_function(N: u32) -> Vec<Vec<f64>> {
        let N = N as i32;
        let Nf = N as f64;
        vec![
            vec![0., 0., 0., 0.],
            vec![2_f64.powi(-N), 0., 0., 0.],
            vec![
                2_f64.powi(-N) * Nf / (1. + Nf),
                2_f64.powi(-N) / (1. + Nf),
                0.,
                0.,
            ],
            vec![
                (Nf - 1.) / (1. + Nf),
                2. * (Nf - 2.) / (1. + Nf),
                2. * (3. - Nf) / (1. + Nf),
                0.,
            ],
            vec![
                Nf.powi(2) / ((1. + Nf) * (2. + Nf)),
                0.,
                4. * Nf / ((1. + Nf) * (2. + Nf)),
                (2. - Nf) / ((1. + Nf) * (2. + Nf)),
            ],
        ]
    }

    pub fn MORK4b_nodes() -> Vec<f64> {
        vec![0., 0.5, 0.5, 1., 1.]
    }

    pub fn MORK4b_weight_graph() -> Vec<Vec<bool>> {
        vec![
            vec![false, false, false, false, false],
            vec![true, false, false, false, false],
            vec![true, true, false, false, false],
            vec![true, true, true, false, false],
            vec![true, false, true, true, false],
        ]
    }

    pub fn MORK4b() -> MORK {
        MORK::new(
            Box::new(MORK4b_weight_function),
            MORK4b_nodes(),
            MORK4b_weight_graph(),
            GMORKType::NodeDetermined,
        )
    }

    pub fn MO_implicit_euler_weight_function(_N: u32) -> Vec<Vec<f64>> {
        vec![vec![1.], vec![1.]]
    }

    pub fn MO_implicit_euler_nodes() -> Vec<f64> {
        vec![1., 1.]
    }

    pub fn MO_implicit_euler_weight_graph() -> Vec<Vec<bool>> {
        vec![vec![true, false], vec![true, false]]
    }

    pub fn MO_implicit_euler() -> MORK {
        MORK::new(
            Box::new(MO_implicit_euler_weight_function),
            MO_implicit_euler_nodes(),
            MO_implicit_euler_weight_graph(),
            GMORKType::NodeDetermined,
        )
    }

    pub fn MO_implicit_midpoint_weight_function(N: u32) -> Vec<Vec<f64>> {
        vec![vec![2_f64.powi(-(N as i32))], vec![1.]]
    }

    pub fn MO_implicit_midpoint_nodes() -> Vec<f64> {
        vec![0.5, 1.]
    }

    pub fn MO_implicit_midpoint_weight_graph() -> Vec<Vec<bool>> {
        vec![vec![true, false], vec![true, false]]
    }

    pub fn MO_implicit_midpoint() -> MORK {
        MORK::new(
            Box::new(MO_implicit_midpoint_weight_function),
            MO_implicit_midpoint_nodes(),
            MO_implicit_midpoint_weight_graph(),
            GMORKType::NodeDetermined,
        )
    }

    pub fn MO_crank_nicolson_weight_function(_N: u32) -> Vec<Vec<f64>> {
        vec![vec![0., 0.], vec![0.5, 0.5], vec![0.5, 0.5]]
    }

    pub fn MO_crank_nicolson_nodes() -> Vec<f64> {
        vec![0., 1., 1.]
    }

    pub fn MO_crank_nicolson_weight_graph() -> Vec<Vec<bool>> {
        vec![
            vec![false, false, false],
            vec![true, true, false],
            vec![true, true, false],
        ]
    }

    pub fn MO_crank_nicolson() -> MORK {
        MORK::new(
            Box::new(MO_crank_nicolson_weight_function),
            MO_crank_nicolson_nodes(),
            MO_crank_nicolson_weight_graph(),
            GMORKType::NodeDetermined,
        )
    }

    pub fn MO_CNb_weight_function(N: u32) -> Vec<Vec<f64>> {
        let Nf = N as f64;
        let c = (2_f64 / 3_f64).powi(N as i32);
        vec![
            vec![0., 0.],
            vec![c * Nf / (1. + Nf), c / (1. + Nf)],
            vec![(2. * Nf - 1.) / (2. * (1. + Nf)), 3. / (2. * (1. + Nf))],
        ]
    }

    pub fn MO_CNb_nodes() -> Vec<f64> {
        vec![0., 2. / 3., 1.]
    }

    pub fn MO_CNb_weight_graph() -> Vec<Vec<bool>> {
        vec![
            vec![false, false, false],
            vec![true, true, false],
            vec![true, true, false],
        ]
    }

    pub fn MO_CNb() -> MORK {
        MORK::new(
            Box::new(MO_CNb_weight_function),
            MO_CNb_nodes(),
            MO_CNb_weight_graph(),
            GMORKType::NodeDetermined,
        )
    }

    pub fn MO_gauss_legendre_weight_function(N: u32) -> Vec<Vec<f64>> {
        let sqrt3 = 3_f64.sqrt();
        let no1 = 0.5 - 3_f64.sqrt() / 6.;
        let no2 = 0.5 + 3_f64.sqrt() / 6.;
        let Nf = N as f64;
        vec![
            vec![
                no1.powi(N as i32) / (1. + Nf) * (1. + Nf / 2. * (1. + sqrt3)),
                -sqrt3 * Nf / (1. + Nf) * no1.powi(N as i32 + 1),
            ],
            vec![
                sqrt3 * Nf / (1. + Nf) * no2.powi(N as i32 + 1),
                no2.powi(N as i32) / (1. + Nf) * (1. + Nf / 2. * (1. - sqrt3)),
            ],
            vec![
                0.5 + sqrt3 * (Nf - 1.) / (2. * (1. + Nf)),
                0.5 - sqrt3 * (Nf - 1.) / (2. * (1. + Nf)),
            ],
        ]
    }

    pub fn MO_gauss_legendre_nodes() -> Vec<f64> {
        vec![0.5 - 3_f64.sqrt() / 6., 0.5 + 3_f64.sqrt() / 6., 1.]
    }

    pub fn MO_gauss_legendre_weight_graph() -> Vec<Vec<bool>> {
        vec![
            vec![true, true, false],
            vec![true, true, false],
            vec![true, true, false],
        ]
    }

    pub fn MO_gauss_legendre() -> MORK {
        MORK::new(
            Box::new(MO_gauss_legendre_weight_function),
            MO_gauss_legendre_nodes(),
            MO_gauss_legendre_weight_graph(),
            GMORKType::NodeDetermined,
        )
    }
}
