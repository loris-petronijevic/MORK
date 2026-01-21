#![allow(non_snake_case)]

/// [Experiment] indicates that a struct  
pub trait Experiment {
    fn differential_equation(&self, t: f64, y: &Vec<Vec<f64>>) -> Vec<f64>;
    fn is_solved(&self) -> bool;
    fn solution(&self, t: f64) -> Option<Vec<Vec<f64>>>; // Should return None if not solution
    fn initial_values(&self) -> (f64, Vec<Vec<f64>>); // t0 and y0
    fn get_parameters(&self) -> (Vec<f64>, Vec<String>);
    fn change_parameters(&mut self, parameters: Vec<f64>);
    fn apply_parameters(&mut self); // If changing parameters involves heavy computations this function will be called before the experiment is used as a trigger for those computations
    fn name(&self) -> String;
}

pub struct SolvedIVP {
    pub differential_equation: fn(f64, &Vec<Vec<f64>>, &Vec<f64>, f64, &Vec<f64>) -> Vec<f64>, // t,y,parameters,t0,cache
    pub solution: fn(f64, &Vec<f64>, f64, &Vec<f64>) -> Vec<Vec<f64>>, // t, parameter,t0, cache
    pub compute_cache: fn(&Vec<f64>, f64) -> Vec<f64>,                 // parameters,t0
    pub t0: f64,
    pub parameters: Vec<f64>,
    pub parameters_names: Vec<String>,
    pub ivp_name: String,
    pub cache: Vec<f64>,
}

impl Experiment for SolvedIVP {
    fn differential_equation(&self, t: f64, y: &Vec<Vec<f64>>) -> Vec<f64> {
        (self.differential_equation)(t, y, &self.parameters, self.t0, &self.cache)
    }
    fn is_solved(&self) -> bool {
        true
    }
    fn apply_parameters(&mut self) {
        self.cache = (self.compute_cache)(&self.parameters, self.t0)
    }
    fn solution(&self, t: f64) -> Option<Vec<Vec<f64>>> {
        Some((self.solution)(t, &self.parameters, self.t0, &self.cache))
    }
    fn change_parameters(&mut self, mut parameters: Vec<f64>) {
        let t0 = parameters.remove(0);
        self.parameters = parameters;
        self.t0 = t0;
    }
    fn get_parameters(&self) -> (Vec<f64>, Vec<String>) {
        let mut parameters = self.parameters.clone();
        parameters.insert(0, self.t0);
        let mut parameters_name = self.parameters_names.clone();
        parameters_name.insert(0, "t0".to_string());
        (parameters, parameters_name)
    }
    fn initial_values(&self) -> (f64, Vec<Vec<f64>>) {
        (
            self.t0,
            (self.solution)(self.t0, &self.parameters, self.t0, &self.cache),
        )
    }
    fn name(&self) -> String {
        self.ivp_name.clone()
    }
}
pub struct IVP {
    pub differential_equation: fn(f64, &Vec<Vec<f64>>, &Vec<f64>, f64, &Vec<f64>) -> Vec<f64>, // t,y,parameters,t0,cache
    pub compute_cache: fn(&Vec<f64>, f64) -> Vec<f64>, // parameters,t0
    pub t0: f64,
    pub y0: Vec<Vec<f64>>,
    pub parameters: Vec<f64>,
    pub parameters_names: Vec<String>,
    pub ivp_name: String,
    pub cache: Vec<f64>,
}

impl Experiment for IVP {
    fn differential_equation(&self, t: f64, y: &Vec<Vec<f64>>) -> Vec<f64> {
        (self.differential_equation)(t, y, &self.parameters, self.t0, &self.cache)
    }
    fn is_solved(&self) -> bool {
        false
    }
    fn apply_parameters(&mut self) {
        self.cache = (self.compute_cache)(&self.parameters, self.t0)
    }
    fn solution(&self, _t: f64) -> Option<Vec<Vec<f64>>> {
        None
    }
    fn change_parameters(&mut self, parameters: Vec<f64>) {
        self.parameters = parameters;
    }
    fn get_parameters(&self) -> (Vec<f64>, Vec<String>) {
        (self.parameters.clone(), self.parameters_names.clone())
    }
    fn initial_values(&self) -> (f64, Vec<Vec<f64>>) {
        (self.t0, self.y0.clone())
    }
    fn name(&self) -> String {
        self.ivp_name.clone()
    }
}

pub struct DynamicParameterSolvedIVP {
    pub differential_equation: fn(f64, &Vec<Vec<f64>>, &Vec<f64>, f64, &Vec<Vec<f64>>) -> Vec<f64>, // t,y,parameters,t0,cache -> highest derivative
    pub solution: fn(f64, &Vec<f64>, f64, &Vec<Vec<f64>>) -> Vec<Vec<f64>>, // t, parameter,t0, cache -> y
    pub compute_cache: fn(&Vec<f64>, f64) -> Vec<Vec<f64>>, // parameters,t0 -> cache
    pub t0: f64,
    pub parameters: Vec<f64>,
    pub parameters_names: Vec<String>,
    pub ivp_name: String,
    pub cache: Vec<Vec<f64>>,
    pub parameters_changer:
        fn(&Vec<String>, &Vec<f64>, Vec<f64>, f64, f64, &Vec<Vec<f64>>) -> (Vec<f64>, Vec<String>), // old names, old parameters, new parameters, old t0, new t0,cache -> new parameters
}

impl Experiment for DynamicParameterSolvedIVP {
    fn differential_equation(&self, t: f64, y: &Vec<Vec<f64>>) -> Vec<f64> {
        (self.differential_equation)(t, y, &self.parameters, self.t0, &self.cache)
    }
    fn is_solved(&self) -> bool {
        true
    }
    fn apply_parameters(&mut self) {
        self.cache = (self.compute_cache)(&self.parameters, self.t0)
    }
    fn solution(&self, t: f64) -> Option<Vec<Vec<f64>>> {
        Some((self.solution)(t, &self.parameters, self.t0, &self.cache))
    }
    fn change_parameters(&mut self, mut new_parameters: Vec<f64>) {
        let new_t0 = new_parameters.remove(0);
        (self.parameters, self.parameters_names) = (self.parameters_changer)(
            &self.parameters_names,
            &self.parameters,
            new_parameters,
            self.t0,
            new_t0,
            &self.cache,
        );
        self.t0 = new_t0;
    }
    fn get_parameters(&self) -> (Vec<f64>, Vec<String>) {
        let mut parameters = self.parameters.clone();
        parameters.insert(0, self.t0);
        let mut parameters_name = self.parameters_names.clone();
        parameters_name.insert(0, "t0".to_string());
        (parameters, parameters_name)
    }
    fn initial_values(&self) -> (f64, Vec<Vec<f64>>) {
        (
            self.t0,
            (self.solution)(self.t0, &self.parameters, self.t0, &self.cache),
        )
    }
    fn name(&self) -> String {
        self.ivp_name.clone()
    }
}

pub fn experiment_sin_cos() -> SolvedIVP {
    let differential_equation =
        |_, y: &Vec<Vec<f64>>, parameters: &Vec<f64>, _t0: f64, _cache: &Vec<f64>| {
            let n = parameters[0] as usize;
            let q = n / 2;
            let r = n % 2;
            let mut s = 0.;
            for k in 0..q {
                s += (-1_f64).powi((q - k) as i32) * y[0][n - 2 * k - r - 1];
            }
            vec![s / (q as f64)]
        };
    let solution = |t: f64, parameters: &Vec<f64>, _to: f64, _cache: &Vec<f64>| -> Vec<Vec<f64>> {
        let n = parameters[0] as usize;
        let v = vec![t.sin() + t.cos(), t.cos() - t.sin()];
        vec![
            (1..=n)
                .map(|N| (-1_f64).powi((n - N) as i32 / 2) * v[(n - N) % 2])
                .collect(),
        ]
    };
    let compute_cache = |_parameters: &Vec<f64>, _t0: f64| Vec::new();
    SolvedIVP {
        differential_equation,
        compute_cache,
        t0: 0.,
        solution,
        parameters: vec![2.],
        parameters_names: vec!["n".to_string()],
        ivp_name: "sin + cos".to_string(),
        cache: Vec::new(),
    }
}

pub fn experiment_exp() -> SolvedIVP {
    let differential_equation =
        |_, y: &Vec<Vec<f64>>, parameters: &Vec<f64>, _t0: f64, _cache: &Vec<f64>| {
            let n = parameters[0] as usize;
            let s: f64 = y[0].iter().sum();
            vec![s / n as f64]
        };
    let solution = |t: f64, parameters: &Vec<f64>, _to: f64, _cache: &Vec<f64>| -> Vec<Vec<f64>> {
        let n = parameters[0] as usize;
        vec![vec![t.exp(); n]]
    };
    let compute_cache = |_parameters: &Vec<f64>, _t0: f64| Vec::new();
    SolvedIVP {
        differential_equation,
        compute_cache,
        t0: 0.,
        solution,
        parameters: vec![2.],
        parameters_names: vec!["n".to_string()],
        ivp_name: "exp".to_string(),
        cache: Vec::new(),
    }
}

pub fn first_coo_extractor(y: &Vec<Vec<f64>>) -> Vec<f64> {
    y[0].clone()
}

pub fn highest_first_coo_extractor(y: &Vec<Vec<f64>>) -> Vec<f64> {
    vec![y[0][0]]
}

pub fn lowest_first_coo_extractor(y: &Vec<Vec<f64>>) -> Vec<f64> {
    vec![y[0][y[0].len() - 1]]
}

pub fn first_coo_metric(x: &Vec<Vec<f64>>, y: &Vec<Vec<f64>>) -> Vec<f64> {
    y[0].iter()
        .zip(x[0].iter())
        .map(|(x0, y0)| (x0 - y0).abs())
        .collect()
}

pub fn highest_first_coo_metric(x: &Vec<Vec<f64>>, y: &Vec<Vec<f64>>) -> Vec<f64> {
    vec![(y[0][0] - x[0][0]).abs()]
}

pub fn lowest_first_coo_metric(x: &Vec<Vec<f64>>, y: &Vec<Vec<f64>>) -> Vec<f64> {
    vec![(y[0][y[0].len() - 1] - x[0][y[0].len() - 1]).abs()]
}

pub fn experiment_linear_single_root() -> DynamicParameterSolvedIVP {
    let parameters = vec![1., 0., 0., 1., 0.];
    let parameters_names = vec![
        "Order".to_string(),
        "root's real part".to_string(),
        "root's imaginary part".to_string(),
        "y1 real part".to_string(),
        "y1 imaginary part".to_string(),
    ];

    let differential_equation =
        |_t: f64, y: &Vec<Vec<f64>>, parameters: &Vec<f64>, _t0: f64, cache: &Vec<Vec<f64>>| {
            let n = parameters[0] as usize;
            let mut sum_r = 0.;
            let mut sum_i = 0.;
            let mut res;
            let mut coef;
            for k in 0..n {
                coef = (-1_i32).pow((n - k) as u32) as f64 / (cache[0][k] * cache[0][n - k]);
                res = complex_mult(
                    cache[1][n - k],
                    cache[2][n - k],
                    y[0][n - k - 1],
                    y[1][n - k - 1],
                );
                sum_r += coef * res.0;
                sum_i += coef * res.1;
            }
            vec![-cache[0][n] * sum_r, -cache[0][n] * sum_i]
        };

    let solution = |t: f64, parameters: &Vec<f64>, t0: f64, cache: &Vec<Vec<f64>>| {
        let n = parameters[0] as usize;
        let mut y = vec![vec![0.; n]; 2];
        let mut sum_r;
        let mut sum_i;
        let mut coef = (parameters[1] * (t - t0)).exp();
        let lambda_exp_r = (parameters[2] * (t - t0)).cos() * coef;
        let lambda_exp_i = (parameters[2] * (t - t0)).sin() * coef;
        for i in 0..n {
            for k in 0..=i {
                sum_r = 0.;
                sum_i = 0.;
                for p in 0..(n - k) {
                    coef = cache[0][p + k] / (cache[0][p]) * ((t - t0).powi(p as i32));
                    sum_r += cache[3][p + k] * coef;
                    sum_i += cache[4][p + k] * coef;
                }
                sum_r /= cache[0][k] * cache[0][i - k];
                sum_i /= cache[0][k] * cache[0][i - k];
                (sum_r, sum_i) = complex_mult(sum_r, sum_i, cache[1][i - k], cache[2][i - k]);
                y[0][n - i - 1] += sum_r;
                y[1][n - i - 1] += sum_i;
            }
            y[0][n - i - 1] *= cache[0][i];
            y[1][n - i - 1] *= cache[0][i];
            (y[0][n - i - 1], y[1][n - i - 1]) =
                complex_mult(y[0][n - i - 1], y[1][n - i - 1], lambda_exp_r, lambda_exp_i)
        }
        y
    };
    let compute_cache = |parameters: &Vec<f64>, _t0: f64| {
        let n = parameters[0] as usize;
        let lambda_r = parameters[1];
        let lambda_i = parameters[2];
        let mut factorials = vec![1.; n + 1];
        let mut powers_r = vec![1.; n + 1];
        let mut powers_i = vec![0.; n + 1];
        let mut coefficient_r = vec![0.; n];
        let mut coefficient_i = vec![0.; n];
        let mut coef;
        let mut res;
        for i in 0..n {
            factorials[i + 1] = factorials[i] * (i as f64 + 1.);
            (powers_r[i + 1], powers_i[i + 1]) =
                complex_mult(powers_r[i], powers_i[i], lambda_r, lambda_i);
            for k in 0..=i {
                coef = (-1_f64).powi(k as i32) / (factorials[i - k] * factorials[k]);
                res = complex_mult(
                    powers_r[k],
                    powers_i[k],
                    parameters[1 + 2 * (n + k - i)],
                    parameters[2 + 2 * (n + k - i)],
                );
                coefficient_r[i] += coef * res.0;
                coefficient_i[i] += coef * res.1;
            }
        }
        vec![factorials, powers_r, powers_i, coefficient_r, coefficient_i]
    };

    let parameters_changer = |old_names: &Vec<String>,
                              old_parameters: &Vec<f64>,
                              mut parameters: Vec<f64>,
                              _old_t0: f64,
                              _t0: f64,
                              _cache: &Vec<Vec<f64>>| {
        if old_parameters[0] == parameters[0] {
            return (parameters, old_names.clone());
        }
        let n = parameters[0] as usize;
        let mut names = vec![
            old_names[0].clone(),
            old_names[1].clone(),
            old_names[2].clone(),
        ];
        names.append(&mut vec!["".to_string(); 2 * n]);
        let lambda_r = parameters[1];
        let lambda_i = parameters[2];
        parameters.truncate(3);
        parameters.append(&mut vec![0.; 2 * n]);
        parameters[1 + 2 * n] = 1.;
        names[1 + 2 * n] = format!["y{n} real part"];
        names[2 + 2 * n] = format!["y{n} imaginary part"];
        // defaults to constant polynomial 1
        for k in 1..n {
            (parameters[1 + 2 * (n - k)], parameters[2 + 2 * (n - k)]) = complex_mult(
                parameters[1 + 2 * (n - k + 1)],
                parameters[2 + 2 * (n - k + 1)],
                lambda_r,
                lambda_i,
            );
            names[1 + 2 * (n - k)] = format!["y{} real part", n - k];
            names[2 + 2 * (n - k)] = format!["y{} imaginary part", n - k];
        }
        return (parameters, names);
    };

    let cache = Vec::new();
    DynamicParameterSolvedIVP {
        differential_equation,
        solution,
        compute_cache,
        t0: 0.,
        parameters,
        parameters_names,
        ivp_name: "single root linear ivp".to_string(),
        cache, // 0 is factorial, root's powers real part, root's powers imaginary part, polynomial's coefficients real part, polynomial's coefficients imaginary part
        parameters_changer,
    }
}

fn complex_mult(a_r: f64, a_i: f64, b_r: f64, b_i: f64) -> (f64, f64) {
    (a_r * b_r - a_i * b_i, a_i * b_r + a_r * b_i)
}
