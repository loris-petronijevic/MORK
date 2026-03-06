use crate::experiments::Experiment;
use crate::plot::{log_plot, plot};
use MORK::Solver;
use eframe::egui::{self, Color32, RichText, Spinner, Widget};
use plotters::style::Color;
use plotters::style::{Palette, Palette99};

use serde_cbor::to_vec;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

const SOLUTION_ERROR_MSG: &str =
    "Experiment was identified as a solved experiment but could not provide a solution";

#[derive(Debug, PartialEq, Clone)]
enum Goal {
    Order,
    Plot,
    Timer,
    Convergence,
}

pub struct OrderParameters {
    left_h_interval: f64,
    right_h_interval: f64,
    samples: u32,
    steps: u32,
}

impl Default for OrderParameters {
    fn default() -> Self {
        OrderParameters {
            left_h_interval: 0.0001,
            right_h_interval: 1.,
            samples: 20,
            steps: 1,
        }
    }
}

pub struct PlotParameters {
    plot_solution: bool,
    step_size: f64,
    delta_t: f64,
}

impl Default for PlotParameters {
    fn default() -> Self {
        PlotParameters {
            plot_solution: true,
            step_size: 0.1,
            delta_t: 10.,
        }
    }
}

pub struct TimerParameters {
    step_size: f64,
    repeats: u32,
    steps: u32,
}

impl Default for TimerParameters {
    fn default() -> Self {
        TimerParameters {
            step_size: 0.1,
            steps: 100,
            repeats: 100,
        }
    }
}

pub struct ConvergenceParameters {
    left_iterations_interval: u32,
    right_iterations_interval: u32,
    delta_t: f64,
    samples: u32,
}

impl Default for ConvergenceParameters {
    fn default() -> Self {
        ConvergenceParameters {
            left_iterations_interval: 10,
            right_iterations_interval: 10000,
            delta_t: 10.,
            samples: 20,
        }
    }
}

pub struct ExperimentPicker {
    experiments: Vec<Box<dyn Experiment>>,
    solvers: Vec<(String, Box<dyn Solver>)>,
    extractors: Vec<(String, Box<dyn Fn(&Vec<Vec<f64>>) -> Vec<f64>>)>,
    metrics: Vec<(
        String,
        Box<dyn Fn(&Vec<Vec<f64>>, &Vec<Vec<f64>>) -> Vec<f64>>,
    )>,
    experiment_index: usize,
    solvers_selected: Vec<bool>,
    extractor_index: usize,
    metrics_index: usize,
    goal: Goal,
    order_parameters: OrderParameters,
    plot_parameters: PlotParameters,
    timer_parameters: TimerParameters,
    convergence_parameters: ConvergenceParameters,
    order_error: bool,
    plot_solution_error: bool,
    path: PathBuf,
}

impl ExperimentPicker {
    pub fn new(
        path: PathBuf,
        experiments: Vec<Box<dyn Experiment>>,
        solvers: Vec<(impl ToString, Box<dyn Solver>)>,
        extractors: Vec<(impl ToString, Box<dyn Fn(&Vec<Vec<f64>>) -> Vec<f64>>)>,
        distances: Vec<(
            impl ToString,
            Box<dyn Fn(&Vec<Vec<f64>>, &Vec<Vec<f64>>) -> Vec<f64>>,
        )>,
    ) -> Self {
        let len_solv = solvers.len();
        ExperimentPicker {
            experiments,
            solvers: solvers
                .into_iter()
                .map(|e| (e.0.to_string(), e.1))
                .collect(),
            extractors: extractors
                .into_iter()
                .map(|e| (e.0.to_string(), e.1))
                .collect(),
            metrics: distances
                .into_iter()
                .map(|e| (e.0.to_string(), e.1))
                .collect(),
            experiment_index: 0,
            solvers_selected: vec![false; len_solv],
            extractor_index: 0,
            metrics_index: 0,
            goal: Goal::Plot,
            order_parameters: OrderParameters::default(),
            plot_parameters: PlotParameters::default(),
            timer_parameters: TimerParameters::default(),
            convergence_parameters: ConvergenceParameters::default(),
            order_error: false,
            plot_solution_error: false,
            path,
        }
    }
}

pub fn order_experiment(
    parameters: &OrderParameters,
    experiment: &mut Box<dyn Experiment>,
    solvers: &mut Vec<(String, Box<dyn Solver>)>,
    solvers_selected: &Vec<bool>,
    distance: &Box<dyn Fn(&Vec<Vec<f64>>, &Vec<Vec<f64>>) -> Vec<f64>>,
    t0: f64,
    y0: Vec<Vec<f64>>,
    plot_path: PathBuf,
    cbor_path: PathBuf,
    title: String,
) {
    let mut names = Vec::new();
    let mut data: Vec<Vec<(f64, f64)>> = Vec::new();

    let ratio = parameters.right_h_interval / parameters.left_h_interval;
    let h_list: Vec<f64> = (0..parameters.samples)
        .map(|i| {
            parameters.left_h_interval * ratio.powf(i as f64 / (parameters.samples - 1) as f64)
        })
        .collect();
    let solution: Vec<Vec<Vec<f64>>> = h_list
        .iter()
        .map(|&h| {
            experiment
                .solution(t0 + parameters.steps as f64 * h)
                .expect(SOLUTION_ERROR_MSG)
        })
        .collect();
    let f: &dyn Fn(f64, &Vec<Vec<f64>>) -> Vec<f64> =
        &|t: f64, y: &Vec<Vec<f64>>| experiment.differential_equation(t, y);

    // Apply selected solvers
    for solver_i in (0..solvers_selected.len()).filter(|i| solvers_selected[*i]) {
        let temp: Vec<Vec<f64>> = h_list
            .iter()
            .enumerate()
            .map(|(i, &h)| {
                let mut y = y0.clone();
                let mut t = t0;
                for _ in 0..parameters.steps {
                    y = solvers[solver_i].1.approximate(t, h, &f, &y);
                    t += h
                }
                distance(&y, &solution[i])
            })
            .collect();
        for k in 0..temp[0].len() {
            data.push(
                (0..parameters.samples as usize)
                    .map(|q| (h_list[q], temp[q][k]))
                    .filter(|(x, y)| *x != 0. && *y != 0.)
                    .collect(),
            );
            names.push(Some(solvers[solver_i].0.clone()));
        }
    }

    // Encode data in cbor file and save it
    let cbor_data: Vec<(Vec<f64>, Vec<f64>)> = (0..data.len())
        .map(|q| {
            (
                (0..data[q].len()).map(|k| data[q][k].0).collect(),
                (0..data[q].len()).map(|k| data[q][k].1).collect(),
            )
        })
        .collect();
    let encoded = to_vec(&cbor_data).expect("failed serialization");
    let mut file = File::create(cbor_path).unwrap();
    file.write_all(&encoded).unwrap();

    // Plot the data
    log_plot(
        &data,
        &(0..data.len())
            .map(|i| Palette99::pick(i).stroke_width(7))
            .collect(),
        &names,
        &plot_path,
        title.to_string(),
        ("\u{0394} t".to_string(), "e".to_string()),
    );
}

pub fn plot_experiment(
    parameters: &PlotParameters,
    experiment: &mut Box<dyn Experiment>,
    solvers: &mut Vec<(String, Box<dyn Solver>)>,
    solvers_selected: &Vec<bool>,
    extractor: &Box<dyn Fn(&Vec<Vec<f64>>) -> Vec<f64>>,
    t0: f64,
    y0: Vec<Vec<f64>>,
    plot_path: PathBuf,
    cbor_path: PathBuf,
    title: String,
) -> bool {
    let mut plot_solution_error = false;

    let mut names = Vec::new();
    let mut data = Vec::new();

    let f: &dyn Fn(f64, &Vec<Vec<f64>>) -> Vec<f64> =
        &|t: f64, y: &Vec<Vec<f64>>| experiment.differential_equation(t, y);
    let q = (parameters.delta_t / parameters.step_size).round() as usize;
    let t: Vec<f64> = (0..=q)
        .map(|q1| t0 + parameters.step_size * q1 as f64)
        .collect();

    if parameters.plot_solution {
        if experiment.is_solved() {
            let y: Vec<Vec<f64>> = t
                .iter()
                .map(|&t1| extractor(&experiment.solution(t1).expect(SOLUTION_ERROR_MSG)))
                .collect();
            for k in 0..y[0].len() {
                let y1: Vec<(f64, f64)> =
                    t.iter().enumerate().map(|(i, t1)| (*t1, y[i][k])).collect();
                data.push(y1);
                names.push(Some("Solution".to_string() + &k.to_string()));
            }
        } else {
            plot_solution_error = true
        }
    }

    for (_, (solver_name, solver)) in solvers
        .iter_mut()
        .enumerate()
        .filter(|x| solvers_selected[x.0])
    {
        let mut y: Vec<Vec<Vec<f64>>> = t.iter().map(|_| y0.clone()).collect();
        for k in 0..q {
            y[k + 1] = solver.approximate(t[k], parameters.step_size, &f, &y[k]);
        }
        let y1: Vec<Vec<f64>> = y.into_iter().map(|x| extractor(&x)).collect();
        for k in 0..y1[0].len() {
            let y2 = t
                .iter()
                .enumerate()
                .map(|(i, t1)| (*t1, y1[i][k]))
                .collect();
            data.push(y2);
            names.push(Some(solver_name.clone() + &k.to_string()));
        }
    }

    let cbor_data: Vec<(Vec<f64>, Vec<f64>)> = (0..data.len())
        .map(|q| {
            (
                (0..data[q].len()).map(|k| data[q][k].0).collect(),
                (0..data[q].len()).map(|k| data[q][k].1).collect(),
            )
        })
        .collect();
    let encoded = to_vec(&cbor_data).expect("failed serialization");
    let mut file = File::create(cbor_path).unwrap();
    file.write_all(&encoded).unwrap();

    plot(
        &data,
        &(0..data.len())
            .map(|i| Palette99::pick(i).stroke_width(7))
            .collect(),
        &names,
        &plot_path,
        title,
        ("t".to_string(), "y".to_string()),
    );
    return plot_solution_error;
}

pub fn timer_experiment(
    parameters: &TimerParameters,
    experiment: &mut Box<dyn Experiment>,
    solvers: &mut Vec<(String, Box<dyn Solver>)>,
    solvers_selected: &Vec<bool>,
    t0: f64,
    y0: Vec<Vec<f64>>,
    cbor_path: PathBuf,
) {
    let f: &dyn Fn(f64, &Vec<Vec<f64>>) -> Vec<f64> =
        &|t: f64, y: &Vec<Vec<f64>>| experiment.differential_equation(t, y);
    let mut data: Vec<Vec<u128>> = Vec::new();
    let t: Vec<f64> = (0..=parameters.steps)
        .map(|q1| t0 + parameters.step_size * q1 as f64)
        .collect();
    for (_, (solver_name, solver)) in solvers
        .iter_mut()
        .enumerate()
        .filter(|(k, _)| solvers_selected[*k])
    {
        data.push(
            (0..(parameters.repeats as usize))
                .map(|_| {
                    let mut y: Vec<Vec<Vec<f64>>> = t.iter().map(|_| y0.clone()).collect();
                    let time = Instant::now();
                    for k in 0..(parameters.steps as usize) {
                        y[k + 1] = solver.approximate(t[k], parameters.step_size, &f, &y[k]);
                    }
                    time.elapsed().as_nanos()
                })
                .collect(),
        );
        let l = data.len() - 1;
        println!(
            "Average time for {solver_name} : {:#?} nanos",
            (data[l].iter().sum::<u128>() / (data[l].len() as u128)) //pretty print formating
                .to_string()
                .as_bytes()
                .rchunks(3)
                .rev()
                .map(std::str::from_utf8)
                .collect::<Result<Vec<&str>, _>>()
                .unwrap()
                .join(" ")
        );
    }
    let encoded = to_vec(&data).expect("failed serialization");
    let mut file = File::create(cbor_path).unwrap();
    file.write_all(&encoded).unwrap();
}

pub fn convergence_experiment(
    parameters: &ConvergenceParameters,
    experiment: &mut Box<dyn Experiment>,
    solvers: &mut Vec<(String, Box<dyn Solver>)>,
    solvers_selected: &Vec<bool>,
    distance: &Box<dyn Fn(&Vec<Vec<f64>>, &Vec<Vec<f64>>) -> Vec<f64>>,
    t0: f64,
    y0: Vec<Vec<f64>>,
    plot_path: PathBuf,
    cbor_path: PathBuf,
    title: String,
) {
    let mut names = Vec::new();
    let mut data: Vec<Vec<(f64, f64)>> = Vec::new();

    let ratio =
        parameters.right_iterations_interval as f64 / parameters.left_iterations_interval as f64;
    #[allow(non_snake_case)]
    let N_list: Vec<u32> = (0..parameters.samples)
        .map(|i| {
            (parameters.left_iterations_interval as f64
                * ratio.powf(i as f64 / (parameters.samples - 1) as f64))
            .round() as u32
        })
        .collect();
    let f: &dyn Fn(f64, &Vec<Vec<f64>>) -> Vec<f64> =
        &|t: f64, y: &Vec<Vec<f64>>| experiment.differential_equation(t, y);

    // Apply selected solvers
    for solver_i in (0..solvers_selected.len()).filter(|i| solvers_selected[*i]) {
        let temp: Vec<Vec<f64>> = N_list
            .iter()
            .enumerate()
            .map(|(_i, &iterations)| {
                let mut y = y0.clone();
                let h = parameters.delta_t / iterations as f64;
                let mut d: Vec<f64> = distance(&y0,&y0);
                let mut solution: Vec<Vec<f64>>;
                for k in 0..iterations {
                    y = solvers[solver_i]
                        .1
                        .approximate(t0 + h * k as f64, h, &f, &y);
                    solution = experiment
                        .solution(t0 + h * k as f64)
                        .expect(SOLUTION_ERROR_MSG);
                    for i in 0..d.len() {
                        d[i] = d[i].max(distance(&y, &solution)[i]);
                    }
                }
                d
            })
            .collect();
        for k in 0..temp[0].len() {
            data.push(
                (0..parameters.samples as usize)
                    .map(|q| (N_list[q] as f64, temp[q][k]))
                    .filter(|(x, y)| *x != 0. && *y != 0.)
                    .collect(),
            );
            names.push(Some(solvers[solver_i].0.clone()));
        }
    }

    // Encode data in cbor file and save it
    let cbor_data: Vec<(Vec<f64>, Vec<f64>)> = (0..data.len())
        .map(|q| {
            (
                (0..data[q].len()).map(|k| data[q][k].0).collect(),
                (0..data[q].len()).map(|k| data[q][k].1).collect(),
            )
        })
        .collect();
    let encoded = to_vec(&cbor_data).expect("failed serialization");
    let mut file = File::create(cbor_path).unwrap();
    file.write_all(&encoded).unwrap();

    // Plot the data
    log_plot(
        &data,
        &(0..data.len())
            .map(|i| Palette99::pick(i).stroke_width(7))
            .collect(),
        &names,
        &plot_path,
        title.to_string(),
        ("N".to_string(), "e".to_string()),
    );
}

impl eframe::App for ExperimentPicker {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::SidePanel::left("exp_solver").show(ctx, |ui| {
            // Select initial value problem
            egui::ComboBox::from_label("Experiments")
                .selected_text(&self.experiments[self.experiment_index].name())
                .show_ui(ui, |ui| {
                    for i in 0..self.experiments.len() {
                        ui.selectable_value(
                            &mut self.experiment_index,
                            i,
                            &self.experiments[i].name(),
                        );
                    }
                });
            // list of solvers
            for i in 0..self.solvers.len() {
                ui.checkbox(&mut self.solvers_selected[i], &self.solvers[i].0);
            }
        });
        egui::SidePanel::left("options").show(ctx, |ui| {
            // select goal
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.goal, Goal::Plot, "Plot");
                ui.selectable_value(&mut self.goal, Goal::Order, "Order");
                ui.selectable_value(&mut self.goal, Goal::Timer, "Timer");
                ui.selectable_value(&mut self.goal, Goal::Convergence, "Convergence");
            });
            // Options according to goal
            match &self.goal {
                Goal::Order => {
                    // select a metric
                    egui::ComboBox::from_label("Metrics")
                        .selected_text(&self.metrics[self.metrics_index].0)
                        .show_ui(ui, |ui| {
                            for i in 0..self.metrics.len() {
                                ui.selectable_value(&mut self.metrics_index, i, &self.metrics[i].0);
                            }
                        });
                    ui.label("Samples");
                    ui.add(
                        egui::DragValue::new(&mut self.order_parameters.samples).max_decimals(0),
                    );
                    ui.label("Left interval");
                    ui.add(
                        egui::DragValue::new(&mut self.order_parameters.left_h_interval)
                            .max_decimals(16)
                            .min_decimals(16),
                    );
                    ui.label("Right interval");
                    ui.add(
                        egui::DragValue::new(&mut self.order_parameters.right_h_interval)
                            .max_decimals(16)
                            .min_decimals(16),
                    );

                    ui.label("number of steps");
                    ui.add(egui::DragValue::new(&mut self.order_parameters.steps).max_decimals(0));
                }

                Goal::Plot => {
                    // select data extractor
                    egui::ComboBox::from_label("Extractor")
                        .selected_text(&self.extractors[self.extractor_index].0)
                        .show_ui(ui, |ui| {
                            for i in 0..self.extractors.len() {
                                ui.selectable_value(
                                    &mut self.extractor_index,
                                    i,
                                    &self.extractors[i].0,
                                );
                            }
                        });
                    ui.checkbox(
                        &mut self.plot_parameters.plot_solution,
                        "Plot solution if possible",
                    );
                    ui.label("Step size");
                    ui.add(
                        egui::DragValue::new(&mut self.plot_parameters.step_size)
                            .max_decimals(16)
                            .min_decimals(16),
                    );
                    ui.label("Delta t");
                    ui.add(
                        egui::DragValue::new(&mut self.plot_parameters.delta_t)
                            .max_decimals(16)
                            .min_decimals(16),
                    );
                }

                Goal::Timer => {
                    ui.label("Step size");
                    ui.add(
                        egui::DragValue::new(&mut self.timer_parameters.step_size)
                            .max_decimals(16)
                            .min_decimals(16),
                    );
                    ui.label("Number of steps");
                    ui.add(egui::DragValue::new(&mut self.timer_parameters.steps).max_decimals(0));
                    ui.label("Number of repeatitions");
                    ui.add(
                        egui::DragValue::new(&mut self.timer_parameters.repeats).max_decimals(0),
                    );
                }

                Goal::Convergence => {
                    // select a metric
                    egui::ComboBox::from_label("Metrics")
                        .selected_text(&self.metrics[self.metrics_index].0)
                        .show_ui(ui, |ui| {
                            for i in 0..self.metrics.len() {
                                ui.selectable_value(&mut self.metrics_index, i, &self.metrics[i].0);
                            }
                        });
                    ui.label("Samples");
                    ui.add(
                        egui::DragValue::new(&mut self.convergence_parameters.samples)
                            .max_decimals(0),
                    );
                    ui.label("Left interval");
                    ui.add(
                        egui::DragValue::new(
                            &mut self.convergence_parameters.left_iterations_interval,
                        )
                        .max_decimals(0),
                    );
                    ui.label("Right interval");
                    ui.add(
                        egui::DragValue::new(
                            &mut self.convergence_parameters.right_iterations_interval,
                        )
                        .max_decimals(0),
                    );

                    ui.label("Delta t");
                    ui.add(
                        egui::DragValue::new(&mut self.convergence_parameters.delta_t)
                            .max_decimals(16)
                            .min_decimals(16),
                    );
                }
            }
            // change parameters of experiment
            let (mut parameters_values, parameters_names) =
                self.experiments[self.experiment_index].get_parameters();
            for (value, name) in parameters_values.iter_mut().zip(parameters_names.iter()) {
                ui.label(name);
                ui.add(egui::DragValue::new(value));
            }
            self.experiments[self.experiment_index].change_parameters(parameters_values);
        });
        egui::CentralPanel::default().show(ctx, |ui| {
            // runs the experiment
            if ui.button("Run").clicked() {
                let experiment = &mut self.experiments[self.experiment_index];
                experiment.apply_parameters();
                self.plot_solution_error = false;
                self.order_error = false;
                let spinner = Spinner::default();
                spinner.ui(ui);
                let experiment_name = experiment.name();
                let (t0, y0) = experiment.initial_values();
                match &self.goal {
                    Goal::Order => {
                        if !experiment.is_solved() {
                            self.order_error = true;
                        } else {
                            let mut plot_path = self.path.clone();
                            plot_path.push("order.svg");
                            let mut cbor_path = self.path.clone();
                            cbor_path.push("order.cbor");
                            let title = format!("Order - {}", experiment_name);
                            let distance: &Box<dyn Fn(&Vec<Vec<f64>>, &Vec<Vec<f64>>) -> Vec<f64>> =
                                &self.metrics[self.metrics_index].1;
                            order_experiment(
                                &self.order_parameters,
                                experiment,
                                &mut self.solvers,
                                &self.solvers_selected,
                                distance,
                                t0,
                                y0,
                                plot_path,
                                cbor_path,
                                title,
                            );
                        }
                    }

                    Goal::Plot => {
                        let mut plot_path = self.path.clone();
                        plot_path.push("plot.svg");
                        let mut cbor_path = self.path.clone();
                        cbor_path.push("plot.cbor");
                        let title = format!("Plot - {}", experiment_name);
                        let extractor = &self.extractors[self.extractor_index].1;
                        self.plot_solution_error = plot_experiment(
                            &self.plot_parameters,
                            experiment,
                            &mut self.solvers,
                            &self.solvers_selected,
                            &extractor,
                            t0,
                            y0,
                            plot_path,
                            cbor_path,
                            title,
                        );
                    }

                    Goal::Timer => {
                        let mut cbor_path = self.path.clone();
                        cbor_path.push("timer.cbor");
                        timer_experiment(
                            &self.timer_parameters,
                            experiment,
                            &mut self.solvers,
                            &self.solvers_selected,
                            t0,
                            y0,
                            cbor_path,
                        );
                    }

                    Goal::Convergence => {
                        if !experiment.is_solved() {
                            self.order_error = true;
                        } else {
                            let mut plot_path = self.path.clone();
                            plot_path.push("convergence.svg");
                            let mut cbor_path = self.path.clone();
                            cbor_path.push("convergence.cbor");
                            let title = format!("Convergence - {}", experiment_name);
                            let distance: &Box<dyn Fn(&Vec<Vec<f64>>, &Vec<Vec<f64>>) -> Vec<f64>> =
                                &self.metrics[self.metrics_index].1;
                            convergence_experiment(
                                &self.convergence_parameters,
                                experiment,
                                &mut self.solvers,
                                &self.solvers_selected,
                                distance,
                                t0,
                                y0,
                                plot_path,
                                cbor_path,
                                title,
                            );
                        }
                    }
                }
            }
            if self.order_error {
                ui.label(RichText::new("Need a solved ivp to show order").color(Color32::RED));
            }
            if self.plot_solution_error {
                ui.label(RichText::new("Need a solved ivp to show solution").color(Color32::RED));
            }
            if ui.button("quit").clicked() {
                ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
            }
        });
    }
    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {}
}
