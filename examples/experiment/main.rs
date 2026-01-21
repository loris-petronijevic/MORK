pub mod experiment_picker;
pub mod experiments;
pub mod plot;

use std::env;
use std::fs::create_dir;
use std::io::ErrorKind;

use crate::experiment_picker::*;
use crate::experiments::*;
use MORK::NDMORK::list::*;
use MORK::RK::list::*;
use MORK::Solver;
use eframe::egui;

#[allow(non_snake_case)]

fn main() {
    let solvers: Vec<(&str, Box<dyn Solver>)> = vec![
        ("MO explicit Euler", Box::new(MO_explicit_euler())),
        ("MO explicit midpoint", Box::new(MO_explicit_midpoint())),
        ("MO Ralston's", Box::new(MO_ralston())),
        ("MO Heun's", Box::new(MO_heun())),
        ("MO RK4", Box::new(MORK4())),
        ("MO RK4b", Box::new(MORK4b())),
        ("MO implicit Euler", Box::new(MO_implicit_euler())),
        ("MO implicit midpoint", Box::new(MO_implicit_midpoint())),
        ("MO Crank Nicolson", Box::new(MO_crank_nicolson())),
        ("MO Crank Nicolson b", Box::new(MO_CNb())),
        ("MO Gauss-Legendre", Box::new(MO_gauss_legendre())),
        ("explicit euler", Box::new(explicit_euler())),
        ("explicit midpoint", Box::new(explicit_midpoint())),
        ("Ralston's", Box::new(ralston())),
        ("Heun's", Box::new(heun())),
        ("Classical RK4", Box::new(RK4())),
        ("RK4b", Box::new(RK4b())),
        ("implicit euler", Box::new(implicit_euler())),
        ("implicit midpoint", Box::new(implicit_midpoint())),
        ("Crank Nicolson", Box::new(crank_nicolson())),
        ("Crank Nicolson b", Box::new(CNb())),
        ("Gauss Legendre", Box::new(gauss_legendre())),
    ];
    let experiments: Vec<Box<dyn Experiment>> = vec![
        Box::new(experiment_sin_cos()),
        Box::new(experiment_exp()),
        Box::new(experiment_linear_single_root()),
    ];
    let extractors: Vec<(&str, Box<dyn Fn(&Vec<Vec<f64>>) -> Vec<f64>>)> = vec![
        ("First coordinate", Box::new(first_coo_extractor)),
        (
            "First coordinate highest derivative",
            Box::new(highest_first_coo_extractor),
        ),
        (
            "First coordinate lowest derivative",
            Box::new(lowest_first_coo_extractor),
        ),
    ];
    let metrics: Vec<(
        &str,
        Box<dyn Fn(&Vec<Vec<f64>>, &Vec<Vec<f64>>) -> Vec<f64>>,
    )> = vec![
        ("First coordinate", Box::new(first_coo_metric)),
        (
            "First coordinate highest derivative",
            Box::new(highest_first_coo_metric),
        ),
        (
            "First coordinate lowest derivative",
            Box::new(lowest_first_coo_metric),
        ),
    ];
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([600.0, 800.0]),
        vsync: false,
        ..Default::default()
    };
    let mut path = env::current_dir().expect("couldn't get current directory");
    path.push("output");
    match create_dir(&path) {
        Ok(_) => {
            println!("Created folder \"output\" at {path:?}");
        }
        Err(error) => {
            if error.kind() == ErrorKind::AlreadyExists {
                println!("Folder \"output\" already existed at {path:?}");
            } else {
                panic!("Couldn't create folder \"output\" at {path:?}, error : \n {error:?}")
            }
        }
    };
    eframe::run_native(
        "Experiment selector",
        options,
        Box::new(|_cc| {
            Ok(Box::new(ExperimentPicker::new(
                path,
                experiments,
                solvers,
                extractors,
                metrics,
            )))
        }),
    )
    .unwrap();
}
