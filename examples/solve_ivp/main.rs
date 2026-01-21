//! A basic use case of the crate on a system of differential equations.

#[allow(unused_imports)]
use MORK::NDMORK::list::*;
#[allow(unused_imports)]
use MORK::RK::list::*;
use MORK::Solver;

fn main() {

	// Initial value problem
	let f = |_t:f64,y:&Vec<Vec<f64>>| vec![-2. * y[0][0] - y[0][1] - y[1][0], 2. * y[0][0] + 2. * y[0][1] + y[1][0]];
	let t0 = 0.;
	let y_initial = vec![vec![-1.,0.5],vec![1.]];

	// Choice of method
	let mut method = MORK4b();

	// Number of iterations and constant step size
	let iterations = 100;
	let h = 0.01;

	// Initialize approximations
	let mut y = y_initial.clone();
	
	// Aplies the method
	for _ in 0..iterations {
		y = method.approximate(t0, h, &f, &y);
	}

	let solution = |t: f64| vec![vec![-0.5 * ((-t).exp() + t.cos()), 0.5 * ((-t).exp() - t.sin())],vec![t.cos()]];

	let exact = solution(h * iterations as f64);
	
	let error = vec![vec![(exact[0][0]-y[0][0]).abs(),(exact[0][1]-y[0][1]).abs()],vec![(exact[1][0]-y[1][0]).abs()]];

	println!("Final approximations at {} are : {:?}", h * iterations as f64, y);
	println!("The exact solution is {:?}", exact);
	println!("The error is {:?}",error)
}