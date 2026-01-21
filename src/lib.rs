/*!
# Multi-Order Runge-Kutta methods (MORK)

`MORK` is an implementation of general multi-order Runge-Kutta methods ([GMORK]), node-determined Runge-Kutta methods ([NDMORK]) and Runge-Kutta methods ([RK]), as described in the paper ["Multi-order Runge-Kutta methods or how to numerically solve initial value problems of any order"](https://doi.org/10.48550/arXiv.2509.23513).
 
All methods implement the [Solver] trait, which requires the implementation of an [approximate][Solver::approximate] function. This function, given a differential equation function, initial instant, initial values, and a step size, returns the approximation of the method/struct which implements this trait. 
 
Some Runge-Kutta methods ([RK::list]) and node-determined multi-order Runge-Kutta methods ([NDMORK::list]) are already implemented.

# Usage

To approximate the solution of an initial value problem, we first need to define an initial value problem. We here consider a system of linear differential equations of dimension two, the initial value problem : 
```
d^2 y1(t) = f1(t,d y1(t), y1(t),y2 (t)) = - 2 d y1(t) - y1(t) - y2(t)
d y2(t) = f2(t,d y1(t), y1(t), y2(t)) = 2 d y1(t) + 2 y1(t) + y2(t)
t0 = 0
d y1(t0) = -1
y1(t0) = 0.5
y2(t0) = 1 
```
We define the function `f = (f1,f2)` and the initial values `y_initial = ((d y1(t0), y1(t0)), y2(t0))` :
```rust
let f = |_t:f64,y:&Vec<Vec<f64>>| vec![-2. * y[0][0] - y[0][1] - y[1][0], 2. * y[0][0] + 2. * y[0][1] + y[1][0]];
let t0 = 0.;
let y_initial = vec![vec![-1.,0.5],vec![1.]];
```

Keep in mind that the firs index of `y_initial` distinguishes the different entries `y1` and `y2`, the second index is for the derivative. The highest derivative is indexed using 0, and as the derivative decreases the index increases. We then choose the method we want to use, we here choose MORK4b :

```rust
let mut method = MORK4b(); 
```

For this example we will simply iterate a certain number of time the method with a constant step size, we hence define :

```rust
let iterations = 100;
let h = 0.01;
```

To apply the method we first initialize the approximations to the initial values, then use the [approximate][Solver::approximate] function, a function that all struct with the trait [Solver] implement.

```rust
let mut y = y_initial.clone();

for _ in 0..iterations {
    y = method.approximate(t0, h, &f, &y);
}
```

To measure the error of the approximation we first need the solution of the initial value problem. For this particular problem the solution is : 
```
d y1(t) = -1/2 (e^(-t) + cos(t))
y1(t) = 1/2 (e^(-t)-sin(t))
y2(t) = cos(t)
```
We hence define a solution function :
```rust
let solution = |t: f64| vec![vec![-0.5 * ((-t).exp() + t.cos()), 0.5 * ((-t).exp() - t.sin())],vec![t.cos()]];
```

We then compute the error at the final time :
```rust
let exact = solution(h * iterations as f64);
	
let error = vec![vec![(exact[0][0]-y[0][0]).abs(),(exact[0][1]-y[0][1]).abs()],vec![(exact[1][0]-y[1][0]).abs()]];
```
*/

#![allow(non_snake_case)]
pub mod GMORK;
pub mod NDMORK;
pub mod RK;
pub mod graph;

use crate::graph::*;

const ERROR_FRACTION: f64 = 0.001;
const MAX_ITER: u32 = 100;
const MIN_ITER: u32 = 100;

/// [Solver] is the used to indicate that a struct is a numerical scheme and can hence approximate the solution of an initial value problem.
pub trait Solver {
    /// Given a differential equation function, initial instant, initial values, and a step size, [approximate][Solver::approximate] returns the approximation of the method/struct which implements this trait.
    fn approximate(
        &mut self,
        t0: f64,
        h: f64,
        f: &dyn Fn(f64, &Vec<Vec<f64>>) -> Vec<f64>,
        y0: &Vec<Vec<f64>>,
    ) -> Vec<Vec<f64>>;
}

/// [enum@SCC] allows to distinguish between implicit and explicit strongly connected components.
#[derive(Debug, Clone)]
pub enum SCC {
    Implicit(Vec<usize>, Vec<usize>), // J and [|1,s|] without J
    Explicit(usize),
}

/// [create_computation_order] takes the maximum weight digraph of a method and outputs an order of computation.
pub fn create_computation_order(weight_graph: &Vec<Vec<bool>>) -> Vec<SCC> {
    let SCC = SCC(weight_graph);
    topological_sort(&contraction(weight_graph, &SCC))
        .into_iter()
        .map(|i| {
            if SCC[i].len() > 1 || weight_graph[SCC[i][0]][SCC[i][0]] {
                let J = SCC[i].clone();
                let comp_J = (0..(weight_graph.len() - 1))
                    .filter(|j| !J.contains(j))
                    .collect();
                SCC::Implicit(J, comp_J)
            } else {
                SCC::Explicit(SCC[i][0])
            }
        })
        .collect()
}