[MORK](https://github.com/loris-petronijevic/MORK) is a rust crate that implements multi-order Runge-Kutta methods and Runge-Kutta methods, as described in the paper "Multi-order Runge-Kutta methods or how to numerically solve initial value problems of any order" available on arXiv : https://arxiv.org/abs/2509.23513

This crate can be used in python using [MORKpy](https://github.com/loris-petronijevic/MORKpy).

See the example [solve_ivp](https://github.com/loris-petronijevic/MORK/tree/master/examples/solve_ivp) for a basic use case.

To read the documentation of this crate download the [doc folder](https://github.com/loris-petronijevic/MORK/tree/master/target), then navigate to the MORK folder and open index.html.

A program included in the code allows to test those methods on predefined initial value problems with a graphical interface :

<img width="375" height="378" alt="screenshot" src="https://github.com/user-attachments/assets/d59d845d-450e-4d70-9237-bb0baf81e1f7" />

To run this program one can either download the experiment executable in the latest release, or install the rust programming language and download the code, then use the command :

```
cargo run --release --example experiment
```

The first pannel allows to choose the initial value problem we want to consider and which method we want to use. The second pannel allows to choose between three options, Plot, Order, Measure.

- Plot : Allows to plot the approximations of a method with a constant step size sequence. The output is a cbor file with the raw data and a svg file with a basic plot of the approximations. There are multiple parameters:
  - Extractor : The function which takes the approximations of a step and outputs the quantity the user wants to plot on the graph
  - Plot solution if possible : If the initial value problems is programmed with the solution and this option is ticked, the program plots the solution and adds it to the raw data.
  - Step size : The step size of the constant step size sequence
  - Delta t : The instant at which the approximations stop
  - Other parameters : The parameters of the initial value problems
- Order : Allows to plot the error of a method for a range of step size. This option requires that the solution of the initial value problem is known. The output is a cbor file with the raw data and a svg file with a basic plot of the error. There are multiple parameters:
  - Metrics : The function which takes the approximations and the solution, and outputs the error
  - Left interval : The minimum step size
  - Right interval : The maximum step size
  - Samples : The number of step sizes to use between the value of Left interval and Right interval, the sample are taken uniformly on a logarithmic scale
  - Number of steps : The number of steps taken with each step size
  - Other parameters : The parameters of the initial value problems
- Measure : Measures the time the implementation takes to approximate the solution of an initial value problem. The output is a cbor file with the raw data and a message in the terminal with the average time taken by to approximate the solution.
  - Step size : The step size of the constant step size sequence
  - Number of steps : The number of steps taken
  - Number of repetitions : The number of times the sequence of approximations is repeated
  - Other parameters : The parameters of the initial value problems
