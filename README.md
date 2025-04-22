# Decentralised Control Algorithm for Dual Mobile Manipulators

A MATLAB implementation of a distributed auction algorithm for decentralised control of dual mobile manipulators in industrial assembly tasks, developed as part of my final year Mechatronics Engineering research/project at the University of Gloucestershire. The full ROS/Gazebo implemtation can be found ![here](https://github.com/knowingwings/dec_cont_BEng_Diss).

![Distributed Auction Algorithm Simulation](https://github.com/knowingwings/BEng_Diss-MATLAB/blob/main/figures/auction_convergence_plots/auction_algorithm_results.png)

## Features

- **Fully Decentralised Control**: Enables robust task allocation without any central coordinator
- **Distributed Auction Algorithm**: Market-based mechanism with provable convergence properties
- **Time-Weighted Consensus**: Robust information sharing with exponential convergence guarantees
- **Failure Recovery**: Self-healing system with autonomous task redistribution after robot failures
- **Communication Resilience**: Maintains performance under packet loss and communication delays
- **Adaptive Bidding Strategies**: Progressive price dynamics for efficient task allocation
- **Comprehensive Visualisation**: Performance metrics and real-time simulation visualisation

## Theory

The algorithm is based on:
- Distributed auction algorithm by Zavlanos et al. (2008)
- Consensus protocols by Olfati-Saber et al. (2007)
- Extensions for failure recovery and communication constraints

Key theoretical properties:
- Convergence guarantee in O(K² · bₘₐₓ/ε) iterations
- Solution quality within 2ε of optimal allocation
- Exponential consensus convergence with rate μ = -ln(1-2w)
- Recovery time bound of O(|Tᶠ|) + O(bₘₐₓ/ε)

## Installation

### Prerequisites
- MATLAB R2021b or later
- Statistics and Machine Learning Toolbox
- Optimization Toolbox (recommended)

### Setup
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/decentralised-control.git
   cd decentralised-control
   ```

2. Add all directories to MATLAB path:
   ```matlab
   addpath(genpath('./'));
   ```

## Usage

### Running the Main Simulation

```matlab
% Run the basic auction algorithm simulation
cd main
auction_algorithm_model
```

### Running Parameter Sensitivity Analysis

```matlab
% Run sensitivity analysis for various parameters
cd main
parameter_sensitivity_analysis
```

### Testing the Consensus Protocol and Failure Recovery

```matlab
% Test consensus and recovery mechanisms
cd main
consensus_and_failure_recovery
```

## Code Structure

```
decentralised_control/
├── main/
│   ├── auction_algorithm_model.m      % Main simulation script
│   ├── consensus_and_failure_recovery.m % Tests consensus and recovery
│   └── parameter_sensitivity_analysis.m % Parameter sensitivity tests
├── common/
│   ├── auction_utils.m                % Auction-related functions
│   ├── environment_utils.m            % Environment creation and visualisation
│   ├── robot_utils.m                  % Robot-related functionality
│   └── task_utils.m                   % Task generation and management
├── figures/                           % Generated figures
├── results/                           % Saved simulation results
└── tests/                             % Additional test scripts
```

### Key Components

- **auction_utils.m**: Contains the core distributed auction algorithm implementation
- **environment_utils.m**: Handles environment creation and visualisation
- **robot_utils.m**: Manages robot state, capabilities, and makespan calculations
- **task_utils.m**: Handles task creation, dependencies, and availability

## Algorithm Details

The distributed auction algorithm works as follows:

1. **Task Representation**: Tasks include position, execution time, required capabilities, and dependencies
2. **Bid Calculation**: Robots calculate bids based on distance, capabilities, workload, and other factors
3. **Price Dynamics**: Task prices increase through bidding, guiding allocation toward efficient solutions
4. **Consensus Protocol**: Time-weighted information sharing maintains consistent knowledge across robots
5. **Failure Recovery**: When a robot fails, its tasks are automatically redistributed based on criticality

### Advanced Features

- **Adaptive Batch Bidding**: Batch size adapts based on workload imbalance and unassigned tasks
- **Progressive Price Reduction**: Aggressive price reduction for persistently unassigned tasks
- **Exponential Bonus Scaling**: Special incentives for difficult-to-assign tasks
- **Communication Resilience**: Maintains performance despite packet loss and communication delays

## Results

The implementation demonstrates:

- Convergence to task assignments within 30-40 iterations for typical scenarios
- Resilience to up to 50% packet loss whilst maintaining 100% success rate
- Recovery from robot failures within 1-10 iterations (4× better than theoretical bound)
- Effective load balancing between robots
- Linear scaling of communication overhead with task count

![Performance Results](https://github.com/knowingwings/BEng_Diss-MATLAB/blob/main/figures/parameter_sensitivity_plots/epsilon_sensitivity.png)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
