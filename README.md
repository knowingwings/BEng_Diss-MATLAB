# Distributed Auction Algorithm for Decentralized Mobile Manipulators

## Project Overview

This repository contains a MATLAB implementation of a distributed auction algorithm for decentralized control of dual mobile manipulators performing collaborative assembly tasks. The implementation is based on the theoretical framework established by Zavlanos et al. (2008), with significant extensions for handling communication constraints, task dependencies, collaborative tasks, and failure recovery.

The algorithm enables mobile manipulators to autonomously allocate tasks among themselves without central coordination, while maintaining provable convergence properties and bounded optimality gaps. This implementation is part of research exploring how mathematically rigorous distributed algorithms can enable robust and efficient collaborative assembly in industrial settings.

## Repository Structure

```
.
├── common/                 # Core algorithm and utility functions
│   ├── auction_utils.m     # Base auction algorithm implementation
│   ├── enhanced_auction_utils.m  # Enhanced version with failure recovery
│   ├── robot_utils.m       # Robot representation and kinematics
│   ├── task_utils.m        # Task management and dependencies
│   └── environment_utils.m # Environment visualization and setup
├── main/                   # Main scripts to run simulations
│   ├── auction_algorithm_model.m     # Basic auction simulation
│   ├── enhanced_auction_algorithm.m  # Enhanced version with recovery
│   ├── consensus_and_failure_recovery.m  # Tests for consensus protocols
│   ├── parameter_sensitivity_analysis.m  # Parameter optimization
│   ├── run_all_experiments.m    # Batch testing script
│   └── optimize_parameters.m    # Parameter optimization
├── utils/                  # Additional utility functions
│   ├── consensus_utils.m   # Time-weighted consensus protocol
│   ├── scheduler_utils.m   # Task scheduling and visualization
│   ├── visualization_utils.m  # Result visualization
│   └── statistical_analysis.m  # Statistical analysis of results
└── documentation/
    └── appendix_b-mathematical-foundations.pdf  # Theoretical background
```

## Installation and Requirements

### Prerequisites

- MATLAB R2021b or newer
- Optimization Toolbox (for parameter optimization)
- Statistics and Machine Learning Toolbox (for analysis)

### Setup

1. Clone this repository or download the source code
2. Add the repository root directory to your MATLAB path:
   ```matlab
   addpath(genpath('/path/to/distributed-auction-algorithm'));
   ```
3. Run the initialization script:
   ```matlab
   run('main/init.m');
   ```

## Usage Instructions

### Basic Usage

To run the basic auction algorithm simulation:

```matlab
% Add required paths
addpath('../common');

% Load utility functions
env_utils = environment_utils();
robot_utils = robot_utils();
task_utils = task_utils();
auction_utils = auction_utils();

% Create a simulation environment
env = env_utils.createEnvironment(4, 4);  % 4m x 4m workspace

% Create robots
robots = robot_utils.createRobots(2, env);

% Create tasks with dependencies
num_tasks = 10;
tasks = task_utils.createTasks(num_tasks, env);
tasks = task_utils.addTaskDependencies(tasks);

% Set algorithm parameters
params.epsilon = 0.05;        % Minimum bid increment
params.alpha = [0.8, 0.3, 1.0, 1.2, 0.2];  % Bid calculation weights
params.gamma = 0.5;           % Consensus weight factor
params.lambda = 0.1;          % Information decay rate

% Run the simulation
[metrics, converged] = auction_utils.runAuctionSimulation(params, env, robots, tasks, true);
```

### Running Enhanced Version with Failure Recovery

The enhanced version includes failure recovery and collaborative task handling:

```matlab
% Add required paths
addpath('../common');

% Load utility functions
enhanced_auction_utils = enhanced_auction_utils();
robot_utils = robot_utils();
task_utils = task_utils();
env_utils = environment_utils();

% Create environment, robots, and tasks as above

% Mark some tasks as collaborative
collaborative_tasks = [3, 5, 8]; % Example collaborative tasks
for i = collaborative_tasks
    tasks(i).collaborative = true;
end

% Set parameters including failure simulation
params.epsilon = 0.05;
params.alpha = [0.8, 0.3, 1.0, 1.2, 0.2];
params.beta = [2.0, 1.5, 1.0];  % Recovery auction weights
params.failure_time = 20;     % Robot fails at iteration 20
params.failed_robot = 1;      % Robot 1 will fail

% Run enhanced simulation
[metrics, converged] = enhanced_auction_utils.runEnhancedAuctionSimulation(params, env, robots, tasks, true);
```

### Running Experiments

To run a comprehensive set of experiments:

```matlab
results = run_full_experiment();
```

To optimize algorithm parameters:

```matlab
optimal_params = optimize_parameters();
```

## Key Features

### Core Auction Algorithm
- Distributed task allocation without central coordination
- Mathematically proven convergence to within 2ε of optimal solutions
- Adaptive bidding based on robot capabilities, workload, and task requirements

### Enhanced Extensions
- Time-weighted consensus protocol for maintaining consistency under communication delays
- Robust failure detection and recovery mechanisms
- Collaborative task handling with leader-follower coordination
- Task dependency management with topological ordering

### Performance Guarantees
- Convergence time bounded by O(K² · bₘₐₓ/ε) iterations
- Optimality gap bounded by 2ε
- Recovery time bounded by O(|Tᶠ|) + O(bₘₐₓ/ε)

## Mathematical Foundation

The theoretical foundation of this implementation is detailed in the `documentation/appendix_b-mathematical-foundations.pdf` document. Key mathematical concepts include:

- Graph theory for modeling communication networks
- Time-weighted consensus protocols with exponential convergence rates
- ε-complementary slackness conditions for optimality guarantees
- Leader-follower paradigm modeled as parallel timed automata

See the mathematical foundations document for complete proofs and derivations.

## Experiments and Testing

The codebase includes comprehensive testing facilities:

1. **Basic Convergence Testing**: Evaluates convergence rates and optimality gaps with different parameter values.
2. **Communication Constraints**: Tests algorithm performance under various communication delays and packet loss rates.
3. **Failure Recovery**: Simulates robot failures and measures recovery performance.
4. **Collaborative Tasks**: Examines coordination for tasks requiring multiple robots.
5. **Parameter Optimization**: Uses particle swarm optimization to find optimal parameter configurations.
6. **Full Factorial Experiment**: Systematically explores parameter interactions with statistical analysis.

## Troubleshooting

### Common Issues:

- **Missing Dependencies**: Ensure all required MATLAB toolboxes are installed.
- **Convergence Problems**: If the algorithm fails to converge:
  - Try increasing the maximum iterations (`max_iterations`) parameter
  - Adjust epsilon value (smaller values improve optimality but may slow convergence)
  - Check for cycles in task dependencies

- **Visualization Issues**: Some visualization functions require additional MATLAB toolboxes.

## References

1. Zavlanos, M.M., Spesivtsev, L. and Pappas, G.J. (2008). "A distributed auction algorithm for the assignment problem." IEEE Conference on Decision and Control, pp. 1212-1217.

2. Olfati-Saber, R., Fax, J.A. and Murray, R.M. (2007). "Consensus and Cooperation in Networked Multi-Agent Systems." Proceedings of the IEEE, 95(1), pp. 215-233.

3. Shorinwa, O., et al. (2023). "Distributed Optimization Methods for Multi-Robot Systems: Part 2—A Survey." IEEE robotics & automation magazine.

4. Talamali, M.S., et al. (2021). "When less is more: Robot swarms adapt better to changes with constrained communication." Science Robotics, 6(56), p. 1416.

## License

This project is part of academic research at the University of Gloucestershire. All rights reserved.

## Acknowledgements

This implementation is based on research for a dissertation in Mechatronics Engineering at the University of Gloucestershire.