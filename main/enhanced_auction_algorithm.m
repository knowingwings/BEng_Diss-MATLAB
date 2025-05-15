% Enhanced Distributed Auction Algorithm for Decentralized Control of Dual Mobile Manipulators
% Based on Zavlanos et al. (2008) with extensions for collaborative tasks and failure recovery

%% Setup and Configuration
clear all;
close all;
clc;

% Add utility functions to path
addpath('../common');

% Load utility functions
env_utils = environment_utils();
robot_utils = robot_utils();
task_utils = task_utils();
auction_utils = auction_utils();
enhanced_auction_utils = enhanced_auction_utils();

% Set random seed for reproducibility
rng(42);

% Algorithm parameters
params.epsilon = 0.05;        % Minimum bid increment
params.alpha = [1.0, 0.5, 2.0, 0.8, 0.3];  % Bid calculation weights
params.gamma = 0.5;           % Consensus weight factor
params.lambda = 0.1;          % Information decay rate
params.beta = [1.5, 1.0];     % Recovery auction weights
params.comm_delay = 0;        % Communication delay (in iterations)
params.packet_loss_prob = 0;  % Probability of packet loss (set to 0 for debugging)
params.failure_time = inf;    % Time of programmed robot failure (inf = no programmed failure)
params.failed_robot = [];     % Which robot has programmed failure

% Heartbeat and failure detection parameters
params.warmup_iterations = 10;     % Don't check heartbeats for first 10 iterations
params.heartbeat_timeout = 5;      % Wait 5 iterations before declaring failure
params.enable_progress_detection = false;  % Disable secondary detection initially

% Create a simulation environment
env = env_utils.createEnvironment(4, 4);  % 4m x 4m workspace

% Create robots
robots = robot_utils.createRobots(2, env);

% Create tasks with dependencies
num_tasks = 15;
tasks = task_utils.createTasks(num_tasks, env);

% Add task dependencies (create a DAG)
tasks = task_utils.addTaskDependencies(tasks);

% Mark some tasks as collaborative (requiring both robots)
num_collaborative = 3;
collaborative_tasks = sort(randperm(num_tasks, num_collaborative));
for i = collaborative_tasks
    tasks(i).collaborative = true;
end

% Run the simulation
[metrics, converged] = enhanced_auction_utils.runEnhancedAuctionSimulation(params, env, robots, tasks, true);

% Save results
if ~exist('../results', 'dir')
    mkdir('../results');
end
save('../results/enhanced_auction_results.mat', 'metrics', 'params', 'tasks', 'robots');

% Save figure
if ~exist('../figures/enhanced_auction_plots', 'dir')
    mkdir('../figures/enhanced_auction_plots');
end
saveas(gcf, '../figures/enhanced_auction_plots/enhanced_auction_results.fig');
saveas(gcf, '../figures/enhanced_auction_plots/enhanced_auction_results.png');

fprintf('\nSimulation complete. Results saved to enhanced_auction_results.mat\n');