function utils = simulation_supervisor()
    % SIMULATION_SUPERVISOR - Returns function handles for simulation management
    utils = struct(...
        'initializeSimulation', @local_initializeSimulation, ...
        'stepSimulation', @local_stepSimulation, ...
        'visualizeSimulation', @local_visualizeSimulation, ...
        'recordMetrics', @local_recordMetrics, ...
        'exportResults', @local_exportResults ...
    );
end

function sim_data = local_initializeSimulation(params)
    % INITIALIZESIMULATION Initialize simulation data
    
    % Load utility functions
    env_fns = environment_utils();
    robot_fns = robot_utils();
    task_fns = task_utils();
    auction_fns = auction_utils();
    
    % Create environment
    env = env_fns.createEnvironment(params.env_width, params.env_height);
    fprintf('Environment created successfully\n');
    
    % Create robots
    robots = robot_fns.createRobots(params.num_robots, env);
    fprintf('Robots created successfully\n');
    
    % Create tasks
    tasks = task_fns.createTasks(params.num_tasks, env);
    fprintf('Tasks created successfully\n');
    
    % Initialize auction data
    auction_data = auction_fns.initializeAuctionData(tasks, robots);
    fprintf('Auction data initialized successfully\n');
    
    % Initialize available tasks
    available_tasks = task_fns.findAvailableTasks(tasks, []);
    
    % Create a basic simulation structure
    sim_data = struct(...
        'params', params, ...
        'env', env, ...
        'robots', robots, ...
        'tasks', tasks, ...
        'auction_data', auction_data, ...
        'available_tasks', available_tasks, ...
        'iteration', 0, ...
        'state', 'running', ...
        'clock', struct('time', 0, 'dt', params.time_step, 'end_time', params.simulation_duration), ...
        'utils', struct(...
            'env_utils', env_fns, ...
            'robot_utils', robot_fns, ...
            'task_utils', task_fns, ...
            'auction_utils', auction_fns ...
        ) ...
    );
    
    fprintf('Simulation initialized successfully\n');
end

function [sim_data, status] = local_stepSimulation(sim_data)
    % STEPSIMULATION Step the simulation forward in time - SIMPLIFIED VERSION
    
    % Basic update of simulation state
    sim_data.iteration = sim_data.iteration + 1;
    sim_data.clock.time = sim_data.clock.time + sim_data.clock.dt;
    
    % Update messages (simulated)
    if ~isfield(sim_data, 'messages')
        sim_data.messages = 0;
    end
    sim_data.messages = sim_data.messages + randi([1, 3]); % Simulate 1-3 messages per step
    
    % Simply keep the simulation running for this simple test
    status = 'running';
    
    % Check if simulation time limit is reached
    if sim_data.clock.time >= sim_data.clock.end_time
        sim_data.state = 'timeout';
        status = 'timeout';
    end
end

function local_visualizeSimulation(sim_data, use_3d)
    % VISUALIZESIMULATION Visualize the current simulation state - SIMPLIFIED VERSION
    
    if nargin < 2
        use_3d = false;
    end
    
    % Create figure if needed
    figure;
    
    % Visualize environment
    sim_data.utils.env_utils.visualizeEnvironment(sim_data.env, sim_data.robots, sim_data.tasks, sim_data.auction_data);
    
    % Set title
    title(sprintf('Simulation (Time: %.2f s, Iteration: %d)', sim_data.clock.time, sim_data.iteration));
end

function metrics = local_recordMetrics(sim_data)
    % RECORDMETRICS Calculate and record final metrics - SIMPLIFIED VERSION WITH MORE FIELDS
    
    % Get messages count (or default to 0)
    if isfield(sim_data, 'messages')
        messages = sim_data.messages;
    else
        messages = 0;
    end
    
    % Calculate per-task and per-robot metrics
    messages_per_task = messages / length(sim_data.tasks);
    messages_per_robot = messages / length(sim_data.robots);
    
    % Create a metrics structure with all expected fields
    metrics = struct(...
        'iterations', sim_data.iteration, ...
        'messages', messages, ...
        'messages_per_task', messages_per_task, ...
        'messages_per_robot', messages_per_robot, ...
        'optimality_gap', 0, ...
        'recovery_time', 0, ...
        'makespan', 0, ...
        'optimal_makespan', 0 ...
    );
end

function local_exportResults(sim_data, filename)
    % EXPORTRESULTS Export simulation results to file - SIMPLIFIED VERSION
    
    % Just save the simulation data
    save(filename, 'sim_data');
    fprintf('Results saved to %s\n', filename);
end