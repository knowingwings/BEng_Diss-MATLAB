% industry_scenarios.m
% Industry-derived scenarios for distributed auction algorithm validation
% Implements realistic automotive assembly task datasets and industrial scenarios

function utils = industry_scenarios()
    % INDUSTRY_SCENARIOS - Returns function handles for industry-specific scenarios
    utils = struct(...
        'createAutomotiveAssemblyTasks', @local_createAutomotiveAssemblyTasks, ...
        'createElectronicsAssemblyTasks', @local_createElectronicsAssemblyTasks, ...
        'createFurnitureAssemblyTasks', @local_createFurnitureAssemblyTasks, ...
        'applyIndustrialTolerances', @local_applyIndustrialTolerances, ...
        'generateTaskDependencyGraph', @local_generateTaskDependencyGraph, ...
        'visualizeIndustrialScenario', @local_visualizeIndustrialScenario, ...
        'simulateEnvironmentalDisturbances', @local_simulateEnvironmentalDisturbances, ...
        'runIndustrialScenario', @local_runIndustrialScenario ...
    );
end

function tasks = local_createAutomotiveAssemblyTasks(env, difficulty_level)
    % CREATEAUTOMOTIVEASSEMBLYTASKS - Create tasks based on automotive assembly processes
    %
    % Parameters:
    %   env - Environment structure with width and height properties
    %   difficulty_level - Level of scenario complexity (1-4, default: 2)
    %
    % Returns:
    %   tasks - Array of task structures with realistic automotive assembly characteristics
    
    % Set default difficulty level
    if nargin < 2
        difficulty_level = 2;
    end
    
    fprintf('Creating automotive assembly scenario (difficulty level %d)...\n', difficulty_level);
    
    % Define automotive assembly components based on difficulty level
    switch difficulty_level
        case 1 % Simple, 8 tasks
            components = {'chassis_positioning', 'wheel_assembly_FL', 'wheel_assembly_FR', ...
                         'wheel_assembly_RL', 'wheel_assembly_RR', 'dashboard_positioning', ...
                         'seat_front_L', 'seat_front_R'};
            num_tasks = 8;
        case 2 % Medium, 12 tasks
            components = {'chassis_positioning', 'engine_positioning', 'gearbox_positioning', ...
                         'wheel_assembly_FL', 'wheel_assembly_FR', 'wheel_assembly_RL', ...
                         'wheel_assembly_RR', 'dashboard_positioning', 'seat_front_L', ...
                         'seat_front_R', 'seat_rear', 'steering_wheel'};
            num_tasks = 12;
        case 3 % Complex, 16 tasks
            components = {'chassis_positioning', 'engine_positioning', 'gearbox_positioning', ...
                         'wheel_assembly_FL', 'wheel_assembly_FR', 'wheel_assembly_RL', ...
                         'wheel_assembly_RR', 'dashboard_positioning', 'seat_front_L', ...
                         'seat_front_R', 'seat_rear_L', 'seat_rear_R', 'steering_wheel', ...
                         'door_assembly_L', 'door_assembly_R', 'roof_assembly'};
            num_tasks = 16;
        case 4 % Very Complex, 20 tasks
            components = {'chassis_positioning', 'engine_positioning', 'gearbox_positioning', ...
                         'exhaust_system', 'cooling_system', 'wheel_assembly_FL', 'wheel_assembly_FR', ...
                         'wheel_assembly_RL', 'wheel_assembly_RR', 'dashboard_positioning', ...
                         'seat_front_L', 'seat_front_R', 'seat_rear_L', 'seat_rear_R', ...
                         'steering_wheel', 'door_assembly_FL', 'door_assembly_FR', ...
                         'door_assembly_RL', 'door_assembly_RR', 'roof_assembly'};
            num_tasks = 20;
        otherwise
            error('Invalid difficulty level. Choose 1-4.');
    end
    
    % Define task execution times (industry-realistic values in minutes)
    execution_times = struct();
    execution_times.chassis_positioning = 5.2;
    execution_times.engine_positioning = 7.8;
    execution_times.gearbox_positioning = 6.5;
    execution_times.exhaust_system = 4.3;
    execution_times.cooling_system = 3.7;
    execution_times.wheel_assembly_FL = 2.1;
    execution_times.wheel_assembly_FR = 2.1;
    execution_times.wheel_assembly_RL = 2.1;
    execution_times.wheel_assembly_RR = 2.1;
    execution_times.dashboard_positioning = 5.8;
    execution_times.seat_front_L = 1.8;
    execution_times.seat_front_R = 1.8;
    execution_times.seat_rear = 2.5;
    execution_times.seat_rear_L = 1.9;
    execution_times.seat_rear_R = 1.9;
    execution_times.steering_wheel = 1.2;
    execution_times.door_assembly_L = 3.4;
    execution_times.door_assembly_R = 3.4;
    execution_times.door_assembly_FL = 3.4;
    execution_times.door_assembly_FR = 3.4;
    execution_times.door_assembly_RL = 3.4;
    execution_times.door_assembly_RR = 3.4;
    execution_times.roof_assembly = 4.6;
    
    % Define capability requirements (5-dimensional vector for each task)
    % Dimensions: [precision, strength, dexterity, speed, coordination]
    capability_requirements = struct();
    capability_requirements.chassis_positioning = [0.7, 1.0, 0.5, 0.3, 0.8];
    capability_requirements.engine_positioning = [0.8, 1.0, 0.7, 0.4, 0.9];
    capability_requirements.gearbox_positioning = [0.9, 0.8, 0.7, 0.4, 0.9];
    capability_requirements.exhaust_system = [0.7, 0.6, 0.8, 0.5, 0.7];
    capability_requirements.cooling_system = [0.8, 0.6, 0.7, 0.5, 0.7];
    capability_requirements.wheel_assembly_FL = [0.5, 0.7, 0.6, 0.8, 0.6];
    capability_requirements.wheel_assembly_FR = [0.5, 0.7, 0.6, 0.8, 0.6];
    capability_requirements.wheel_assembly_RL = [0.5, 0.7, 0.6, 0.8, 0.6];
    capability_requirements.wheel_assembly_RR = [0.5, 0.7, 0.6, 0.8, 0.6];
    capability_requirements.dashboard_positioning = [0.9, 0.5, 0.8, 0.4, 0.9];
    capability_requirements.seat_front_L = [0.5, 0.6, 0.7, 0.7, 0.5];
    capability_requirements.seat_front_R = [0.5, 0.6, 0.7, 0.7, 0.5];
    capability_requirements.seat_rear = [0.5, 0.7, 0.6, 0.7, 0.5];
    capability_requirements.seat_rear_L = [0.5, 0.6, 0.7, 0.7, 0.5];
    capability_requirements.seat_rear_R = [0.5, 0.6, 0.7, 0.7, 0.5];
    capability_requirements.steering_wheel = [0.8, 0.3, 0.9, 0.6, 0.7];
    capability_requirements.door_assembly_L = [0.7, 0.8, 0.8, 0.6, 0.8];
    capability_requirements.door_assembly_R = [0.7, 0.8, 0.8, 0.6, 0.8];
    capability_requirements.door_assembly_FL = [0.7, 0.8, 0.8, 0.6, 0.8];
    capability_requirements.door_assembly_FR = [0.7, 0.8, 0.8, 0.6, 0.8];
    capability_requirements.door_assembly_RL = [0.7, 0.8, 0.8, 0.6, 0.8];
    capability_requirements.door_assembly_RR = [0.7, 0.8, 0.8, 0.6, 0.8];
    capability_requirements.roof_assembly = [0.8, 0.7, 0.7, 0.5, 0.9];
    
    % Define dependency structure based on automotive assembly process
    dependency_structure = struct();
    dependency_structure.chassis_positioning = [];  % No prerequisites
    dependency_structure.engine_positioning = {'chassis_positioning'};
    dependency_structure.gearbox_positioning = {'engine_positioning'};
    dependency_structure.exhaust_system = {'chassis_positioning'};
    dependency_structure.cooling_system = {'engine_positioning'};
    dependency_structure.wheel_assembly_FL = {'chassis_positioning'};
    dependency_structure.wheel_assembly_FR = {'chassis_positioning'};
    dependency_structure.wheel_assembly_RL = {'chassis_positioning'};
    dependency_structure.wheel_assembly_RR = {'chassis_positioning'};
    dependency_structure.dashboard_positioning = {'chassis_positioning'};
    dependency_structure.seat_front_L = {'dashboard_positioning'};
    dependency_structure.seat_front_R = {'dashboard_positioning'};
    dependency_structure.seat_rear = {'chassis_positioning'};
    dependency_structure.seat_rear_L = {'chassis_positioning'};
    dependency_structure.seat_rear_R = {'chassis_positioning'};
    dependency_structure.steering_wheel = {'dashboard_positioning'};
    dependency_structure.door_assembly_L = {'chassis_positioning'};
    dependency_structure.door_assembly_R = {'chassis_positioning'};
    dependency_structure.door_assembly_FL = {'chassis_positioning'};
    dependency_structure.door_assembly_FR = {'chassis_positioning'};
    dependency_structure.door_assembly_RL = {'chassis_positioning'};
    dependency_structure.door_assembly_RR = {'chassis_positioning'};
    dependency_structure.roof_assembly = {'chassis_positioning'};
    
    % Define task positions based on automotive assembly line layout
    % Create a grid layout for the assembly line
    grid_width = ceil(sqrt(num_tasks * 1.5));
    grid_height = ceil(num_tasks / grid_width);
    
    grid_x = linspace(0.5, env.width - 0.5, grid_width);
    grid_y = linspace(0.5, env.height - 0.5, grid_height);
    
    [X, Y] = meshgrid(grid_x, grid_y);
    positions = [X(:), Y(:)];
    
    % Shuffle positions to simulate realistic layout
    rng(42);  % For reproducibility
    shuffled_indices = randperm(size(positions, 1));
    positions = positions(shuffled_indices, :);
    
    % Create task structures
    tasks = struct([]);
    
    for i = 1:num_tasks
        tasks(i).id = i;
        tasks(i).name = components{i};
        tasks(i).position = positions(i, :);
        tasks(i).execution_time = execution_times.(components{i});
        tasks(i).capabilities_required = capability_requirements.(components{i});
        
        % Convert dependencies from names to IDs
        prereq_names = dependency_structure.(components{i});
        prereq_ids = [];
        
        for j = 1:length(prereq_names)
            prereq_idx = find(strcmp(components, prereq_names{j}));
            if ~isempty(prereq_idx)
                prereq_ids = [prereq_ids, prereq_idx];
            end
        end
        
        tasks(i).prerequisites = prereq_ids;
        
        % Add precision requirements in mm (based on automotive industry standards)
        if contains(components{i}, 'engine') || contains(components{i}, 'gearbox')
            tasks(i).precision_requirement = 0.05;  % 0.05 mm precision
        elseif contains(components{i}, 'wheel')
            tasks(i).precision_requirement = 0.1;   % 0.1 mm precision
        elseif contains(components{i}, 'dashboard') || contains(components{i}, 'steering')
            tasks(i).precision_requirement = 0.2;   % 0.2 mm precision
        else
            tasks(i).precision_requirement = 0.5;   % 0.5 mm precision
        end
        
        % Add force requirements in Newtons (realistic assembly forces)
        if contains(components{i}, 'wheel')
            tasks(i).force_requirement = 80;  % 80N for wheel tightening
        elseif contains(components{i}, 'engine') || contains(components{i}, 'chassis')
            tasks(i).force_requirement = 50;  % 50N for positioning heavy components
        elseif contains(components{i}, 'door')
            tasks(i).force_requirement = 40;  % 40N for door assembly
        else
            tasks(i).force_requirement = 20;  % 20N for general assembly
        end
    end
    
    fprintf('Created %d automotive assembly tasks with realistic characteristics.\n', num_tasks);
    
    % Print out some task information for verification
    fprintf('Sample task details:\n');
    for i = 1:min(3, num_tasks)
        fprintf('  Task %d: %s, Time: %.1f min, Prerequisites: ', ...
                i, tasks(i).name, tasks(i).execution_time);
        
        if isempty(tasks(i).prerequisites)
            fprintf('None');
        else
            fprintf('%d ', tasks(i).prerequisites);
        end
        
        fprintf('\n');
    end
end

function tasks = local_createElectronicsAssemblyTasks(env, difficulty_level)
    % CREATEELECTRONICSASSEMBLYTASKS - Create tasks based on electronics assembly processes
    %
    % Parameters:
    %   env - Environment structure with width and height properties
    %   difficulty_level - Level of scenario complexity (1-4, default: 2)
    %
    % Returns:
    %   tasks - Array of task structures with realistic electronics assembly characteristics
    
    % Set default difficulty level
    if nargin < 2
        difficulty_level = 2;
    end
    
    fprintf('Creating electronics assembly scenario (difficulty level %d)...\n', difficulty_level);
    
    % Define electronics assembly components based on difficulty level
    switch difficulty_level
        case 1 % Simple, 8 tasks
            components = {'pcb_positioning', 'microcontroller_placement', 'resistor_array', ...
                         'capacitor_array', 'led_placement', 'usb_connector', ...
                         'power_regulator', 'header_pins'};
            num_tasks = 8;
        case 2 % Medium, 12 tasks
            components = {'pcb_positioning', 'microcontroller_placement', 'memory_chip', ...
                         'resistor_array_1', 'resistor_array_2', 'capacitor_array_1', ...
                         'capacitor_array_2', 'led_array', 'usb_connector', ...
                         'power_regulator', 'header_pins', 'wifi_module'};
            num_tasks = 12;
        case 3 % Complex, 16 tasks
            components = {'pcb_positioning', 'microcontroller_placement', 'memory_chip', ...
                         'resistor_array_1', 'resistor_array_2', 'capacitor_array_1', ...
                         'capacitor_array_2', 'led_array', 'usb_connector', ...
                         'power_regulator', 'header_pins', 'wifi_module', ...
                         'bluetooth_module', 'accelerometer', 'temperature_sensor', 'display_connector'};
            num_tasks = 16;
        case 4 % Very Complex, 20 tasks
            components = {'pcb_positioning', 'microcontroller_placement', 'memory_chip', 'fpga_chip', ...
                         'resistor_array_1', 'resistor_array_2', 'capacitor_array_1', 'capacitor_array_2', ...
                         'led_array', 'usb_connector', 'ethernet_connector', 'power_regulator', ...
                         'header_pins', 'wifi_module', 'bluetooth_module', 'accelerometer', ...
                         'temperature_sensor', 'humidity_sensor', 'display_connector', 'battery_connector'};
            num_tasks = 20;
        otherwise
            error('Invalid difficulty level. Choose 1-4.');
    end
    
    % Define task execution times (industry-realistic values in minutes)
    execution_times = struct();
    execution_times.pcb_positioning = 1.2;
    execution_times.microcontroller_placement = 2.5;
    execution_times.memory_chip = 2.3;
    execution_times.fpga_chip = 3.0;
    execution_times.resistor_array = 1.8;
    execution_times.resistor_array_1 = 1.8;
    execution_times.resistor_array_2 = 1.8;
    execution_times.capacitor_array = 1.5;
    execution_times.capacitor_array_1 = 1.5;
    execution_times.capacitor_array_2 = 1.5;
    execution_times.led_placement = 1.0;
    execution_times.led_array = 1.5;
    execution_times.usb_connector = 1.2;
    execution_times.ethernet_connector = 1.4;
    execution_times.power_regulator = 1.8;
    execution_times.header_pins = 1.0;
    execution_times.wifi_module = 2.0;
    execution_times.bluetooth_module = 1.8;
    execution_times.accelerometer = 1.6;
    execution_times.temperature_sensor = 1.2;
    execution_times.humidity_sensor = 1.2;
    execution_times.display_connector = 1.5;
    execution_times.battery_connector = 1.3;
    
    % Define capability requirements (5-dimensional vector for each task)
    % Dimensions: [precision, strength, dexterity, speed, coordination]
    capability_requirements = struct();
    capability_requirements.pcb_positioning = [0.7, 0.3, 0.6, 0.5, 0.7];
    capability_requirements.microcontroller_placement = [0.9, 0.2, 0.9, 0.4, 0.9];
    capability_requirements.memory_chip = [0.9, 0.2, 0.9, 0.4, 0.9];
    capability_requirements.fpga_chip = [0.9, 0.2, 0.9, 0.4, 0.9];
    capability_requirements.resistor_array = [0.7, 0.1, 0.8, 0.7, 0.8];
    capability_requirements.resistor_array_1 = [0.7, 0.1, 0.8, 0.7, 0.8];
    capability_requirements.resistor_array_2 = [0.7, 0.1, 0.8, 0.7, 0.8];
    capability_requirements.capacitor_array = [0.7, 0.1, 0.8, 0.7, 0.8];
    capability_requirements.capacitor_array_1 = [0.7, 0.1, 0.8, 0.7, 0.8];
    capability_requirements.capacitor_array_2 = [0.7, 0.1, 0.8, 0.7, 0.8];
    capability_requirements.led_placement = [0.6, 0.1, 0.8, 0.6, 0.7];
    capability_requirements.led_array = [0.6, 0.1, 0.8, 0.7, 0.7];
    capability_requirements.usb_connector = [0.8, 0.4, 0.7, 0.5, 0.7];
    capability_requirements.ethernet_connector = [0.8, 0.4, 0.7, 0.5, 0.7];
    capability_requirements.power_regulator = [0.8, 0.3, 0.7, 0.5, 0.7];
    capability_requirements.header_pins = [0.7, 0.2, 0.8, 0.6, 0.7];
    capability_requirements.wifi_module = [0.8, 0.2, 0.8, 0.5, 0.8];
    capability_requirements.bluetooth_module = [0.8, 0.2, 0.8, 0.5, 0.8];
    capability_requirements.accelerometer = [0.8, 0.2, 0.8, 0.5, 0.8];
    capability_requirements.temperature_sensor = [0.7, 0.2, 0.8, 0.6, 0.7];
    capability_requirements.humidity_sensor = [0.7, 0.2, 0.8, 0.6, 0.7];
    capability_requirements.display_connector = [0.8, 0.3, 0.7, 0.5, 0.8];
    capability_requirements.battery_connector = [0.7, 0.4, 0.7, 0.5, 0.7];
    
    % Define dependency structure based on electronics assembly process
    dependency_structure = struct();
    dependency_structure.pcb_positioning = [];  % No prerequisites
    dependency_structure.microcontroller_placement = {'pcb_positioning'};
    dependency_structure.memory_chip = {'pcb_positioning'};
    dependency_structure.fpga_chip = {'pcb_positioning'};
    dependency_structure.resistor_array = {'pcb_positioning'};
    dependency_structure.resistor_array_1 = {'pcb_positioning'};
    dependency_structure.resistor_array_2 = {'pcb_positioning'};
    dependency_structure.capacitor_array = {'pcb_positioning'};
    dependency_structure.capacitor_array_1 = {'pcb_positioning'};
    dependency_structure.capacitor_array_2 = {'pcb_positioning'};
    dependency_structure.led_placement = {'pcb_positioning'};
    dependency_structure.led_array = {'pcb_positioning'};
    dependency_structure.usb_connector = {'pcb_positioning'};
    dependency_structure.ethernet_connector = {'pcb_positioning'};
    dependency_structure.power_regulator = {'pcb_positioning'};
    dependency_structure.header_pins = {'pcb_positioning'};
    dependency_structure.wifi_module = {'pcb_positioning', 'microcontroller_placement'};
    dependency_structure.bluetooth_module = {'pcb_positioning', 'microcontroller_placement'};
    dependency_structure.accelerometer = {'pcb_positioning', 'microcontroller_placement'};
    dependency_structure.temperature_sensor = {'pcb_positioning'};
    dependency_structure.humidity_sensor = {'pcb_positioning'};
    dependency_structure.display_connector = {'pcb_positioning'};
    dependency_structure.battery_connector = {'pcb_positioning', 'power_regulator'};
    
    % Define task positions based on electronics assembly layout
    % Create a grid layout for the PCB assembly
    grid_width = ceil(sqrt(num_tasks * 1.5));
    grid_height = ceil(num_tasks / grid_width);
    
    grid_x = linspace(0.5, env.width - 0.5, grid_width);
    grid_y = linspace(0.5, env.height - 0.5, grid_height);
    
    [X, Y] = meshgrid(grid_x, grid_y);
    positions = [X(:), Y(:)];
    
    % Shuffle positions to simulate realistic layout
    rng(42);  % For reproducibility
    shuffled_indices = randperm(size(positions, 1));
    positions = positions(shuffled_indices, :);
    
    % Create task structures
    tasks = struct([]);
    
    for i = 1:num_tasks
        tasks(i).id = i;
        tasks(i).name = components{i};
        tasks(i).position = positions(i, :);
        tasks(i).execution_time = execution_times.(components{i});
        tasks(i).capabilities_required = capability_requirements.(components{i});
        
        % Convert dependencies from names to IDs
        prereq_names = dependency_structure.(components{i});
        prereq_ids = [];
        
        for j = 1:length(prereq_names)
            prereq_idx = find(strcmp(components, prereq_names{j}));
            if ~isempty(prereq_idx)
                prereq_ids = [prereq_ids, prereq_idx];
            end
        end
        
        tasks(i).prerequisites = prereq_ids;
        
        % Add precision requirements in mm (based on electronics industry standards)
        if contains(components{i}, 'microcontroller') || contains(components{i}, 'chip')
            tasks(i).precision_requirement = 0.02;  % 0.02 mm precision (20 microns)
        elseif contains(components{i}, 'resistor') || contains(components{i}, 'capacitor')
            tasks(i).precision_requirement = 0.05;  % 0.05 mm precision
        elseif contains(components{i}, 'connector')
            tasks(i).precision_requirement = 0.1;   % 0.1 mm precision
        else
            tasks(i).precision_requirement = 0.2;   % 0.2 mm precision
        end
        
        % Add force requirements in Newtons (realistic electronics assembly forces)
        if contains(components{i}, 'connector')
            tasks(i).force_requirement = 15;  % 15N for connector insertion
        elseif contains(components{i}, 'chip')
            tasks(i).force_requirement = 5;   % 5N for chip placement
        else
            tasks(i).force_requirement = 2;   % 2N for general electronics assembly
        end
    end
    
    fprintf('Created %d electronics assembly tasks with realistic characteristics.\n', num_tasks);
    
    % Print out some task information for verification
    fprintf('Sample task details:\n');
    for i = 1:min(3, num_tasks)
        fprintf('  Task %d: %s, Time: %.1f min, Prerequisites: ', ...
                i, tasks(i).name, tasks(i).execution_time);
        
        if isempty(tasks(i).prerequisites)
            fprintf('None');
        else
            fprintf('%d ', tasks(i).prerequisites);
        end
        
        fprintf('\n');
    end
end

function tasks = local_createFurnitureAssemblyTasks(env, difficulty_level)
    % CREATEFURNITUREASSEMBLYTASKS - Create tasks based on furniture assembly processes
    %
    % Parameters:
    %   env - Environment structure with width and height properties
    %   difficulty_level - Level of scenario complexity (1-4, default: 2)
    %
    % Returns:
    %   tasks - Array of task structures with realistic furniture assembly characteristics
    
    % Set default difficulty level
    if nargin < 2
        difficulty_level = 2;
    end
    
    fprintf('Creating furniture assembly scenario (difficulty level %d)...\n', difficulty_level);
    
    % Define furniture assembly components based on difficulty level
    switch difficulty_level
        case 1 % Simple chair, 8 tasks
            components = {'base_frame', 'left_leg', 'right_leg', 'back_leg_left', ...
                         'back_leg_right', 'seat_panel', 'back_panel', 'final_tightening'};
            num_tasks = 8;
        case 2 % Desk, 12 tasks
            components = {'desk_frame', 'left_leg', 'right_leg', 'back_leg_left', ...
                         'back_leg_right', 'support_beam_front', 'support_beam_back', ...
                         'support_beam_left', 'support_beam_right', 'desk_surface', ...
                         'drawer_assembly', 'final_tightening'};
            num_tasks = 12;
        case 3 % Bookshelf, 16 tasks
            components = {'base_panel', 'top_panel', 'left_panel', 'right_panel', ...
                         'back_panel', 'shelf_1', 'shelf_2', 'shelf_3', 'shelf_4', ...
                         'support_rod_left', 'support_rod_right', 'door_assembly', ...
                         'hinge_installation', 'handle_installation', 'leveling_feet', ...
                         'final_tightening'};
            num_tasks = 16;
        case 4 % Complex wardrobe, 20 tasks
            components = {'base_panel', 'top_panel', 'left_panel', 'right_panel', ...
                         'back_panel_lower', 'back_panel_upper', 'center_divider', ...
                         'shelf_1', 'shelf_2', 'shelf_3', 'shoe_rack', ...
                         'hanging_rod_left', 'hanging_rod_right', 'drawer_assembly_1', ...
                         'drawer_assembly_2', 'door_assembly_left', 'door_assembly_right', ...
                         'hinge_installation', 'handle_installation', 'final_tightening'};
            num_tasks = 20;
        otherwise
            error('Invalid difficulty level. Choose 1-4.');
    end
    
    % Define task execution times (industry-realistic values in minutes)
    execution_times = struct();
    % Chair components
    execution_times.base_frame = 3.5;
    execution_times.left_leg = 1.5;
    execution_times.right_leg = 1.5;
    execution_times.back_leg_left = 1.5;
    execution_times.back_leg_right = 1.5;
    execution_times.seat_panel = 2.0;
    execution_times.back_panel = 2.5;
    
    % Desk components
    execution_times.desk_frame = 4.0;
    execution_times.support_beam_front = 2.0;
    execution_times.support_beam_back = 2.0;
    execution_times.support_beam_left = 2.0;
    execution_times.support_beam_right = 2.0;
    execution_times.desk_surface = 3.0;
    execution_times.drawer_assembly = 5.0;
    
    % Bookshelf components
    execution_times.base_panel = 2.5;
    execution_times.top_panel = 2.5;
    execution_times.left_panel = 2.5;
    execution_times.right_panel = 2.5;
    execution_times.back_panel = 3.5;
    execution_times.back_panel_lower = 3.0;
    execution_times.back_panel_upper = 3.0;
    execution_times.center_divider = 3.0;
    execution_times.shelf_1 = 2.0;
    execution_times.shelf_2 = 2.0;
    execution_times.shelf_3 = 2.0;
    execution_times.shelf_4 = 2.0;
    execution_times.support_rod_left = 1.5;
    execution_times.support_rod_right = 1.5;
    execution_times.door_assembly = 4.0;
    execution_times.door_assembly_left = 4.0;
    execution_times.door_assembly_right = 4.0;
    execution_times.hinge_installation = 3.0;
    execution_times.handle_installation = 2.0;
    execution_times.leveling_feet = 2.0;
    execution_times.hanging_rod_left = 1.5;
    execution_times.hanging_rod_right = 1.5;
    execution_times.drawer_assembly_1 = 5.0;
    execution_times.drawer_assembly_2 = 5.0;
    execution_times.shoe_rack = 3.0;
    
    % Common
    execution_times.final_tightening = 2.0;
    
    % Define capability requirements (5-dimensional vector for each task)
    % Dimensions: [precision, strength, dexterity, speed, coordination]
    capability_requirements = struct();
    % Base structures
    capability_requirements.base_frame = [0.7, 0.6, 0.5, 0.4, 0.7];
    capability_requirements.desk_frame = [0.7, 0.7, 0.5, 0.4, 0.7];
    capability_requirements.base_panel = [0.6, 0.7, 0.5, 0.4, 0.7];
    capability_requirements.top_panel = [0.6, 0.7, 0.5, 0.4, 0.7];
    capability_requirements.left_panel = [0.6, 0.7, 0.5, 0.4, 0.7];
    capability_requirements.right_panel = [0.6, 0.7, 0.5, 0.4, 0.7];
    
    % Legs and supports
    capability_requirements.left_leg = [0.5, 0.6, 0.6, 0.5, 0.6];
    capability_requirements.right_leg = [0.5, 0.6, 0.6, 0.5, 0.6];
    capability_requirements.back_leg_left = [0.5, 0.6, 0.6, 0.5, 0.6];
    capability_requirements.back_leg_right = [0.5, 0.6, 0.6, 0.5, 0.6];
    capability_requirements.support_beam_front = [0.6, 0.7, 0.5, 0.4, 0.7];
    capability_requirements.support_beam_back = [0.6, 0.7, 0.5, 0.4, 0.7];
    capability_requirements.support_beam_left = [0.6, 0.7, 0.5, 0.4, 0.7];
    capability_requirements.support_beam_right = [0.6, 0.7, 0.5, 0.4, 0.7];
    
    % Panels and surfaces
    capability_requirements.seat_panel = [0.7, 0.5, 0.6, 0.4, 0.7];
    capability_requirements.back_panel = [0.7, 0.5, 0.6, 0.4, 0.7];
    capability_requirements.desk_surface = [0.7, 0.8, 0.6, 0.4, 0.7];
    capability_requirements.back_panel = [0.6, 0.7, 0.6, 0.4, 0.7];
    capability_requirements.back_panel_lower = [0.6, 0.7, 0.6, 0.4, 0.7];
    capability_requirements.back_panel_upper = [0.6, 0.7, 0.6, 0.4, 0.7];
    capability_requirements.center_divider = [0.7, 0.6, 0.6, 0.4, 0.7];
    
    % Shelves and rods
    capability_requirements.shelf_1 = [0.6, 0.5, 0.7, 0.5, 0.6];
    capability_requirements.shelf_2 = [0.6, 0.5, 0.7, 0.5, 0.6];
    capability_requirements.shelf_3 = [0.6, 0.5, 0.7, 0.5, 0.6];
    capability_requirements.shelf_4 = [0.6, 0.5, 0.7, 0.5, 0.6];
    capability_requirements.support_rod_left = [0.5, 0.4, 0.8, 0.6, 0.7];
    capability_requirements.support_rod_right = [0.5, 0.4, 0.8, 0.6, 0.7];
    capability_requirements.hanging_rod_left = [0.5, 0.4, 0.8, 0.6, 0.7];
    capability_requirements.hanging_rod_right = [0.5, 0.4, 0.8, 0.6, 0.7];
    capability_requirements.shoe_rack = [0.6, 0.5, 0.7, 0.5, 0.7];
    
    % Complexities
    capability_requirements.drawer_assembly = [0.8, 0.5, 0.8, 0.6, 0.8];
    capability_requirements.drawer_assembly_1 = [0.8, 0.5, 0.8, 0.6, 0.8];
    capability_requirements.drawer_assembly_2 = [0.8, 0.5, 0.8, 0.6, 0.8];
    capability_requirements.door_assembly = [0.7, 0.6, 0.7, 0.5, 0.8];
    capability_requirements.door_assembly_left = [0.7, 0.6, 0.7, 0.5, 0.8];
    capability_requirements.door_assembly_right = [0.7, 0.6, 0.7, 0.5, 0.8];
    capability_requirements.hinge_installation = [0.8, 0.4, 0.9, 0.6, 0.8];
    capability_requirements.handle_installation = [0.8, 0.3, 0.9, 0.6, 0.8];
    capability_requirements.leveling_feet = [0.7, 0.5, 0.8, 0.5, 0.7];
    
    % Final steps
    capability_requirements.final_tightening = [0.8, 0.7, 0.7, 0.5, 0.8];
    
    % Define dependency structure based on furniture assembly process
    dependency_structure = struct();
    
    % Chair dependencies
    dependency_structure.base_frame = [];  % No prerequisites
    dependency_structure.left_leg = {'base_frame'};
    dependency_structure.right_leg = {'base_frame'};
    dependency_structure.back_leg_left = {'base_frame'};
    dependency_structure.back_leg_right = {'base_frame'};
    dependency_structure.seat_panel = {'left_leg', 'right_leg', 'back_leg_left', 'back_leg_right'};
    dependency_structure.back_panel = {'back_leg_left', 'back_leg_right'};
    
    % Desk dependencies
    dependency_structure.desk_frame = [];  % No prerequisites
    dependency_structure.support_beam_front = {'left_leg', 'right_leg'};
    dependency_structure.support_beam_back = {'back_leg_left', 'back_leg_right'};
    dependency_structure.support_beam_left = {'left_leg', 'back_leg_left'};
    dependency_structure.support_beam_right = {'right_leg', 'back_leg_right'};
    dependency_structure.desk_surface = {'support_beam_front', 'support_beam_back', 'support_beam_left', 'support_beam_right'};
    dependency_structure.drawer_assembly = {'desk_surface'};
    
    % Bookshelf dependencies
    dependency_structure.base_panel = [];  % No prerequisites
    dependency_structure.left_panel = {'base_panel'};
    dependency_structure.right_panel = {'base_panel'};
    dependency_structure.top_panel = {'left_panel', 'right_panel'};
    dependency_structure.back_panel = {'left_panel', 'right_panel', 'top_panel'};
    dependency_structure.back_panel_lower = {'left_panel', 'right_panel'};
    dependency_structure.back_panel_upper = {'left_panel', 'right_panel', 'top_panel'};
    dependency_structure.center_divider = {'base_panel', 'top_panel'};
    dependency_structure.shelf_1 = {'left_panel', 'right_panel'};
    dependency_structure.shelf_2 = {'left_panel', 'right_panel'};
    dependency_structure.shelf_3 = {'left_panel', 'right_panel'};
    dependency_structure.shelf_4 = {'left_panel', 'right_panel'};
    dependency_structure.support_rod_left = {'left_panel'};
    dependency_structure.support_rod_right = {'right_panel'};
    dependency_structure.hanging_rod_left = {'left_panel', 'center_divider'};
    dependency_structure.hanging_rod_right = {'right_panel', 'center_divider'};
    dependency_structure.door_assembly = {'left_panel', 'right_panel', 'top_panel'};
    dependency_structure.door_assembly_left = {'left_panel', 'top_panel', 'base_panel'};
    dependency_structure.door_assembly_right = {'right_panel', 'top_panel', 'base_panel'};
    dependency_structure.hinge_installation = {'door_assembly'};
    dependency_structure.handle_installation = {'door_assembly'};
    dependency_structure.drawer_assembly_1 = {'left_panel', 'right_panel', 'shelf_1'};
    dependency_structure.drawer_assembly_2 = {'left_panel', 'right_panel', 'shelf_2'};
    dependency_structure.shoe_rack = {'base_panel', 'left_panel', 'right_panel'};
    dependency_structure.leveling_feet = {'base_panel'};
    
    % Common
    if strcmp(difficulty_level, '1')  % Chair
        dependency_structure.final_tightening = {'seat_panel', 'back_panel'};
    elseif strcmp(difficulty_level, '2')  % Desk
        dependency_structure.final_tightening = {'desk_surface', 'drawer_assembly'};
    elseif strcmp(difficulty_level, '3')  % Bookshelf
        dependency_structure.final_tightening = {'back_panel', 'shelf_1', 'shelf_2', 'shelf_3', 'shelf_4', 'door_assembly', 'hinge_installation', 'handle_installation'};
    else  % Wardrobe
        dependency_structure.final_tightening = {'back_panel_lower', 'back_panel_upper', 'door_assembly_left', 'door_assembly_right', 'drawer_assembly_1', 'drawer_assembly_2'};
    end
    
    % Define task positions based on furniture assembly layout
    % Create a grid layout for the assembly area
    grid_width = ceil(sqrt(num_tasks * 1.5));
    grid_height = ceil(num_tasks / grid_width);
    
    grid_x = linspace(0.5, env.width - 0.5, grid_width);
    grid_y = linspace(0.5, env.height - 0.5, grid_height);
    
    [X, Y] = meshgrid(grid_x, grid_y);
    positions = [X(:), Y(:)];
    
    % Shuffle positions to simulate realistic layout
    rng(42);  % For reproducibility
    shuffled_indices = randperm(size(positions, 1));
    positions = positions(shuffled_indices, :);
    
    % Create task structures
    tasks = struct([]);
    
    for i = 1:num_tasks
        tasks(i).id = i;
        tasks(i).name = components{i};
        tasks(i).position = positions(i, :);
        tasks(i).execution_time = execution_times.(components{i});
        tasks(i).capabilities_required = capability_requirements.(components{i});
        
        % Convert dependencies from names to IDs
        prereq_names = dependency_structure.(components{i});
        prereq_ids = [];
        
        for j = 1:length(prereq_names)
            prereq_idx = find(strcmp(components, prereq_names{j}));
            if ~isempty(prereq_idx)
                prereq_ids = [prereq_ids, prereq_idx];
            end
        end
        
        tasks(i).prerequisites = prereq_ids;
        
        % Add precision requirements in mm (based on furniture industry standards)
        if contains(components{i}, 'hinge') || contains(components{i}, 'handle')
            tasks(i).precision_requirement = 0.2;  % 0.2 mm precision
        elseif contains(components{i}, 'drawer') || contains(components{i}, 'door')
            tasks(i).precision_requirement = 0.5;  % 0.5 mm precision
        elseif contains(components{i}, 'panel') || contains(components{i}, 'surface')
            tasks(i).precision_requirement = 1.0;  % 1.0 mm precision
        else
            tasks(i).precision_requirement = 2.0;  % 2.0 mm precision
        end
        
        % Add force requirements in Newtons (realistic furniture assembly forces)
        if contains(components{i}, 'final')
            tasks(i).force_requirement = 35;  % 35N for final tightening
        elseif contains(components{i}, 'panel') || contains(components{i}, 'surface')
            tasks(i).force_requirement = 30;  % 30N for panel placement
        elseif contains(components{i}, 'leg') || contains(components{i}, 'support')
            tasks(i).force_requirement = 25;  % 25N for leg/support assembly
        else
            tasks(i).force_requirement = 15;  % 15N for general assembly
        end
    end
    
    fprintf('Created %d furniture assembly tasks with realistic characteristics.\n', num_tasks);
    
    % Print out some task information for verification
    fprintf('Sample task details:\n');
    for i = 1:min(3, num_tasks)
        fprintf('  Task %d: %s, Time: %.1f min, Prerequisites: ', ...
                i, tasks(i).name, tasks(i).execution_time);
        
        if isempty(tasks(i).prerequisites)
            fprintf('None');
        else
            fprintf('%d ', tasks(i).prerequisites);
        end
        
        fprintf('\n');
    end
end

function tasks = local_applyIndustrialTolerances(tasks, industry_type)
    % APPLYINDUSTRIALTOLETRANCES - Apply realistic industrial tolerances to task requirements
    %
    % Parameters:
    %   tasks - Array of task structures
    %   industry_type - Type of industry ('automotive', 'electronics', 'furniture')
    %
    % Returns:
    %   tasks - Updated task structures with realistic tolerances
    
    fprintf('Applying industrial tolerances for %s industry...\n', industry_type);
    
    % Define industry-specific tolerances
    tolerances = struct();
    
    switch lower(industry_type)
        case 'automotive'
            % Automotive industry tolerances
            tolerances.precision_baseline = 0.1;  % 0.1 mm baseline precision
            tolerances.precision_variation = 0.05;  % Variation in precision
            tolerances.force_baseline = 5.0;  % 5N baseline force variation
            tolerances.force_variation = 2.0;  % Variation in force
            tolerances.time_variation = 0.2;  % 20% variation in time
            
        case 'electronics'
            % Electronics industry tolerances
            tolerances.precision_baseline = 0.05;  % 0.05 mm baseline precision
            tolerances.precision_variation = 0.02;  % Variation in precision
            tolerances.force_baseline = 1.0;  % 1N baseline force variation
            tolerances.force_variation = 0.5;  % Variation in force
            tolerances.time_variation = 0.15;  % 15% variation in time
            
        case 'furniture'
            % Furniture industry tolerances
            tolerances.precision_baseline = 0.5;  % 0.5 mm baseline precision
            tolerances.precision_variation = 0.2;  % Variation in precision
            tolerances.force_baseline = 3.0;  % 3N baseline force variation
            tolerances.force_variation = 1.5;  % Variation in force
            tolerances.time_variation = 0.25;  % 25% variation in time
            
        otherwise
            error('Invalid industry type. Choose "automotive", "electronics", or "furniture".');
    end
    
    % Apply tolerances to tasks
    rng(42);  % For reproducibility
    
    for i = 1:length(tasks)
        % Apply precision tolerance if exists
        if isfield(tasks(i), 'precision_requirement')
            nominal_precision = tasks(i).precision_requirement;
            precision_tolerance = tolerances.precision_baseline + tolerances.precision_variation * rand();
            tasks(i).precision_tolerance = precision_tolerance;
            
            % Add min/max precision ranges
            tasks(i).precision_min = max(0, nominal_precision - precision_tolerance);
            tasks(i).precision_max = nominal_precision + precision_tolerance;
        end
        
        % Apply force tolerance if exists
        if isfield(tasks(i), 'force_requirement')
            nominal_force = tasks(i).force_requirement;
            force_tolerance = tolerances.force_baseline + tolerances.force_variation * rand();
            tasks(i).force_tolerance = force_tolerance;
            
            % Add min/max force ranges
            tasks(i).force_min = max(0, nominal_force - force_tolerance);
            tasks(i).force_max = nominal_force + force_tolerance;
        end
        
        % Apply time variations
        nominal_time = tasks(i).execution_time;
        time_variation = tolerances.time_variation * (2 * rand() - 1);  % +/- variation
        
        % Adjust execution time
        tasks(i).execution_time_min = nominal_time * (1 - tolerances.time_variation);
        tasks(i).execution_time_max = nominal_time * (1 + tolerances.time_variation);
        
        % Adjust execution time itself for this run
        tasks(i).execution_time = nominal_time * (1 + time_variation);
    end
    
    fprintf('Applied industrial tolerances to %d tasks.\n', length(tasks));
end

function [graph, critical_path] = local_generateTaskDependencyGraph(tasks)
    % GENERATETASKDEPENDENCYGRAPH - Generate and analyze the task dependency graph
    %
    % Parameters:
    %   tasks - Array of task structures
    %
    % Returns:
    %   graph - Structure containing graph information
    %   critical_path - Array of task IDs in the critical path
    
    fprintf('Generating task dependency graph...\n');
    
    num_tasks = numel(tasks);
    
    % Create adjacency matrix for dependencies
    adjacency_matrix = zeros(num_tasks, num_tasks);
    
    for i = 1:num_tasks
        if isfield(tasks(i), 'prerequisites') && ~isempty(tasks(i).prerequisites)
            for j = 1:length(tasks(i).prerequisites)
                prereq = tasks(i).prerequisites(j);
                if prereq <= num_tasks
                    adjacency_matrix(prereq, i) = 1;  % Edge from prerequisite to task
                end
            end
        end
    end
    
    % Calculate in-degree and out-degree
    in_degree = sum(adjacency_matrix, 1)';
    out_degree = sum(adjacency_matrix, 2);
    
    % Identify start and end nodes
    start_nodes = find(in_degree == 0);
    end_nodes = find(out_degree == 0);
    
    % Perform topological sort
    topo_order = [];
    queue = start_nodes;
    visited = false(num_tasks, 1);
    
    while ~isempty(queue)
        node = queue(1);
        queue(1) = [];
        
        if ~visited(node)
            visited(node) = true;
            topo_order = [topo_order, node];
            
            % Find all successors
            successors = find(adjacency_matrix(node, :) > 0);
            
            for i = 1:length(successors)
                successor = successors(i);
                
                % Reduce in-degree
                in_degree(successor) = in_degree(successor) - 1;
                
                % If all prerequisites are visited, add to queue
                if in_degree(successor) == 0
                    queue = [queue, successor];
                end
            end
        end
    end
    
    % Check for cycles
    if length(topo_order) < num_tasks
        fprintf('Warning: Dependency graph contains cycles!\n');
    end
    
    % Calculate earliest start and finish times
    earliest_start = zeros(num_tasks, 1);
    earliest_finish = zeros(num_tasks, 1);
    
    for i = 1:length(topo_order)
        node = topo_order(i);
        
        % Find predecessors
        predecessors = find(adjacency_matrix(:, node) > 0);
        
        % Calculate earliest start time
        if ~isempty(predecessors)
            earliest_start(node) = max(earliest_finish(predecessors));
        else
            earliest_start(node) = 0;
        end
        
        % Calculate earliest finish time
        earliest_finish(node) = earliest_start(node) + tasks(node).execution_time;
    end
    
    % Calculate latest start and finish times
    latest_start = inf(num_tasks, 1);
    latest_finish = inf(num_tasks, 1);
    
    % Set latest finish time for end nodes
    max_finish = max(earliest_finish);
    for i = 1:length(end_nodes)
        latest_finish(end_nodes(i)) = max_finish;
    end
    
    % Backward pass
    for i = length(topo_order):-1:1
        node = topo_order(i);
        
        % Find successors
        successors = find(adjacency_matrix(node, :) > 0);
        
        % Calculate latest finish time
        if ~isempty(successors)
            latest_finish(node) = min(latest_start(successors));
        end
        
        % Calculate latest start time
        latest_start(node) = latest_finish(node) - tasks(node).execution_time;
    end
    
    % Calculate slack
    slack = latest_start - earliest_start;
    
    % Identify critical path (zero slack)
    critical_nodes = find(slack < 1e-6);
    
    % Sort critical nodes by topological order
    [~, sorted_idx] = ismember(critical_nodes, topo_order);
    [~, critical_idx] = sort(sorted_idx);
    critical_path = critical_nodes(critical_idx);
    
    % Prepare graph structure
    graph = struct();
    graph.adjacency_matrix = adjacency_matrix;
    graph.topo_order = topo_order;
    graph.start_nodes = start_nodes;
    graph.end_nodes = end_nodes;
    graph.earliest_start = earliest_start;
    graph.earliest_finish = earliest_finish;
    graph.latest_start = latest_start;
    graph.latest_finish = latest_finish;
    graph.slack = slack;
    graph.makespan = max_finish;
    graph.critical_path = critical_path;  % Add critical path to the graph structure
    
    fprintf('Task dependency graph generated successfully.\n');
    fprintf('Makespan: %.2f minutes\n', graph.makespan);
    fprintf('Critical path: ');
    fprintf('%d ', critical_path);
    fprintf('\n');
end

function local_visualizeIndustrialScenario(env, robots, tasks, graph)
    % VISUALIZEINDUSTRIALSCENARIO - Visualize the industrial scenario with tasks and dependencies
    %
    % Parameters:
    %   env - Environment structure
    %   robots - Array of robot structures
    %   tasks - Array of task structures
    %   graph - Graph structure from generateTaskDependencyGraph
    
    fprintf('Visualizing industrial scenario...\n');
    
    % Create figure
    figure('Name', 'Industrial Assembly Scenario', 'Position', [50, 50, 1200, 800]);
    
    % Create layout with 2 subplots
    subplot(1, 2, 1);
    
    % Draw environment boundary
    rectangle('Position', [0, 0, env.width, env.height], 'EdgeColor', 'k', 'LineWidth', 2);
    hold on;
    
    % Draw robots - using numel instead of length
    num_robots = numel(robots);
    for i = 1:num_robots
        if isfield(robots(i), 'failed') && robots(i).failed
            plot(robots(i).position(1), robots(i).position(2), 'rx', 'MarkerSize', 15, 'LineWidth', 3);
        else
            plot(robots(i).position(1), robots(i).position(2), 'bo', 'MarkerSize', 12, 'LineWidth', 2);
        end
        text(robots(i).position(1) + 0.1, robots(i).position(2) + 0.1, ...
             sprintf('Robot %d', i), 'FontSize', 10);
    end
    
    % Draw tasks with colors based on type - using numel instead of length
    num_tasks = numel(tasks);
    for i = 1:num_tasks
        % Determine task type based on name for coloring
        if isfield(tasks(i), 'name')
            if contains(tasks(i).name, {'engine', 'wheel', 'chassis', 'gearbox'})
                color = 'r';  % Red for powertrain/chassis
            elseif contains(tasks(i).name, {'seat', 'dashboard', 'door', 'panel'})
                color = 'g';  % Green for interior/body
            elseif contains(tasks(i).name, {'pcb', 'micro', 'chip', 'resistor'})
                color = 'b';  % Blue for electronics
            elseif contains(tasks(i).name, {'final', 'tightening'})
                color = 'm';  % Magenta for final assembly
            else
                color = 'k';  % Black for other
            end
        else
            color = 'k';
        end
        
        % Draw task
        plot(tasks(i).position(1), tasks(i).position(2), 's', 'Color', color, ...
             'MarkerSize', 10, 'LineWidth', 2, 'MarkerFaceColor', color);
        
        % Display task name
        if isfield(tasks(i), 'name')
            task_name = tasks(i).name;
        else
            task_name = sprintf('Task %d', i);
        end
        text(tasks(i).position(1) + 0.1, tasks(i).position(2) + 0.1, ...
             task_name, 'FontSize', 8);
    end
    
    % Draw dependencies as arrows
    if nargin >= 4 && ~isempty(graph)
        % Use adjacency matrix to find connections
        adj_size = size(graph.adjacency_matrix);
        for i = 1:adj_size(1)
            for j = 1:adj_size(2)
                if graph.adjacency_matrix(i, j) > 0
                    % Draw arrow from task i to task j
                    arrow_x = [tasks(i).position(1), tasks(j).position(1)];
                    arrow_y = [tasks(i).position(2), tasks(j).position(2)];
                    
                    % Calculate arrow direction and length
                    dx = arrow_x(2) - arrow_x(1);
                    dy = arrow_y(2) - arrow_y(1);
                    arrow_length = sqrt(dx^2 + dy^2);
                    
                    % Shorten arrow to avoid overlapping with markers
                    if arrow_length > 0.3
                        arrow_x(2) = arrow_x(1) + 0.9 * dx;
                        arrow_y(2) = arrow_y(1) + 0.9 * dy;
                    end
                    
                    % Check if this is part of the critical path
                    is_critical = ismember(i, graph.critical_path) && ...
                                  ismember(j, graph.critical_path) && ...
                                  abs(find(graph.critical_path == i) - ...
                                      find(graph.critical_path == j)) == 1;
                    
                    if is_critical
                        % Critical path - red, thicker arrow
                        arrow_color = 'r';
                        arrow_width = 1.5;
                    else
                        % Normal dependency - gray, thin arrow
                        arrow_color = [0.7, 0.7, 0.7];
                        arrow_width = 0.5;
                    end
                    
                    % Draw arrow
                    quiver(arrow_x(1), arrow_y(1), ...
                           arrow_x(2) - arrow_x(1), arrow_y(2) - arrow_y(1), ...
                           0, 'Color', arrow_color, 'LineWidth', arrow_width);
                end
            end
        end
    else
        % If no graph provided, use task prerequisites
        for i = 1:num_tasks
            if isfield(tasks(i), 'prerequisites') && ~isempty(tasks(i).prerequisites)
                prereq_count = numel(tasks(i).prerequisites);
                for j = 1:prereq_count
                    prereq = tasks(i).prerequisites(j);
                    if prereq <= num_tasks
                        % Draw arrow from prerequisite to task
                        arrow_x = [tasks(prereq).position(1), tasks(i).position(1)];
                        arrow_y = [tasks(prereq).position(2), tasks(i).position(2)];
                        
                        % Calculate arrow direction and length
                        dx = arrow_x(2) - arrow_x(1);
                        dy = arrow_y(2) - arrow_y(1);
                        arrow_length = sqrt(dx^2 + dy^2);
                        
                        % Shorten arrow to avoid overlapping with markers
                        if arrow_length > 0.3
                            arrow_x(2) = arrow_x(1) + 0.9 * dx;
                            arrow_y(2) = arrow_y(1) + 0.9 * dy;
                        end
                        
                        % Draw arrow
                        quiver(arrow_x(1), arrow_y(1), ...
                               arrow_x(2) - arrow_x(1), arrow_y(2) - arrow_y(1), ...
                               0, 'Color', [0.7, 0.7, 0.7], 'LineWidth', 0.5);
                    end
                end
            end
        end
    end
    
    % Set axis properties
    axis([0, env.width, 0, env.height]);
    axis equal;
    grid on;
    xlabel('X (m)');
    ylabel('Y (m)');
    title('Industrial Assembly Layout');
    
    % Draw legend
    h_legend = [];
    legend_labels = {};
    
    % Add robot to legend
    h = plot(NaN, NaN, 'bo', 'MarkerSize', 10, 'LineWidth', 2);
    h_legend = [h_legend, h];
    legend_labels{end+1} = 'Robot';
    
    % Add failed robot to legend if any
    has_failed = false;
    for i = 1:num_robots
        if isfield(robots(i), 'failed') && robots(i).failed
            has_failed = true;
            break;
        end
    end
    
    if has_failed
        h = plot(NaN, NaN, 'rx', 'MarkerSize', 10, 'LineWidth', 2);
        h_legend = [h_legend, h];
        legend_labels{end+1} = 'Failed Robot';
    end
    
    % Add task types to legend
    task_colors = {'r', 'g', 'b', 'm', 'k'};
    task_types = {'Powertrain/Chassis', 'Interior/Body', 'Electronics', 'Final Assembly', 'Other'};
    
    for i = 1:numel(task_colors)
        h = plot(NaN, NaN, 's', 'Color', task_colors{i}, 'MarkerSize', 10, ...
                 'LineWidth', 2, 'MarkerFaceColor', task_colors{i});
        h_legend = [h_legend, h];
        legend_labels{end+1} = task_types{i};
    end
    
    % Add dependency types to legend
    if nargin >= 4 && ~isempty(graph)
        % Normal dependency
        h = quiver(NaN, NaN, 0, 0, 'Color', [0.7, 0.7, 0.7], 'LineWidth', 0.5);
        h_legend = [h_legend, h];
        legend_labels{end+1} = 'Dependency';
        
        % Critical path
        h = quiver(NaN, NaN, 0, 0, 'Color', 'r', 'LineWidth', 1.5);
        h_legend = [h_legend, h];
        legend_labels{end+1} = 'Critical Path';
    else
        % Just dependency
        h = quiver(NaN, NaN, 0, 0, 'Color', [0.7, 0.7, 0.7], 'LineWidth', 0.5);
        h_legend = [h_legend, h];
        legend_labels{end+1} = 'Dependency';
    end
    
    % Add legend to plot
    legend(h_legend, legend_labels, 'Location', 'Best');
    
    % Second subplot: Gantt chart
    subplot(1, 2, 2);
    
    if nargin >= 4 && ~isempty(graph)
        % Gantt chart using graph data
        % Get earliest start and finish times
        est = graph.earliest_start;
        eft = graph.earliest_finish;
        
        % Create colormap based on task types
        gantt_colors = zeros(num_tasks, 3);
        
        for i = 1:num_tasks
            if isfield(tasks(i), 'name')
                if contains(tasks(i).name, {'engine', 'wheel', 'chassis', 'gearbox'})
                    gantt_colors(i, :) = [1, 0, 0];  % Red
                elseif contains(tasks(i).name, {'seat', 'dashboard', 'door', 'panel'})
                    gantt_colors(i, :) = [0, 0.7, 0];  % Green
                elseif contains(tasks(i).name, {'pcb', 'micro', 'chip', 'resistor'})
                    gantt_colors(i, :) = [0, 0, 1];  % Blue
                elseif contains(tasks(i).name, {'final', 'tightening'})
                    gantt_colors(i, :) = [1, 0, 1];  % Magenta
                else
                    gantt_colors(i, :) = [0.5, 0.5, 0.5];  % Gray
                end
            else
                gantt_colors(i, :) = [0.5, 0.5, 0.5];  % Gray
            end
        end
        
        % Sort tasks by earliest start time
        [~, sort_idx] = sort(est);
        
        % Create Gantt chart
        for i = 1:num_tasks
            idx = sort_idx(i);
            
            % Draw task bar
            if ismember(idx, graph.critical_path)
                % Critical path - draw with red outline
                rectangle('Position', [est(idx), i-0.4, eft(idx)-est(idx), 0.8], ...
                          'FaceColor', gantt_colors(idx, :), 'EdgeColor', 'r', 'LineWidth', 2);
            else
                % Normal task
                rectangle('Position', [est(idx), i-0.4, eft(idx)-est(idx), 0.8], ...
                          'FaceColor', gantt_colors(idx, :), 'EdgeColor', 'k');
            end
            
            % Add task label
            if isfield(tasks(idx), 'name')
                task_name = tasks(idx).name;
            else
                task_name = sprintf('Task %d', idx);
            end
            
            text(est(idx) + 0.1, i, task_name, 'FontSize', 8, 'VerticalAlignment', 'middle');
            
            % Add duration label
            duration = eft(idx) - est(idx);
            if duration > 1
                text(eft(idx) - 0.5, i, sprintf('%.1f', duration), ...
                     'FontSize', 8, 'HorizontalAlignment', 'right', ...
                     'VerticalAlignment', 'middle', 'Color', 'w');
            end
        end
        
        % Add grid lines
        for i = 0:ceil(graph.makespan)
            line([i, i], [0, num_tasks+1], 'Color', [0.9, 0.9, 0.9], 'LineStyle', ':');
        end
        
        % Set axis properties
        ylim([0, num_tasks+1]);
        xlim([0, ceil(graph.makespan * 1.1)]);
        xlabel('Time (minutes)');
        ylabel('Tasks');
        title('Assembly Schedule');
        
        % Create custom y-tick labels
        yticks(1:num_tasks);
        idx_labels = arrayfun(@(x) sprintf('%d', x), sort_idx, 'UniformOutput', false);
        yticklabels(idx_labels);
        
        % Add critical path annotation
        cp_text = sprintf('Critical Path: ');
        critical_path_size = numel(graph.critical_path);
        for i = 1:critical_path_size
            if i > 1
                cp_text = [cp_text, ' → '];
            end
            cp_text = [cp_text, num2str(graph.critical_path(i))];
        end
        cp_text = [cp_text, sprintf('\nMakespan: %.1f minutes', graph.makespan)];
        
        annotation('textbox', [0.53, 0.04, 0.45, 0.06], 'String', cp_text, ...
                   'EdgeColor', 'none', 'FitBoxToText', 'on', 'FontSize', 10);
    else
        % If no graph provided, display message
        text(0.5, 0.5, 'Gantt chart requires task dependency graph', ...
             'HorizontalAlignment', 'center', 'FontSize', 14);
        axis off;
    end
    
    hold off;
    
    fprintf('Industrial scenario visualization complete.\n');
end

function [disturbances] = local_simulateEnvironmentalDisturbances(env, num_disturbances)
    % SIMULATEENVIRONMENTALDISTURBANCES - Create random environmental disturbances
    %
    % Parameters:
    %   env - Environment structure
    %   num_disturbances - Number of disturbances to create (default: 5)
    %
    % Returns:
    %   disturbances - Structure containing environmental disturbances
    
    if nargin < 2
        num_disturbances = 5;
    end
    
    fprintf('Generating %d environmental disturbances...\n', num_disturbances);
    
    % Initialize disturbances structure
    disturbances = struct();
    
    % Pre-initialize obstacle array with all possible fields
    obstacle_count = ceil(num_disturbances / 3);
    obstacles(obstacle_count) = struct('type', '', 'position', [0, 0], 'radius', 0, 'size', [0, 0], 'orientation', 0);
    
    % Generate random obstacles
    for i = 1:obstacle_count
        % Determine obstacle type (circle or rectangle)
        if rand() < 0.5
            % Circular obstacle
            obstacles(i).type = 'circle';
            obstacles(i).position = [rand() * env.width, rand() * env.height];
            obstacles(i).radius = 0.2 + 0.3 * rand();  % 0.2 - 0.5m radius
            % Keep other fields at default values
        else
            % Rectangular obstacle
            obstacles(i).type = 'rectangle';
            obstacles(i).position = [rand() * env.width, rand() * env.height];
            obstacles(i).size = [0.3 + 0.4 * rand(), 0.3 + 0.4 * rand()];  % 0.3 - 0.7m sides
            obstacles(i).orientation = rand() * pi;  % Random orientation
            % radius is unused but must exist in the structure
        end
    end
    
    % Assign to disturbances
    disturbances.obstacles = obstacles;
    
    % Pre-initialize comm_blackouts with consistent structure
    blackout_count = ceil(num_disturbances / 3);
    comm_blackouts(blackout_count) = struct('start_time', 0, 'duration', 0, 'position', [0, 0], 'radius', 0);
    
    % Generate communication blackouts
    for i = 1:blackout_count
        % Random start time
        comm_blackouts(i).start_time = 5 + 45 * rand();  % Start between 5-50 minutes
        
        % Random duration
        comm_blackouts(i).duration = 1 + 4 * rand();  % 1-5 minutes
        
        % Random affected area (circular region)
        comm_blackouts(i).position = [rand() * env.width, rand() * env.height];
        comm_blackouts(i).radius = 1.0 + 1.0 * rand();  % 1-2m radius
    end
    
    disturbances.comm_blackouts = comm_blackouts;
    
    % Pre-initialize sensor_noise with consistent structure
    noise_count = ceil(num_disturbances / 3);
    sensor_noise(noise_count) = struct('start_time', 0, 'duration', 0, 'type', '', 'magnitude', 0);
    
    % Generate sensor noise profiles
    for i = 1:noise_count
        % Random start time
        sensor_noise(i).start_time = 10 + 30 * rand();  % Start between 10-40 minutes
        
        % Random duration
        sensor_noise(i).duration = 2 + 3 * rand();  % 2-5 minutes
        
        % Random noise type (position, force, vision)
        noise_types = {'position', 'force', 'vision'};
        sensor_noise(i).type = noise_types{randi(length(noise_types))};
        
        % Noise magnitude
        switch sensor_noise(i).type
            case 'position'
                sensor_noise(i).magnitude = 0.002 + 0.005 * rand();  % 2-7mm position noise
            case 'force'
                sensor_noise(i).magnitude = 1.0 + 2.0 * rand();  % 1-3N force noise
            case 'vision'
                sensor_noise(i).magnitude = 0.05 + 0.1 * rand();  % 5-15% vision noise
        end
    end
    
    disturbances.sensor_noise = sensor_noise;
    
    fprintf('Environmental disturbances generated:\n');
    fprintf('  - %d obstacles\n', length(disturbances.obstacles));
    fprintf('  - %d communication blackouts\n', length(disturbances.comm_blackouts));
    fprintf('  - %d sensor noise profiles\n', length(disturbances.sensor_noise));
end

function [results] = local_runIndustrialScenario(scenario_type, difficulty_level, include_disturbances)
    % RUNINDUSTRIALSCENARIO - Run a complete industrial assembly scenario
    %
    % Parameters:
    %   scenario_type - Type of scenario ('automotive', 'electronics', 'furniture')
    %   difficulty_level - Level of scenario complexity (1-4, default: 2)
    %   include_disturbances - Whether to include environmental disturbances (default: true)
    %
    % Returns:
    %   results - Structure containing scenario results
    
    if nargin < 2
        difficulty_level = 2;
    end
    
    if nargin < 3
        include_disturbances = true;
    end
    
    fprintf('==============================================\n');
    fprintf('RUNNING INDUSTRIAL SCENARIO: %s (Level %d)\n', upper(scenario_type), difficulty_level);
    fprintf('==============================================\n\n');
    
    % Load utility functions using utils_manager for general utilities
    utils = utils_manager();
    
    % Load auction utilities directly to avoid field name issues
    auction_utils_instance = auction_utils();
    
    % Create environment
    env = utils.env.createEnvironment(4, 4);
    fprintf('Environment created successfully\n');
    
    % Create robots with different capabilities
    robots = utils.robot.createRobots(2, env);
    fprintf('Robots created successfully\n');
    
    % Create industry-specific tasks
    switch lower(scenario_type)
        case 'automotive'
            tasks = local_createAutomotiveAssemblyTasks(env, difficulty_level);
            
        case 'electronics'
            tasks = local_createElectronicsAssemblyTasks(env, difficulty_level);
            
        case 'furniture'
            tasks = local_createFurnitureAssemblyTasks(env, difficulty_level);
            
        otherwise
            error('Invalid scenario type. Choose "automotive", "electronics", or "furniture".');
    end
    
    % Apply industrial tolerances
    tasks = local_applyIndustrialTolerances(tasks, scenario_type);
    
    % Generate task dependency graph
    [graph, critical_path] = local_generateTaskDependencyGraph(tasks);
    
    % Create environmental disturbances if requested
    if include_disturbances
        disturbances = local_simulateEnvironmentalDisturbances(env, 5);
    else
        disturbances = [];
    end
    
    % Visualize the scenario
    local_visualizeIndustrialScenario(env, robots, tasks, graph);
    
    % Set up auction parameters
    params = struct();
    params.epsilon = 0.02;        % Reduced from 0.05 to allow finer bidding
    params.alpha = [2.0, 1.5, 0.5, 0.8, 0.3];  % Increased capability weight
    params.gamma = 0.7;           % Increased from 0.5 for faster consensus
    params.beta = [2.0, 1.5];     % Stronger recovery weights for failures
    params.lambda = 0.1;          % Information decay rate
    params.comm_delay = 0;        % Communication delay (in iterations)
    params.packet_loss_prob = 0;  % Probability of packet loss
    
    % Add failure scenario for higher difficulty levels
    if difficulty_level >= 3
        params.failure_time = 20;     % Time of robot failure
        params.failed_robot = 1;      % Robot 1 fails
    else
        params.failure_time = inf;    % No failure
        params.failed_robot = [];     % No robot fails
    end
    
    % Initialize auction data - using direct auction_utils call
    auction_data = auction_utils_instance.initializeAuctionData(tasks, robots);
    
    % Run auction algorithm simulation - using direct auction_utils call
    fprintf('\nRunning distributed auction algorithm...\n');
    [metrics, converged] = auction_utils_instance.runAuctionSimulation(params, env, robots, tasks, true);
    
    % Create result structure
    results = struct();
    results.scenario_type = scenario_type;
    results.difficulty_level = difficulty_level;
    results.include_disturbances = include_disturbances;
    results.tasks = tasks;
    results.robots = robots;
    results.env = env;
    results.graph = graph;
    results.critical_path = critical_path;
    results.metrics = metrics;
    results.converged = converged;
    results.params = params;
    results.auction_data = auction_data;
    
    if include_disturbances
        results.disturbances = disturbances;
    end
    
    % Print summary
    fprintf('\n==============================================\n');
    fprintf('SCENARIO SUMMARY\n');
    fprintf('==============================================\n');
    fprintf('Scenario: %s (Level %d)\n', scenario_type, difficulty_level);
    fprintf('Number of tasks: %d\n', length(tasks));
    fprintf('Optimal makespan: %.2f\n', graph.makespan);
    fprintf('Auction makespan: %.2f\n', metrics.makespan);
    fprintf('Optimality gap: %.2f\n', metrics.optimality_gap);
    fprintf('Iterations to converge: %d\n', metrics.iterations);
    fprintf('Messages exchanged: %d\n', metrics.messages);
    
    if difficulty_level >= 3
        fprintf('Failure recovery time: %d\n', metrics.recovery_time);
    end
    
    fprintf('\nSaving results...\n');
    
    % Create filename
    filename = sprintf('%s_level%d_scenario_results.mat', scenario_type, difficulty_level);
    save(filename, 'results');
    
    fprintf('Results saved to %s\n', filename);
end