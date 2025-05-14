function utils = motion_planning_utils()
    % MOTION_PLANNING_UTILS - Returns function handles for motion planning
    utils = struct(...
        'createPathRRT', @createPathRRT, ...
        'createPathAStar', @createPathAStar, ...
        'generateTrajectory', @generateTrajectory, ...
        'checkCollision', @checkCollision, ...
        'planManipulatorMotion', @planManipulatorMotion ...
    );
    
    function path = createPathRRT(start, goal, obstacles, env, max_iterations)
        % CREATEPATHRRT Create a path using RRT algorithm
        % start: Start position [x, y]
        % goal: Goal position [x, y]
        % obstacles: Array of obstacle structures
        % env: Environment structure
        % max_iterations: Maximum number of iterations
        
        if nargin < 5
            max_iterations = 5000;
        end
        
        % Initialize RRT
        nodes = start;
        parents = 0;
        
        % RRT parameters
        step_size = 0.2;
        goal_bias = 0.1;
        
        for i = 1:max_iterations
            % Sample random point
            if rand() < goal_bias
                sample = goal;
            else
                sample = [rand() * env.width, rand() * env.height];
            end
            
            % Find nearest node
            [nearest_idx, nearest_node] = findNearestNode(nodes, sample);
            
            % Extend towards sample
            direction = sample - nearest_node;
            distance = norm(direction);
            
            if distance > 0
                direction = direction / distance;
                new_node = nearest_node + direction * min(step_size, distance);
                
                % Check if new node is valid
                if ~checkCollision(nearest_node, new_node, obstacles, env)
                    % Add node to tree
                    nodes = [nodes; new_node];
                    parents = [parents; nearest_idx];
                    
                    % Check if goal is reached
                    if norm(new_node - goal) < step_size
                        % Construct path
                        path = constructPath(nodes, parents, size(nodes, 1));
                        return;
                    end
                end
            end
        end
        
        % If goal not reached, return empty path
        fprintf('RRT could not find a path within %d iterations\n', max_iterations);
        path = [];
    end
    
    function [nearest_idx, nearest_node] = findNearestNode(nodes, sample)
        % Find the nearest node in the tree
        distances = sum((nodes - sample).^2, 2);
        [~, nearest_idx] = min(distances);
        nearest_node = nodes(nearest_idx, :);
    end
    
    function path = constructPath(nodes, parents, goal_idx)
        % Construct path from start to goal
        path = nodes(goal_idx, :);
        parent_idx = parents(goal_idx);
        
        while parent_idx > 0
            path = [nodes(parent_idx, :); path];
            parent_idx = parents(parent_idx);
        end
    end
    
    function path = createPathAStar(start, goal, obstacles, env, grid_resolution)
        % CREATEPATHASTAR Create a path using A* algorithm
        % start: Start position [x, y]
        % goal: Goal position [x, y]
        % obstacles: Array of obstacle structures
        % env: Environment structure
        % grid_resolution: Resolution of the grid [m]
        
        if nargin < 5
            grid_resolution = 0.1;
        end
        
        % Create grid representation
        grid_width = ceil(env.width / grid_resolution);
        grid_height = ceil(env.height / grid_resolution);
        
        % Initialize grid (0 = free, 1 = obstacle)
        grid = zeros(grid_width, grid_height);
        
        % Add obstacles to grid
        for i = 1:length(obstacles)
            if isfield(obstacles, 'position') && isfield(obstacles, 'radius')
                % Circular obstacle
                center_x = obstacles(i).position(1);
                center_y = obstacles(i).position(2);
                radius = obstacles(i).radius;
                
                % Find grid cells that intersect with obstacle
                x_min = max(1, floor((center_x - radius) / grid_resolution) + 1);
                x_max = min(grid_width, ceil((center_x + radius) / grid_resolution));
                y_min = max(1, floor((center_y - radius) / grid_resolution) + 1);
                y_max = min(grid_height, ceil((center_y + radius) / grid_resolution));
                
                for x = x_min:x_max
                    for y = y_min:y_max
                        cell_x = (x - 0.5) * grid_resolution;
                        cell_y = (y - 0.5) * grid_resolution;
                        
                        if norm([cell_x - center_x, cell_y - center_y]) <= radius
                            grid(x, y) = 1;
                        end
                    end
                end
            elseif isfield(obstacles, 'position') && isfield(obstacles, 'size')
                % Rectangular obstacle
                pos_x = obstacles(i).position(1);
                pos_y = obstacles(i).position(2);
                size_x = obstacles(i).size(1);
                size_y = obstacles(i).size(2);
                
                % Find grid cells that intersect with obstacle
                x_min = max(1, floor((pos_x - size_x/2) / grid_resolution) + 1);
                x_max = min(grid_width, ceil((pos_x + size_x/2) / grid_resolution));
                y_min = max(1, floor((pos_y - size_y/2) / grid_resolution) + 1);
                y_max = min(grid_height, ceil((pos_y + size_y/2) / grid_resolution));
                
                grid(x_min:x_max, y_min:y_max) = 1;
            end
        end
        
        % Convert start and goal to grid coordinates
        start_grid = round([start(1), start(2)] / grid_resolution) + 1;
        goal_grid = round([goal(1), goal(2)] / grid_resolution) + 1;
        
        % Ensure start and goal are within grid bounds
        start_grid = min(max(start_grid, [1, 1]), [grid_width, grid_height]);
        goal_grid = min(max(goal_grid, [1, 1]), [grid_width, grid_height]);
        
        % Check if start or goal is in obstacle
        if grid(start_grid(1), start_grid(2)) == 1
            fprintf('Start position is in obstacle\n');
            path = [];
            return;
        end
        
        if grid(goal_grid(1), goal_grid(2)) == 1
            fprintf('Goal position is in obstacle\n');
            path = [];
            return;
        end
        
        % A* algorithm
        % Initialize open and closed sets
        open_set = PriorityQueue();
        closed_set = zeros(grid_width, grid_height);
        
        % Initialize g and f scores
        g_score = inf(grid_width, grid_height);
        f_score = inf(grid_width, grid_height);
        
        % Initialize parent pointers
        parent_x = zeros(grid_width, grid_height);
        parent_y = zeros(grid_width, grid_height);
        
        % Set start node
        g_score(start_grid(1), start_grid(2)) = 0;
        f_score(start_grid(1), start_grid(2)) = heuristic(start_grid, goal_grid);
        
        % Add start node to open set
        open_set.insert([start_grid(1), start_grid(2)], f_score(start_grid(1), start_grid(2)));
        
        % A* main loop
        while ~open_set.isEmpty()
            % Get node with lowest f_score
            current = open_set.pop();
            
            % If goal is reached
            if current(1) == goal_grid(1) && current(2) == goal_grid(2)
                % Reconstruct path
                path = reconstructPath(parent_x, parent_y, start_grid, goal_grid, grid_resolution);
                return;
            end
            
            % Mark current node as closed
            closed_set(current(1), current(2)) = 1;
            
            % Check neighbors
            for dx = -1:1
                for dy = -1:1
                    % Skip current node
                    if dx == 0 && dy == 0
                        continue;
                    end
                    
                    % Calculate neighbor coordinates
                    neighbor = [current(1) + dx, current(2) + dy];
                    
                    % Check if neighbor is within grid bounds
                    if neighbor(1) < 1 || neighbor(1) > grid_width || ...
                       neighbor(2) < 1 || neighbor(2) > grid_height
                        continue;
                    end
                    
                    % Check if neighbor is obstacle or closed
                    if grid(neighbor(1), neighbor(2)) == 1 || closed_set(neighbor(1), neighbor(2)) == 1
                        continue;
                    end
                    
                    % Calculate tentative g score
                    if dx == 0 || dy == 0
                        % Orthogonal movement
                        tentative_g = g_score(current(1), current(2)) + 1;
                    else
                        % Diagonal movement
                        tentative_g = g_score(current(1), current(2)) + sqrt(2);
                    end
                    
                    % If neighbor not in open set or better path found
                    if tentative_g < g_score(neighbor(1), neighbor(2))
                        % Update parent
                        parent_x(neighbor(1), neighbor(2)) = current(1);
                        parent_y(neighbor(1), neighbor(2)) = current(2);
                        
                        % Update scores
                        g_score(neighbor(1), neighbor(2)) = tentative_g;
                        f_score(neighbor(1), neighbor(2)) = tentative_g + heuristic(neighbor, goal_grid);
                        
                        % Add to open set
                        open_set.insert(neighbor, f_score(neighbor(1), neighbor(2)));
                    end
                end
            end
        end
        
        % If goal not reached, return empty path
        fprintf('A* could not find a path\n');
        path = [];
    end
    
    function h = heuristic(node, goal)
        % Euclidean distance heuristic
        h = norm(node - goal);
    end
    
    function path = reconstructPath(parent_x, parent_y, start_grid, goal_grid, grid_resolution)
        % Reconstruct path from goal to start
        path = [goal_grid - 0.5] * grid_resolution;
        current = goal_grid;
        
        while current(1) ~= start_grid(1) || current(2) ~= start_grid(2)
            current = [parent_x(current(1), current(2)), parent_y(current(1), current(2))];
            path = [(current - 0.5) * grid_resolution; path];
        end
    end
    
    function trajectory = generateTrajectory(path, max_velocity, max_acceleration, dt)
        % GENERATETRAJECTORY Generate a time-parameterized trajectory from a path
        % path: Array of waypoints [x, y]
        % max_velocity: Maximum velocity [m/s]
        % max_acceleration: Maximum acceleration [m/s^2]
        % dt: Time step [s]
        
        if isempty(path) || size(path, 1) < 2
            trajectory = struct('positions', [], 'velocities', [], 'accelerations', [], 'times', []);
            return;
        end
        
        % Calculate path lengths
        segments = diff(path, 1);
        segment_lengths = sqrt(sum(segments.^2, 2));
        
        % Calculate path directions
        directions = segments ./ segment_lengths;
        
        % Time to traverse each segment with constant velocity
        segment_times = segment_lengths / max_velocity;
        
        % Total path length
        total_length = sum(segment_lengths);
        
        % Time to accelerate to max velocity
        t_acc = max_velocity / max_acceleration;
        
        % Distance covered during acceleration
        d_acc = 0.5 * max_acceleration * t_acc^2;
        
        % Check if path is too short for max velocity
        if total_length < 2 * d_acc
            % Cannot reach max velocity, calculate peak velocity
            peak_velocity = sqrt(max_acceleration * total_length);
            t_acc = peak_velocity / max_acceleration;
            d_acc = 0.5 * max_acceleration * t_acc^2;
            t_const = 0;  % No constant velocity phase
        else
            % Can reach max velocity
            peak_velocity = max_velocity;
            t_const = (total_length - 2 * d_acc) / max_velocity;  % Time at constant velocity
        end
        
        % Total time
        total_time = 2 * t_acc + t_const;
        
        % Number of time steps
        num_steps = ceil(total_time / dt);
        
        % Initialize trajectory
        positions = zeros(num_steps, 2);
        velocities = zeros(num_steps, 2);
        accelerations = zeros(num_steps, 2);
        times = (0:num_steps-1)' * dt;
        
        % Generate trajectory
        for i = 1:num_steps
            t = times(i);
            
            % Calculate distance along path
            if t < t_acc
                % Acceleration phase
                s = 0.5 * max_acceleration * t^2;
                v = max_acceleration * t;
                a = max_acceleration;
            elseif t < t_acc + t_const
                % Constant velocity phase
                s = d_acc + max_velocity * (t - t_acc);
                v = max_velocity;
                a = 0;
            else
                % Deceleration phase
                t_dec = t - (t_acc + t_const);
                s = total_length - 0.5 * max_acceleration * (t_acc - t_dec)^2;
                v = max_velocity - max_acceleration * t_dec;
                a = -max_acceleration;
            end
            
            % Find segment containing s
            s_accumulated = 0;
            seg_idx = 1;
            
            while seg_idx < length(segment_lengths) && s_accumulated + segment_lengths(seg_idx) < s
                s_accumulated = s_accumulated + segment_lengths(seg_idx);
                seg_idx = seg_idx + 1;
            end
            
            % Calculate position within segment
            s_in_segment = s - s_accumulated;
            ratio = min(1, s_in_segment / segment_lengths(seg_idx));
            
            % Calculate position
            positions(i, :) = path(seg_idx, :) + ratio * segments(seg_idx, :);
            
            % Calculate velocity and acceleration vectors
            velocities(i, :) = v * directions(seg_idx, :);
            accelerations(i, :) = a * directions(seg_idx, :);
        end
        
        % Return trajectory structure
        trajectory = struct(...
            'positions', positions, ...
            'velocities', velocities, ...
            'accelerations', accelerations, ...
            'times', times ...
        );
    end
    
    function collision = checkCollision(p1, p2, obstacles, env)
        % CHECKCOLLISION Check if a line segment collides with obstacles
        % p1, p2: Start and end points of line segment
        % obstacles: Array of obstacle structures
        % env: Environment structure
        
        % Check environment boundaries
        if p1(1) < 0 || p1(1) > env.width || p1(2) < 0 || p1(2) > env.height || ...
           p2(1) < 0 || p2(1) > env.width || p2(2) < 0 || p2(2) > env.height
            collision = true;
            return;
        end
        
        % Check obstacles
        for i = 1:length(obstacles)
            if isfield(obstacles, 'position') && isfield(obstacles, 'radius')
                % Circular obstacle
                if checkLineCircleIntersection(p1, p2, obstacles(i).position, obstacles(i).radius)
                    collision = true;
                    return;
                end
            elseif isfield(obstacles, 'position') && isfield(obstacles, 'size')
                % Rectangular obstacle
                if checkLineRectIntersection(p1, p2, obstacles(i).position, obstacles(i).size)
                    collision = true;
                    return;
                end
            end
        end
        
        % No collision
        collision = false;
    end
    
    function collision = checkLineCircleIntersection(p1, p2, center, radius)
        % Check if line segment intersects with circle
        
        % Vector from p1 to p2
        v = p2 - p1;
        
        % Vector from p1 to circle center
        w = center - p1;
        
        % Length of line segment
        len_v = norm(v);
        
        if len_v == 0
            % p1 and p2 are the same point
            collision = (norm(w) <= radius);
            return;
        end
        
        % Normalize v
        v = v / len_v;
        
        % Projection of w onto v
        proj = dot(w, v);
        
        % Closest point on line segment to circle center
        if proj < 0
            closest = p1;
        elseif proj > len_v
            closest = p2;
        else
            closest = p1 + proj * v;
        end
        
        % Check if closest point is within radius
        collision = (norm(closest - center) <= radius);
    end
    
    function collision = checkLineRectIntersection(p1, p2, rect_center, rect_size)
        % Check if line segment intersects with rectangle
        
        % Rectangle corners
        rect_min = rect_center - rect_size/2;
        rect_max = rect_center + rect_size/2;
        
        % Check if either endpoint is inside rectangle
        if (p1(1) >= rect_min(1) && p1(1) <= rect_max(1) && ...
            p1(2) >= rect_min(2) && p1(2) <= rect_max(2)) || ...
           (p2(1) >= rect_min(1) && p2(1) <= rect_max(1) && ...
            p2(2) >= rect_min(2) && p2(2) <= rect_max(2))
            collision = true;
            return;
        end
        
        % Check if line segment intersects rectangle edges
        % Line from p1 to p2: p1 + t * (p2 - p1), t in [0, 1]
        
        % Test against x = rect_min(1)
        if p1(1) ~= p2(1)  % Not vertical
            t = (rect_min(1) - p1(1)) / (p2(1) - p1(1));
            if t >= 0 && t <= 1
                y = p1(2) + t * (p2(2) - p1(2));
                if y >= rect_min(2) && y <= rect_max(2)
                    collision = true;
                    return;
                end
            end
        end
        
        % Test against x = rect_max(1)
        if p1(1) ~= p2(1)  % Not vertical
            t = (rect_max(1) - p1(1)) / (p2(1) - p1(1));
            if t >= 0 && t <= 1
                y = p1(2) + t * (p2(2) - p1(2));
                if y >= rect_min(2) && y <= rect_max(2)
                    collision = true;
                    return;
                end
            end
        end
        
        % Test against y = rect_min(2)
        if p1(2) ~= p2(2)  % Not horizontal
            t = (rect_min(2) - p1(2)) / (p2(2) - p1(2));
            if t >= 0 && t <= 1
                x = p1(1) + t * (p2(1) - p1(1));
                if x >= rect_min(1) && x <= rect_max(1)
                    collision = true;
                    return;
                end
            end
        end
        
        % Test against y = rect_max(2)
        if p1(2) ~= p2(2)  % Not horizontal
            t = (rect_max(2) - p1(2)) / (p2(2) - p1(2));
            if t >= 0 && t <= 1
                x = p1(1) + t * (p2(1) - p1(1));
                if x >= rect_min(1) && x <= rect_max(1)
                    collision = true;
                    return;
                end
            end
        end
        
        % No intersection
        collision = false;
    end
    
    function [joint_trajectory, success] = planManipulatorMotion(start_angles, goal_angles, obstacle_points, joint_limits, steps)
        % PLANMANIPULATORMOTION Plan a trajectory for the manipulator joints
        % start_angles: Initial joint angles [rad]
        % goal_angles: Goal joint angles [rad]
        % obstacle_points: Array of obstacle points in workspace
        % joint_limits: Joint limits [min, max]
        % steps: Number of steps in trajectory
        
        if nargin < 5
            steps = 50;
        end
        
        % Load kinematics utilities
        kinematic_fns = kinematics_utils();
        
        % Simple linear interpolation between start and goal
        joint_trajectory = zeros(steps, length(start_angles));
        
        for i = 1:steps
            t = (i-1) / (steps-1);
            joint_trajectory(i, :) = (1-t) * start_angles(:)' + t * goal_angles(:)';
        end
        
        % Check for collisions and joint limits
        success = true;
        
        % Check joint limits
        for i = 1:steps
            for j = 1:length(start_angles)
                if joint_trajectory(i, j) < joint_limits(j, 1) || joint_trajectory(i, j) > joint_limits(j, 2)
                    success = false;
                    break;
                end
            end
            
            if ~success
                break;
            end
        end
        
        % TODO: Add collision checking with obstacles
        
        return;
    end
end