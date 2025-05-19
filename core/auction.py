# core/auction.py
import numpy as np
import time
import random
import torch

from core.gpu_accelerator import GPUAccelerator

class DistributedAuction:
    def __init__(self, epsilon=0.01, communication_delay=0, packet_loss_prob=0, use_gpu=True):
        """Initialize distributed auction algorithm
        
        Args:
            epsilon: Minimum bid increment (controls optimality gap and convergence)
            communication_delay: Communication delay in ms
            packet_loss_prob: Probability of packet loss (0-1)
            use_gpu: Whether to use GPU acceleration when available
        """
        self.epsilon = epsilon
        self.communication_delay = communication_delay / 1000.0  # Convert ms to seconds
        self.packet_loss_prob = packet_loss_prob
        self.weights = {
            'alpha1': 0.3,  # Distance weight
            'alpha2': 0.2,  # Configuration cost weight
            'alpha3': 0.3,  # Capability similarity weight
            'alpha4': 0.1,  # Workload weight
            'alpha5': 0.1,  # Energy consumption weight
            'W': np.eye(6)  # Weight matrix for configuration
        }
        self.beta_weights = {
            'beta1': 0.5,  # Progress weight
            'beta2': 0.3,  # Criticality weight
            'beta3': 0.2   # Urgency weight
        }
        
        # Initialize GPU accelerator if requested
        self.use_gpu = use_gpu
        if use_gpu:
            try:
                self.gpu = GPUAccelerator()
                self.use_gpu = self.gpu.using_gpu
            except Exception as e:
                print(f"Could not initialize GPU acceleration: {e}")
                self.use_gpu = False
    
    def run_auction(self, robots, tasks, task_graph):
        """Run distributed auction algorithm for task allocation
        
        Args:
            robots: List of Robot objects
            tasks: List of Task objects
            task_graph: TaskDependencyGraph object
            
        Returns:
            tuple: (assignments dict, message count)
        """
        # Find unassigned tasks that are ready (prerequisites completed)
        unassigned_tasks = [task for task in tasks 
                           if task.assigned_to == 0 and 
                           task.status == 'pending' and 
                           task_graph.is_available(task.id)]
        
        if not unassigned_tasks:
            return {}, 0  # No tasks to assign
            
        # Use GPU implementation if enabled and possible
        if self.use_gpu and len(robots) > 0 and len(unassigned_tasks) > 0:
            return self._run_auction_gpu(robots, unassigned_tasks, tasks)
        else:
            return self._run_auction_cpu(robots, unassigned_tasks, tasks)
    
    def _run_auction_gpu(self, robots, unassigned_tasks, all_tasks):
        """GPU-accelerated auction implementation"""
        # Prepare data for GPU processing
        robot_positions = np.array([robot.position for robot in robots])
        robot_capabilities = np.array([robot.capabilities for robot in robots])
        robot_statuses = [robot.status for robot in robots]
        
        task_positions = np.array([task.position for task in unassigned_tasks])
        task_capabilities = np.array([task.capabilities for task in unassigned_tasks])
        
        # Map unassigned tasks to their indices in all_tasks
        task_id_to_idx = {task.id: i for i, task in enumerate(unassigned_tasks)}
        
        # Current assignments and prices
        task_assignments = np.zeros(len(unassigned_tasks), dtype=np.int32)
        prices = np.zeros(len(unassigned_tasks), dtype=np.float32)
        
        # Run GPU auction
        new_assignments, new_prices, messages = self.gpu.run_auction_gpu(
            robot_positions, robot_capabilities, robot_statuses,
            task_positions, task_capabilities, task_assignments,
            self.epsilon, prices
        )
        
        # Convert results back to dictionary format
        assignments = {}
        for i, task in enumerate(unassigned_tasks):
            robot_idx = new_assignments[i]
            if robot_idx > 0:  # If assigned
                robot_id = robots[robot_idx-1].id
                task.assigned_to = robot_id
                assignments[task.id] = robot_id
        
        return assignments, messages
    
    # core/robot.py
import numpy as np

class Robot:
    def __init__(self, robot_id, position, orientation, config=None, capabilities=None):
        """Initialize a robot with parameters matching TurtleBot3 Waffle Pi with OpenMANIPULATOR-X
        
        Args:
            robot_id: Unique identifier for the robot
            position: Initial position as [x, y] array
            orientation: Initial orientation in radians
            config: Initial joint configuration (4-DOF manipulator + gripper)
            capabilities: Capability vector describing robot abilities
        """
        self.id = robot_id
        self.position = np.array(position, dtype=float)
        self.orientation = float(orientation)
        # Reduced from 6-DOF to 4-DOF + gripper to match OpenMANIPULATOR-X
        self.config = np.zeros(5) if config is None else np.array(config)
        self.capabilities = np.random.rand(5) if capabilities is None else np.array(capabilities)
        self.status = 'operational'  # 'operational', 'failed', 'partial_failure'
        self.workload = 0
        self.assigned_tasks = []
        self.trajectory = [np.copy(position)]  # Store trajectory for visualization
        
        # Kinematics parameters - Updated to match TurtleBot3 Waffle Pi
        self.max_linear_velocity = 0.26  # m/s (reduced from 0.5 to match platform)
        self.max_angular_velocity = 1.82  # rad/s (updated from 1.0 to match platform)
        self.base_dimensions = [0.281, 0.306]  # width, length in meters
        self.manipulator_reach = 0.380  # meters (reduced from 0.85m)
        self.manipulator_payload = 0.5  # kg (reduced from 5kg)
        
        # Virtual F/T sensor for force-controlled operations
        self.ft_sensor = {'force': np.zeros(3), 'torque': np.zeros(3)}
        
        # For visualization
        self.color = 'blue'
    
    def update_status(self, status):
        """Update robot operational status"""
        self.status = status
        if status != 'operational':
            self.color = 'red' if status == 'failed' else 'orange'
        else:
            self.color = 'blue'
    
    def calculate_bid(self, task, weights, current_workload, energy_consumption=0):
        """Calculate bid for a task based on the formula from the paper
        
        b_ij = (α₁/d_ij) + (α₂/c_ij) + α₃·s_ij - α₄·l_i - α₅·e_ij
        
        Args:
            task: Task object to calculate bid for
            weights: Dictionary of weight parameters
            current_workload: Current workload value
            energy_consumption: Estimated energy consumption (optional)
            
        Returns:
            float: Bid value
        """
        # Extract weights
        alpha1 = weights['alpha1']
        alpha2 = weights['alpha2']
        alpha3 = weights['alpha3']
        alpha4 = weights['alpha4']
        alpha5 = weights['alpha5']
        W = weights['W']
        
        # Calculate distance to task
        # Use faster Euclidean distance calculation for 2D points
        dx = self.position[0] - task.position[0]
        dy = self.position[1] - task.position[1]
        distance = (dx*dx + dy*dy)**0.5
        
        # Calculate configuration transition cost - adapted for 4-DOF manipulator
        # Use only first 4 components for configuration to match OpenMANIPULATOR-X
        config_diff = self.config[:4] - task.required_config[:4]
        # Ensure W matrix is properly sized for 4-DOF
        W_sub = W[:4,:4] if W.shape[0] > 4 else W
        config_cost = np.sqrt(config_diff.T @ W_sub @ config_diff)
        
        # Calculate capability similarity
        robot_cap_norm = np.linalg.norm(self.capabilities)
        task_cap_norm = np.linalg.norm(task.capabilities)
        
        if robot_cap_norm > 0 and task_cap_norm > 0:
            # Use dot product for faster calculation
            capability_similarity = np.dot(self.capabilities, task.capabilities) / (robot_cap_norm * task_cap_norm)
        else:
            capability_similarity = 0
        
        # Calculate bid
        term1 = alpha1 / distance if distance > 0 else float('inf')
        term2 = alpha2 / config_cost if config_cost > 0 else float('inf')
        
        bid = term1 + term2 + (alpha3 * capability_similarity) - (alpha4 * current_workload) - (alpha5 * energy_consumption)
        return bid
    
    def calculate_recovery_bid(self, standard_bid, task_progress, task_criticality, task_urgency, beta_weights):
        """Calculate recovery bid for task reallocation after robot failure
        
        b^r_ij = b_ij + β₁(1 - p_j) + β₂·criticality(j) + β₃·urgency(j)
        
        Args:
            standard_bid: Standard bid value
            task_progress: Current progress on the task (0-1)
            task_criticality: Criticality measure of the task
            task_urgency: Urgency measure of the task
            beta_weights: Dictionary of beta weight parameters
            
        Returns:
            float: Recovery bid value
        """
        beta1 = beta_weights['beta1']
        beta2 = beta_weights['beta2']
        beta3 = beta_weights['beta3']
        
        recovery_bid = standard_bid + \
                      beta1 * (1 - task_progress) + \
                      beta2 * task_criticality + \
                      beta3 * task_urgency
        return recovery_bid
    
    def update_position(self, target_position, dt, max_speed=None):
        """Move robot toward target position
        
        Args:
            target_position: Target position to move toward
            dt: Time step
            max_speed: Maximum speed (or None to use robot's max_linear_velocity)
            
        Returns:
            bool: True if target reached, False otherwise
        """
        if max_speed is None:
            max_speed = self.max_linear_velocity
            
        # Get direction vector
        direction = target_position - self.position
        distance = np.linalg.norm(direction)
        
        # If already at target (or very close)
        if distance < 0.1:
            return True
            
        # Normalize direction
        direction = direction / distance
        
        # Calculate movement distance for this time step
        move_distance = min(max_speed * dt, distance)
        
        # Update position
        new_position = self.position + move_distance * direction
        self.position = new_position
        
        # Store trajectory point
        self.trajectory.append(np.copy(new_position))
        
        # Update orientation based on movement direction
        target_orientation = np.arctan2(direction[1], direction[0])
        orientation_diff = target_orientation - self.orientation
        
        # Normalize to [-pi, pi]
        while orientation_diff > np.pi:
            orientation_diff -= 2*np.pi
        while orientation_diff < -np.pi:
            orientation_diff += 2*np.pi
        
        # Apply angular velocity constraint
        max_rotation = self.max_angular_velocity * dt
        if abs(orientation_diff) > max_rotation:
            orientation_diff = np.sign(orientation_diff) * max_rotation
            
        # Update orientation
        self.orientation += orientation_diff
        
        # Return whether target is reached
        return distance < 0.1
    
    def estimate_travel_time(self, target_position):
        """Estimate travel time to target position"""
        distance = np.linalg.norm(target_position - self.position)
        return distance / self.max_linear_velocity
    
    def apply_partial_failure(self, capability_reduction_factor=0.5):
        """Apply partial failure to robot by reducing capabilities"""
        # Create random mask with values between capability_reduction_factor and 1
        mask = capability_reduction_factor + (1-capability_reduction_factor) * np.random.rand(len(self.capabilities))
        self.capabilities = self.capabilities * mask
        self.update_status('partial_failure')
        
    def reset_trajectory(self):
        """Reset stored trajectory"""
        self.trajectory = [np.copy(self.position)]
        
    def update_ft_sensor(self, force, torque):
        """Update virtual force/torque sensor readings
        
        Args:
            force: 3D force vector [fx, fy, fz]
            torque: 3D torque vector [tx, ty, tz]
        """
        self.ft_sensor['force'] = np.array(force)
        self.ft_sensor['torque'] = np.array(torque)

    def _run_auction_cpu(self, robots, unassigned_tasks, tasks):
        """Original CPU implementation"""
        # Track assignments and prices
        prices = {task.id: 0.0 for task in tasks}
        assignments = {task.id: 0 for task in tasks}
        messages_sent = 0
        
        # Set max_iterations based on theoretical bound: K² · bₘₐₓ/ε 
        # where K is task count and bₘₐₓ is maximum possible bid
        K = len(unassigned_tasks)
        b_max = 100.0  # Estimate of maximum possible bid value
        theoretical_max_iter = int(K**2 * b_max / self.epsilon)
        
        # Set a practical upper limit to prevent excessive iterations
        max_iterations = min(theoretical_max_iter, 1000)
        
        # Run auction algorithm
        iter_count = 0
        
        while unassigned_tasks and iter_count < max_iterations:
            iter_count += 1
            tasks_assigned_this_iter = 0
            
            # For each robot, calculate bids for unassigned tasks
            for robot in robots:
                # Skip failed robots
                if robot.status != 'operational':
                    continue
                
                # Calculate current workload
                workload = sum(task.completion_time for task in tasks 
                            if task.assigned_to == robot.id and
                            task.status != 'completed')
                robot.workload = workload
                
                # Find best task for this robot
                best_utility = float('-inf')
                best_task = None
                best_bid = 0
                
                for task in unassigned_tasks:
                    # Skip collaborative tasks for simplicity
                    if task.collaborative:
                        continue
                    
                    # Calculate bid
                    bid = robot.calculate_bid(task, self.weights, workload)
                    
                    # Apply communication delay
                    if self.communication_delay > 0:
                        time.sleep(self.communication_delay)
                    
                    # Check for packet loss
                    if random.random() < self.packet_loss_prob:
                        continue  # Simulate packet loss
                    
                    messages_sent += 1
                    
                    # Calculate utility (bid - price)
                    utility = bid - prices[task.id]
                    
                    if utility > best_utility:
                        best_utility = utility
                        best_task = task
                        best_bid = bid
                
                # If found a task with positive utility, propose assignment
                if best_task and best_utility > 0:
                    # Update price - critical for maintaining 2ε optimality gap
                    prices[best_task.id] = prices[best_task.id] + self.epsilon + best_utility
                    
                    # Assign task to robot
                    best_task.assigned_to = robot.id
                    assignments[best_task.id] = robot.id
                    
                    # Remove from unassigned tasks
                    unassigned_tasks.remove(best_task)
                    
                    messages_sent += 1  # Assignment message
                    tasks_assigned_this_iter += 1
            
            # If no tasks were assigned in this iteration, break to ensure convergence
            if tasks_assigned_this_iter == 0:
                break
        
        # Log if we hit the iteration limit (useful for debugging)
        if iter_count >= max_iterations and unassigned_tasks:
            print(f"Warning: CPU Auction reached maximum iterations ({max_iterations}) with {len(unassigned_tasks)} tasks still unassigned.")
        
        # Handle collaborative tasks (simplified)
        collaborative_tasks = [task for task in tasks 
                            if task.collaborative and task.assigned_to == 0 and
                            task.status == 'pending']
        
        for task in collaborative_tasks:
            # Check if at least two robots are operational
            operational_robots = [r for r in robots if r.status == 'operational']
            if len(operational_robots) >= 2:
                # For simplicity, assign to the first operational robot
                # In a real system, this would require coordination
                task.assigned_to = operational_robots[0].id
                assignments[task.id] = operational_robots[0].id
                messages_sent += 1
        
        return assignments, messages_sent
    
    def run_recovery_auction(self, operational_robots, failed_tasks, task_graph):
        """Special auction for task reallocation after failure
        
        Args:
            operational_robots: List of operational Robot objects
            failed_tasks: List of Task objects that need reallocation
            task_graph: TaskDependencyGraph object
            
        Returns:
            tuple: (assignments dict, message count)
        """
        assignments = {}
        messages_sent = 0
        
        # Use GPU acceleration for recovery when possible
        if self.use_gpu and len(operational_robots) > 0 and len(failed_tasks) > 0:
            # Prepare data structures for GPU processing
            robot_positions = np.array([robot.position for robot in operational_robots])
            robot_capabilities = np.array([robot.capabilities for robot in operational_robots])
            robot_statuses = ['operational'] * len(operational_robots)
            
            task_positions = np.array([task.position for task in failed_tasks])
            task_capabilities = np.array([task.capabilities for task in failed_tasks])
            
            # Current assignments and prices
            task_assignments = np.zeros(len(failed_tasks), dtype=np.int32)
            prices = np.zeros(len(failed_tasks), dtype=np.float32)
            
            # Run GPU auction with parameters adjusted for recovery
            new_assignments, _, messages = self.gpu.run_auction_gpu(
                robot_positions, robot_capabilities, robot_statuses,
                task_positions, task_capabilities, task_assignments,
                self.epsilon * 2,  # Higher epsilon for faster convergence in recovery
                prices
            )
            
            # Process results
            for i, task in enumerate(failed_tasks):
                robot_idx = new_assignments[i]
                if robot_idx > 0:  # If assigned
                    robot_id = operational_robots[robot_idx-1].id
                    task.assigned_to = robot_id
                    assignments[task.id] = robot_id
            
            return assignments, messages
        
        # Fall back to CPU implementation
        for task in failed_tasks:
            best_bid = float('-inf')
            best_robot = None
            
            for robot in operational_robots:
                # Calculate standard bid
                standard_bid = robot.calculate_bid(task, self.weights, robot.workload)
                
                # Calculate task criticality (number of dependent tasks)
                criticality = task_graph.get_task_criticality(task.id)
                
                # Calculate urgency
                urgency = task.progress if task.status == 'in_progress' else 0
                
                # Calculate recovery bid
                recovery_bid = robot.calculate_recovery_bid(standard_bid, task.progress, 
                                                          criticality, urgency, 
                                                          self.beta_weights)
                
                # Apply communication delay and check for packet loss
                if self.communication_delay > 0:
                    time.sleep(self.communication_delay)
                
                if random.random() < self.packet_loss_prob:
                    continue  # Simulate packet loss
                
                messages_sent += 1
                
                if recovery_bid > best_bid:
                    best_bid = recovery_bid
                    best_robot = robot
            
            if best_robot:
                task.assigned_to = best_robot.id
                assignments[task.id] = best_robot.id
                messages_sent += 1
        
        return assignments, messages_sent