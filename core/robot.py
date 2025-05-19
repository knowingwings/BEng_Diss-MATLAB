# decentralized_control/core/robot.py

import numpy as np

class Robot:
    def __init__(self, robot_id, position, orientation, config=None, capabilities=None):
        """Initialize a robot with given parameters
        
        Args:
            robot_id: Unique identifier for the robot
            position: Initial position as [x, y] array
            orientation: Initial orientation in radians
            config: Initial joint configuration (6-DOF manipulator)
            capabilities: Capability vector describing robot abilities
        """
        self.id = robot_id
        self.position = np.array(position, dtype=float)
        self.orientation = float(orientation)
        self.config = np.zeros(6) if config is None else np.array(config)
        self.capabilities = np.random.rand(5) if capabilities is None else np.array(capabilities)
        self.status = 'operational'  # 'operational', 'failed', 'partial_failure'
        self.workload = 0
        self.assigned_tasks = []
        self.trajectory = [np.copy(position)]  # Store trajectory for visualization
        
        # Kinematics parameters
        self.max_linear_velocity = 0.5  # m/s
        self.max_angular_velocity = 1.0  # rad/s
        
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
        
        where:
        - d_ij: Euclidean distance to task
        - c_ij: Configuration transition cost
        - s_ij: Capability similarity
        - l_i: Current workload
        - e_ij: Estimated energy consumption
        """
        # Extract weights
        alpha1 = weights['alpha1']
        alpha2 = weights['alpha2']
        alpha3 = weights['alpha3']
        alpha4 = weights['alpha4']
        alpha5 = weights['alpha5']
        W = weights['W']
        
        # Calculate distance to task
        distance = np.linalg.norm(self.position - task.position)
        
        # Calculate configuration transition cost
        config_diff = self.config - task.required_config
        config_cost = np.sqrt(config_diff.T @ W @ config_diff)
        
        # Calculate capability similarity
        robot_cap_norm = np.linalg.norm(self.capabilities)
        task_cap_norm = np.linalg.norm(task.capabilities)
        
        if robot_cap_norm > 0 and task_cap_norm > 0:
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
        
        where:
        - b_ij: Standard bid
        - p_j: Task progress
        - criticality(j): Number of dependent tasks
        - urgency(j): Time urgency factor
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