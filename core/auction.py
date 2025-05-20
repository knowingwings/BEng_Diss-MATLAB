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
                self.gpu = GPUAccelerator(communication_delay, packet_loss_prob)
                self.use_gpu = self.gpu.using_gpu
            except Exception as e:
                print(f"Could not initialize GPU acceleration: {e}")
                self.use_gpu = False
                
        # Add this line to maintain task prices between auctions
        self.task_prices = {}  # Dictionary to store task prices
    
    def run_auction(self, robots, tasks, task_graph):
        """Run distributed auction algorithm for task allocation
        
        Args:
            robots: List of Robot objects
            tasks: List of Task objects
            task_graph: TaskDependencyGraph object
            
        Returns:
            tuple: (assignments dict, message count)
        """
        # Print debug information about parameters
        print(f"Running auction with epsilon={self.epsilon}, delay={self.communication_delay*1000}ms, packet_loss={self.packet_loss_prob}")
        
        # Find unassigned tasks that are ready (prerequisites completed)
        unassigned_tasks = [task for task in tasks 
                        if task.assigned_to == 0 and 
                        task.status == 'pending' and 
                        task_graph.is_available(task.id)]
        
        if not unassigned_tasks:
            return {}, 0  # No tasks to assign
        
        # Initialize prices for new tasks
        for task in unassigned_tasks:
            if task.id not in self.task_prices:
                self.task_prices[task.id] = 0.0
        
        # Get current prices for tasks in this auction
        prices = {task.id: self.task_prices.get(task.id, 0.0) for task in tasks}
            
        # Use GPU implementation if enabled and possible
        if self.use_gpu and len(robots) > 0 and len(unassigned_tasks) > 0:
            # Update GPU accelerator communication parameters
            self.gpu.set_communication_params(self.communication_delay, self.packet_loss_prob)
            assignments, new_prices, messages = self._run_auction_gpu(robots, unassigned_tasks, tasks)
            
            # Update stored prices
            for task_id, price in new_prices.items():
                self.task_prices[task_id] = price
            
            return assignments, messages
        else:
            assignments, messages = self._run_auction_cpu(robots, unassigned_tasks, tasks)
            
            # Update stored prices from CPU auction
            for task_id, price in prices.items():
                self.task_prices[task_id] = price
            
            return assignments, messages
        
    def _run_auction_gpu(self, robots, unassigned_tasks, all_tasks):
        """GPU-accelerated auction implementation"""
        # Debug logging to track epsilon's effect
        print(f"DEBUG: Using epsilon={self.epsilon} in GPU auction")
        
        # Prepare data for GPU processing
        robot_positions = np.array([robot.position for robot in robots])
        robot_capabilities = np.array([robot.capabilities for robot in robots])
        robot_statuses = [robot.status for robot in robots]
        
        task_positions = np.array([task.position for task in unassigned_tasks])
        task_capabilities = np.array([task.capabilities for task in unassigned_tasks])
        
        # Map unassigned tasks to their indices in all_tasks
        task_id_to_idx = {task.id: i for i, task in enumerate(unassigned_tasks)}
        
        # Retrieve current prices for tasks in this auction
        current_prices = np.array([self.task_prices.get(task.id, 0.0) for task in unassigned_tasks], dtype=np.float32)
        
        # Current assignments and prices
        task_assignments = np.zeros(len(unassigned_tasks), dtype=np.int32)
        
        # Create weights dictionary with epsilon explicitly included
        weights = self.weights.copy()
        weights['epsilon'] = self.epsilon  # Critical: ensure epsilon is passed to GPU
        
        # Run GPU auction
        new_assignments, new_prices, messages = self.gpu.run_auction_gpu(
            robot_positions, robot_capabilities, robot_statuses,
            task_positions, task_capabilities, task_assignments,
            self.epsilon, current_prices,  # Use current prices instead of zeros
            weights=weights  # Pass weights including epsilon
        )
        
        # Convert results back to dictionary format
        assignments = {}
        price_updates = {}
        
        for i, task in enumerate(unassigned_tasks):
            robot_idx = new_assignments[i]
            if robot_idx > 0:  # If assigned
                robot_id = robots[robot_idx-1].id
                task.assigned_to = robot_id
                assignments[task.id] = robot_id
            
            # Update task prices in the class dictionary
            price_updates[task.id] = new_prices[i]
        
        return assignments, price_updates, messages
        
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
                    # Apply communication delay for assignment message
                    if self.communication_delay > 0:
                        time.sleep(self.communication_delay)
                    
                    # Check for packet loss for assignment message  
                    if random.random() < self.packet_loss_prob:
                        continue  # Packet loss on assignment message
                        
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
        # Print debug info
        print(f"Running recovery auction for {len(failed_tasks)} tasks with {len(operational_robots)} robots")
        
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
            
            # Update GPU accelerator communication parameters
            self.gpu.set_communication_params(self.communication_delay, self.packet_loss_prob)
            
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
                # Apply communication delay for bid calculation
                if self.communication_delay > 0:
                    time.sleep(self.communication_delay)
                
                # Check for packet loss
                if random.random() < self.packet_loss_prob:
                    continue  # Simulate packet loss
                    
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
                
                messages_sent += 1
                
                if recovery_bid > best_bid:
                    best_bid = recovery_bid
                    best_robot = robot
            
            if best_robot:
                # Apply communication delay for assignment
                if self.communication_delay > 0:
                    time.sleep(self.communication_delay)
                
                # Check for packet loss on assignment
                if random.random() < self.packet_loss_prob:
                    continue  # Packet loss on assignment message
                    
                task.assigned_to = best_robot.id
                assignments[task.id] = best_robot.id
                messages_sent += 1
        
        return assignments, messages_sent