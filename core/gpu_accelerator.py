# core/gpu_accelerator.py
import torch
import numpy as np
import os

class GPUAccelerator:
    """GPU acceleration with support for AMD GPU detected through CUDA compatibility"""
    
    def __init__(self):
        """Initialize GPU accelerator for the unique configuration detected"""
        self.using_gpu = False
        self.device = torch.device('cpu')
        self.is_amd_via_cuda = False
        
        try:
            if torch.cuda.is_available():
                # Check if this is actually an AMD GPU detected through CUDA
                device_name = torch.cuda.get_device_name(0).lower()
                if 'amd' in device_name or 'radeon' in device_name:
                    self.is_amd_via_cuda = True
                    print(f"AMD GPU detected through CUDA: {torch.cuda.get_device_name(0)}")
                else:
                    print(f"NVIDIA GPU detected: {torch.cuda.get_device_name(0)}")
                
                self.device = torch.device('cuda')
                self.using_gpu = True
                
                # Set specific settings for AMD GPUs detected through CUDA
                if self.is_amd_via_cuda:
                    # Limit memory usage to avoid crashes
                    torch.cuda.set_per_process_memory_fraction(0.7)
            else:
                print("No GPU detected, using CPU")
                # Optimize CPU settings
                torch.set_num_threads(torch.get_num_threads())
                print(f"Using {torch.get_num_threads()} CPU threads")
                
        except Exception as e:
            print(f"Error during GPU initialization: {e}")
            print("Falling back to CPU")
            self.device = torch.device('cpu')
            self.using_gpu = False
        
        # Pre-allocate tensors for common operations
        self.eye4 = torch.eye(4, device=self.device)
        
        # Verify GPU is working
        if self.using_gpu:
            try:
                # Simple test tensor operation
                test_tensor = torch.ones(10, device=self.device)
                test_result = test_tensor + 1
                # Check for numeric results (avoiding NaNs)
                if torch.isnan(test_result).any():
                    raise Exception("GPU produced NaN values")
                print("GPU test successful")
            except Exception as e:
                print(f"GPU test failed: {e}")
                self.device = torch.device('cpu')
                self.using_gpu = False
                print("Falling back to CPU")
    
    def to_tensor(self, array):
        """Convert numpy array to tensor on correct device"""
        if isinstance(array, torch.Tensor):
            return array.to(self.device)
        return torch.tensor(array, dtype=torch.float32, device=self.device)
    
    def to_numpy(self, tensor):
        """Convert tensor to numpy array"""
        if isinstance(tensor, np.ndarray):
            return tensor
        return tensor.cpu().numpy()
        
    def calculate_bids_batch(self, robot_positions, robot_capabilities, task_positions, 
                          task_capabilities, workloads, weights):
        """Calculate bids for all robot-task combinations in a single batch operation
        
        Args:
            robot_positions: Tensor of robot positions [num_robots, 2]
            robot_capabilities: Tensor of robot capabilities [num_robots, cap_dim]
            task_positions: Tensor of task positions [num_tasks, 2]
            task_capabilities: Tensor of task capabilities [num_tasks, cap_dim]
            workloads: Tensor of robot workloads [num_robots]
            weights: Dictionary of weight parameters
            
        Returns:
            Tensor of bids [num_robots, num_tasks]
        """
        # Convert inputs to tensors
        r_pos = self.to_tensor(robot_positions)
        r_cap = self.to_tensor(robot_capabilities)
        t_pos = self.to_tensor(task_positions)
        t_cap = self.to_tensor(task_capabilities)
        work = self.to_tensor(workloads)
        
        # Extract weights
        alpha1 = weights['alpha1']
        alpha2 = weights['alpha2']
        alpha3 = weights['alpha3']
        alpha4 = weights['alpha4']
        alpha5 = weights.get('alpha5', 0.1)
        
        # Get dimensions
        num_robots = r_pos.shape[0]
        num_tasks = t_pos.shape[0]
        
        # Special handling for AMD GPUs via CUDA to prevent instability
        if self.is_amd_via_cuda and (num_robots > 10 or num_tasks > 50):
            # Break calculation into smaller chunks to avoid memory issues
            # This is a simple approach - we could optimize this further if needed
            chunk_size = 10
            bids = torch.zeros((num_robots, num_tasks), device=self.device)
            
            for i in range(0, num_robots, chunk_size):
                end_i = min(i + chunk_size, num_robots)
                for j in range(0, num_tasks, chunk_size):
                    end_j = min(j + chunk_size, num_tasks)
                    
                    # Process this chunk
                    r_pos_chunk = r_pos[i:end_i]
                    r_cap_chunk = r_cap[i:end_i]
                    t_pos_chunk = t_pos[j:end_j]
                    t_cap_chunk = t_cap[j:end_j]
                    work_chunk = work[i:end_i]
                    
                    # Calculate distances for this chunk
                    r_pos_expanded = r_pos_chunk.unsqueeze(1)
                    t_pos_expanded = t_pos_chunk.unsqueeze(0)
                    distances = torch.sqrt(torch.sum((r_pos_expanded - t_pos_expanded)**2, dim=2) + 1e-10)
                    
                    # Calculate terms
                    distance_term = alpha1 / (distances + 1e-10)
                    config_term = alpha2 * torch.ones((end_i-i, end_j-j), device=self.device)
                    
                    # Calculate capability similarity
                    r_cap_norm = torch.norm(r_cap_chunk, dim=1, keepdim=True)
                    t_cap_norm = torch.norm(t_cap_chunk, dim=1, keepdim=True)
                    r_cap_expanded = r_cap_chunk.unsqueeze(1)
                    t_cap_expanded = t_cap_chunk.unsqueeze(0)
                    dot_products = torch.sum(r_cap_expanded * t_cap_expanded, dim=2)
                    norms_product = r_cap_norm * t_cap_norm.t()
                    similarity = dot_products / (norms_product + 1e-10)
                    capability_term = alpha3 * similarity
                    
                    # Calculate workload term
                    workload_term = alpha4 * work_chunk.unsqueeze(1).expand(-1, end_j-j)
                    
                    # Calculate chunk bids
                    chunk_bids = distance_term + config_term + capability_term - workload_term
                    bids[i:end_i, j:end_j] = chunk_bids
                    
                    # Free up memory
                    torch.cuda.empty_cache()
            
            return bids
        else:
            # Standard calculation for CPU or stable GPU operations
            # Calculate distances (batch operation)
            r_pos_expanded = r_pos.unsqueeze(1)
            t_pos_expanded = t_pos.unsqueeze(0)
            distances = torch.sqrt(torch.sum((r_pos_expanded - t_pos_expanded)**2, dim=2) + 1e-10)
            
            # Calculate distance term
            distance_term = alpha1 / (distances + 1e-10)
            
            # For simplification, we'll set configuration cost term as constant
            config_term = alpha2 * torch.ones((num_robots, num_tasks), device=self.device)
            
            # Calculate capability similarity
            r_cap_norm = torch.norm(r_cap, dim=1, keepdim=True)
            t_cap_norm = torch.norm(t_cap, dim=1, keepdim=True)
            r_cap_expanded = r_cap.unsqueeze(1)  # [num_robots, 1, cap_dim]
            t_cap_expanded = t_cap.unsqueeze(0)  # [1, num_tasks, cap_dim]
            dot_products = torch.sum(r_cap_expanded * t_cap_expanded, dim=2)
            norms_product = r_cap_norm * t_cap_norm.t()
            similarity = dot_products / (norms_product + 1e-10)
            capability_term = alpha3 * similarity
            
            # Calculate workload term
            workload_term = alpha4 * work.unsqueeze(1).expand(-1, num_tasks)
            
            # Calculate final bids
            bids = distance_term + config_term + capability_term - workload_term
            
            return bids
        
    def run_auction_gpu(self, robot_positions, robot_capabilities, robot_statuses, 
                    task_positions, task_capabilities, task_assignments, 
                    epsilon, prices):
        """Run distributed auction algorithm on GPU
        
        Args:
            robot_positions: Tensor of robot positions [num_robots, 2]
            robot_capabilities: Tensor of robot capabilities [num_robots, cap_dim]
            robot_statuses: List of robot status strings ['operational', 'failed', etc.]
            task_positions: Tensor of task positions [num_tasks, 2]
            task_capabilities: Tensor of task capabilities [num_tasks, cap_dim]
            task_assignments: Tensor of current task assignments [num_tasks]
            epsilon: Minimum bid increment
            prices: Tensor of current prices [num_tasks]
            
        Returns:
            tuple: (new_assignments, new_prices, message_count)
        """
        # Default weights (can be passed as parameter)
        weights = {
            'alpha1': 0.3,
            'alpha2': 0.2,
            'alpha3': 0.3,
            'alpha4': 0.1,
            'alpha5': 0.1
        }
        
        # Convert inputs to tensors
        r_pos = self.to_tensor(robot_positions)
        r_cap = self.to_tensor(robot_capabilities)
        t_pos = self.to_tensor(task_positions)
        t_cap = self.to_tensor(task_capabilities)
        t_assign = self.to_tensor(task_assignments).long()
        prices_tensor = self.to_tensor(prices)
        
        # Create mask for operational robots
        op_robots = torch.tensor([i for i, status in enumerate(robot_statuses) 
                                if status == 'operational'], device=self.device)
        
        if len(op_robots) == 0:
            # No operational robots
            return t_assign.cpu().numpy(), prices_tensor.cpu().numpy(), 0
        
        # Get dimensions
        num_tasks = t_pos.shape[0]
        
        # Calculate workloads of robots
        workloads = torch.zeros(len(robot_statuses), device=self.device)
        
        # Set max_iterations based on theoretical bound: K² · bₘₐₓ/ε 
        # where K is task count and bₘₐₓ is maximum possible bid
        K = num_tasks
        b_max = 100.0  # Estimate of maximum possible bid value
        theoretical_max_iter = int(K**2 * b_max / epsilon)
        
        # Set a practical upper limit to prevent excessive iterations
        max_iterations = min(theoretical_max_iter, 1000)
        
        # Initialize message counter and iteration counter
        messages = 0
        iterations_used = 0
        
        for iteration in range(max_iterations):
            iterations_used += 1
            
            # Find unassigned tasks
            unassigned = (t_assign == 0).nonzero().flatten()
            
            if len(unassigned) == 0:
                break  # All tasks assigned
            
            # Track tasks assigned in this iteration
            tasks_assigned_this_iter = 0
            
            # For each operational robot
            for r_idx in op_robots:
                # Calculate bids for all unassigned tasks
                task_indices = unassigned.cpu().numpy()
                
                if len(task_indices) == 0:
                    continue
                
                # Select only unassigned tasks
                t_pos_unassigned = t_pos[task_indices]
                t_cap_unassigned = t_cap[task_indices]
                
                # Calculate bids for this robot with unassigned tasks
                bids = self.calculate_bids_batch(
                    r_pos[r_idx:r_idx+1], 
                    r_cap[r_idx:r_idx+1],
                    t_pos_unassigned,
                    t_cap_unassigned,
                    workloads[r_idx:r_idx+1],
                    weights
                )
                
                # Count messages - one message per unassigned task (bid request)
                messages += len(task_indices)
                
                # Calculate utilities (bid - price)
                prices_unassigned = prices_tensor[task_indices]
                utilities = bids[0] - prices_unassigned
                
                # Find best task for this robot
                if len(utilities) > 0:
                    best_idx = torch.argmax(utilities).item()
                    best_utility = utilities[best_idx].item()
                    
                    if best_utility > 0:
                        # Get original task index
                        task_idx = task_indices[best_idx]
                        
                        # Update price - critical for maintaining 2ε optimality gap
                        prices_tensor[task_idx] = prices_tensor[task_idx] + epsilon + best_utility
                        
                        # Assign task to robot
                        t_assign[task_idx] = r_idx + 1  # +1 because assignment 0 means unassigned
                        
                        # Update unassigned tasks
                        unassigned = (t_assign == 0).nonzero().flatten()
                        
                        # Message for assignment notification
                        messages += 1
                        
                        # Track assignments in this iteration
                        tasks_assigned_this_iter += 1
            
            # If no tasks were assigned in this iteration, break
            # This prevents unnecessary iterations and ensures convergence
            if tasks_assigned_this_iter == 0:
                break
        
        # For debugging/analysis - log if we hit iteration limit
        if iterations_used >= max_iterations and len(unassigned) > 0:
            import logging
            logging.warning(f"GPU Auction reached maximum iterations ({max_iterations}) "
                        f"with {len(unassigned)} tasks still unassigned.")
        
        return t_assign.cpu().numpy(), prices_tensor.cpu().numpy(), messages