# core/gpu_accelerator.py
import torch
import numpy as np

class GPUAccelerator:
    """GPU acceleration for compute-intensive operations"""
    
    def __init__(self):
        # Check if GPU is available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.using_gpu = self.device.type == 'cuda'
        
        # Pre-allocate tensors for common operations
        self.eye6 = torch.eye(6, device=self.device)
        
        if self.using_gpu:
            print(f"GPU acceleration enabled: {torch.cuda.get_device_name(0)}")
        else:
            print("GPU not available, using CPU")
    
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
        
        # Calculate distances (batch operation)
        # Reshape for broadcasting: [num_robots, 1, 2] - [1, num_tasks, 2]
        r_pos_expanded = r_pos.unsqueeze(1)
        t_pos_expanded = t_pos.unsqueeze(0)
        
        # Calculate Euclidean distances: [num_robots, num_tasks]
        distances = torch.sqrt(torch.sum((r_pos_expanded - t_pos_expanded)**2, dim=2) + 1e-10)
        
        # Calculate distance term (avoid division by zero with epsilon)
        distance_term = alpha1 / (distances + 1e-10)
        
        # For simplification, we'll set configuration cost term as constant
        # In a real implementation, you'd calculate this based on the configurations
        config_term = alpha2 * torch.ones((num_robots, num_tasks), device=self.device)
        
        # Calculate capability similarity
        # Normalize capabilities
        r_cap_norm = torch.norm(r_cap, dim=1, keepdim=True)
        t_cap_norm = torch.norm(t_cap, dim=1, keepdim=True)
        
        # Reshape for batch dot product
        r_cap_expanded = r_cap.unsqueeze(1)  # [num_robots, 1, cap_dim]
        t_cap_expanded = t_cap.unsqueeze(0)  # [1, num_tasks, cap_dim]
        
        # Calculate dot products
        dot_products = torch.sum(r_cap_expanded * t_cap_expanded, dim=2)
        
        # Calculate similarity (cosine similarity)
        norms_product = r_cap_norm * t_cap_norm.t()
        similarity = dot_products / (norms_product + 1e-10)
        
        # Calculate capability term
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
        
        # Run auction iterations
        max_iterations = 100
        messages = 0
        
        for _ in range(max_iterations):
            # Find unassigned tasks
            unassigned = (t_assign == 0).nonzero().flatten()
            
            if len(unassigned) == 0:
                break
            
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
                
                # Count messages
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
                        
                        # Update price
                        prices_tensor[task_idx] = prices_tensor[task_idx] + epsilon + bids[0, best_idx]
                        
                        # Assign task to robot
                        t_assign[task_idx] = r_idx + 1  # +1 because assignment 0 means unassigned
                        
                        # Update unassigned tasks
                        unassigned = (t_assign == 0).nonzero().flatten()
                        
                        # Message for assignment
                        messages += 1
            
            # If no tasks were assigned in this iteration, break
            if len(unassigned) == len(task_indices):
                break
        
        return t_assign.cpu().numpy(), prices_tensor.cpu().numpy(), messages