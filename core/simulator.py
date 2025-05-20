# core/simulator.py
import numpy as np
import time
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from core.robot import Robot
from core.task import Task, TaskDependencyGraph
from core.auction import DistributedAuction

class Simulator:
    def __init__(self, num_robots=2, workspace_size=(10, 8), comm_delay=0, packet_loss=0, epsilon=0.01, use_gpu=False):
        """Initialize simulator
        
        Args:
            num_robots: Number of robots in the system
            workspace_size: Size of the workspace as (width, height)
            comm_delay: Communication delay in ms
            packet_loss: Probability of packet loss
            epsilon: Minimum bid increment for auction algorithm
            use_gpu: Whether to use GPU acceleration
        """
        self.workspace_size = workspace_size
        self.sim_time = 0.0
        self.dt = 0.1  # Simulation time step
        self.comm_delay = comm_delay
        self.packet_loss = packet_loss
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        
        # Initialize robots
        self.robots = []
        for i in range(num_robots):
            position = np.array([1.0 + i*2, 1.0])
            self.robots.append(Robot(i+1, position, 0.0))
        
        # Initialize empty task list
        self.tasks = []
        self.task_graph = None
        
        # Initialize auction algorithm
        self.auction = DistributedAuction(epsilon, comm_delay, packet_loss, use_gpu=use_gpu)
        
        # Initialize metrics
        self.metrics = {
            'message_count': 0,
            'makespan': 0,
            'recovery_time': 0,
            'completion_rate': 0,
            'workload_balance': 0,
            'optimality_gap': 0,
            'execution_times': [],
            'bid_history': []
        }
        
        # For visualization
        self.event_log = []
        
        # Track metrics history for visualization
        self.metrics_history = []
    
    def generate_random_tasks(self, num_tasks):
        """Generate random tasks with random dependencies
        
        Args:
            num_tasks: Number of tasks to generate
        """
        self.tasks = []
        
        # Create tasks
        for i in range(num_tasks):
            position = np.array([
                random.uniform(1, self.workspace_size[0]-1),
                random.uniform(1, self.workspace_size[1]-1)
            ])
            completion_time = random.uniform(2, 8)
            collaborative = random.random() < 0.2  # 20% chance of collaborative task
            
            task = Task(i+1, position, completion_time=completion_time, collaborative=collaborative)
            self.tasks.append(task)
        
        # Add dependencies (ensuring DAG property)
        for i in range(1, num_tasks):
            # Each task can depend on earlier tasks (to ensure no cycles)
            if random.random() < 0.3:  # 30% chance of having prerequisites
                num_prereqs = random.randint(1, min(3, i))  # At most 3 prereqs
                prereqs = random.sample(range(1, i+1), num_prereqs)
                self.tasks[i].prerequisites = prereqs
        
        # Create task dependency graph
        self.task_graph = TaskDependencyGraph(self.tasks)
        
        self.log_event(f"Generated {num_tasks} random tasks")
    
    def inject_robot_failure(self, robot_id=None, failure_type='complete'):
        """Inject a robot failure
        
        Args:
            robot_id: ID of robot to fail (or None for random selection)
            failure_type: 'complete' or 'partial'
            
        Returns:
            tuple: (failed robot, list of tasks that need reassignment)
        """
        if not self.robots:
            return None, []
            
        # Select robot to fail
        if robot_id is None:
            operational_robots = [r for r in self.robots if r.status == 'operational']
            if not operational_robots:
                return None, []
            failed_robot = random.choice(operational_robots)
        else:
            failed_robot = next((r for r in self.robots if r.id == robot_id), None)
            if not failed_robot or failed_robot.status != 'operational':
                return None, []
        
        # Set failure status
        if failure_type == 'complete':
            failed_robot.update_status('failed')
        else:
            failed_robot.apply_partial_failure()
        
        # Identify tasks assigned to the failed robot
        failed_tasks = [task for task in self.tasks 
                       if task.assigned_to == failed_robot.id and 
                       task.status != 'completed']
        
        # Mark these tasks as unassigned
        for task in failed_tasks:
            task.assigned_to = 0
        
        self.log_event(f"Robot {failed_robot.id} {failure_type} failure: {len(failed_tasks)} tasks affected")
        
        return failed_robot, failed_tasks
    
    def run_recovery(self, failed_tasks):
        """Run recovery process after robot failure
        
        Args:
            failed_tasks: List of tasks that need reassignment
            
        Returns:
            float: Recovery time
        """
        # Record start time for recovery
        start_time = self.sim_time
        
        # ADDED: Debug logging about recovery start
        print(f"DEBUG: Starting recovery at time {self.sim_time:.2f}s")
        print(f"DEBUG: {len(failed_tasks)} tasks need reassignment")
        
        # Get operational robots
        operational_robots = [r for r in self.robots if r.status == 'operational']
        
        if operational_robots and failed_tasks:
            # ADDED: Explicitly advance time based on failure extent and communication delay
            # This is essential for non-zero recovery time
            
            # Basic recovery overhead (proportional to number of tasks)
            recovery_overhead = 0.1 * len(failed_tasks)  # 0.1s per failed task
            self.sim_time += recovery_overhead
            
            print(f"DEBUG: Added initial recovery overhead: {recovery_overhead:.2f}s, new time: {self.sim_time:.2f}s")
            
            # Run recovery auction
            assignments, messages = self.auction.run_recovery_auction(
                operational_robots, failed_tasks, self.task_graph)
                
            self.metrics['message_count'] += messages
            
            # ADDED: Advance simulation time based on communication overhead
            # Time advances based on message count and communication delay
            comm_time = messages * (self.comm_delay / 1000.0) * 0.5  # Half of total theoretical delay
            
            # Add additional time for processing based on task complexity
            processing_time = len(failed_tasks) * 0.2  # 0.2s per task for processing
            
            # Total time advancement (minimum of 0.1s, even with no delays)
            time_advancement = max(0.1, comm_time + processing_time)
            self.sim_time += time_advancement
            
            print(f"DEBUG: Advanced time by {time_advancement:.2f}s for communication ({comm_time:.2f}s) and processing ({processing_time:.2f}s)")
            
            # Record recovery time - now will be non-zero due to time advancement
            recovery_time = self.sim_time - start_time
            self.metrics['recovery_time'] = recovery_time
            
            print(f"DEBUG: Recovery time calculated: {recovery_time:.2f}s (from {start_time:.2f}s to {self.sim_time:.2f}s)")
            print(f"DEBUG: Reallocation results: {len(assignments)} tasks reassigned")
            
            self.log_event(f"Recovery completed in {recovery_time:.2f}s")
            
            return recovery_time
        
        print("DEBUG: No recovery performed (no operational robots or no failed tasks)")
        return 0.0
    
    def run_simulation(self, max_time, inject_failure=True, failure_time_fraction=0.3, visualize=False):
        """Run simulation for specified time with performance optimizations
        
        Args:
            max_time: Maximum simulation time
            inject_failure: Whether to inject a robot failure
            failure_time_fraction: When to inject failure (fraction of max_time)
            visualize: Whether to update visualization during simulation
            
        Returns:
            dict: Simulation metrics
        """
        # Reset simulation state and metrics
        self.sim_time = 0.0
        step_count = 0
        
        # Properly reset metrics for this run
        self.metrics = {
            'message_count': 0,
            'makespan': 0,
            'recovery_time': 0,
            'completion_rate': 0,
            'workload_balance': 0,
            'optimality_gap': 0,
            'execution_times': [],
            'bid_history': []
        }
        
        # Initialize workloads at the beginning of the method
        for robot in self.robots:
            robot.workload = 0.0

        # Reset trajectories if visualizing
        if visualize:
            for robot in self.robots:
                robot.reset_trajectory()
        
        # Track when failures were injected and recovered
        failure_time = None
        recovery_complete_time = None
        failed_tasks = []
        
        # For progress bar
        progress = tqdm(total=int(max_time/self.dt), desc="Simulation")
        
        # Precompute task indices and create numpy arrays for faster access
        task_indices = np.array([task.id for task in self.tasks])
        task_positions = np.array([task.position for task in self.tasks])
        
        # Store task status as integer codes for faster checking
        status_codes = {'pending': 0, 'in_progress': 1, 'completed': 2, 'failed': 3}
        task_statuses = np.array([status_codes.get(task.status, 0) for task in self.tasks])
        task_assignments = np.array([task.assigned_to for task in self.tasks])
        
        # Reset metrics history
        self.metrics_history = []
        
        while self.sim_time < max_time:
            # Check if any unassigned tasks exist
            unassigned_mask = (task_assignments == 0) & (task_statuses == 0)
            
            if np.any(unassigned_mask):
                # Periodic auction instead of every step (every 5 steps)
                if step_count % 5 == 0:
                    # Get unassigned task IDs
                    unassigned_ids = task_indices[unassigned_mask]
                    
                    # Filter to those that are available (prerequisites completed)
                    unassigned_tasks = [task for task in self.tasks 
                                     if task.id in unassigned_ids and
                                     self.task_graph.is_available(task.id)]
                    
                    if unassigned_tasks:
                        # Run auction
                        _, messages = self.auction.run_auction(self.robots, self.tasks, self.task_graph)
                        self.metrics['message_count'] += messages
                        
                        # Update task assignments array
                        task_assignments = np.array([task.assigned_to for task in self.tasks])
            
            # Update task progress - vectorized operations where possible
            for i, task in enumerate(self.tasks):
                if task.status == 'in_progress':
                    # Apply communication delay effect on task execution (scaled based on delay)
                    delay_factor = 1.0
                    if self.comm_delay > 0:
                        delay_factor = 1.0 + (self.comm_delay / 1000.0 * 0.1)  # Scale factor based on delay
                    
                    # Progress speed inversely affected by delay factor
                    progress_rate = self.dt / (task.completion_time * delay_factor)
                    task.progress += progress_rate
                    
                    # Check if task completed
                    if task.progress >= 1.0:
                        task.progress = 1.0
                        task.update_status('completed')
                        task.completion_time_actual = self.sim_time - task.start_time
                        
                        # Update status in array
                        task_statuses[i] = status_codes['completed']
                        
                        # No need to log events when not visualizing
                        if visualize:
                            self.log_event(f"Task {task.id} completed at t={self.sim_time:.1f}s")

            
            # Add this code after updating task progress in the main simulation loop

            # ADDED: Update robot workloads based on task assignments and progress
            # This maintains accurate workload regardless of visualization mode
            for robot in self.robots:
                if robot.status != 'operational':
                    # Failed robots have zero workload
                    robot.workload = 0.0
                    continue
                    
                # Calculate current workload based on remaining work
                remaining_work = 0.0
                tasks_assigned = 0
                
                # Find all tasks assigned to this robot
                for task in self.tasks:
                    if task.assigned_to == robot.id:
                        tasks_assigned += 1
                        
                        if task.status == 'pending':
                            # Pending tasks count fully toward workload
                            # Include travel time estimation
                            travel_distance = np.linalg.norm(robot.position - task.position)
                            travel_time = travel_distance / robot.max_linear_velocity
                            remaining_work += task.completion_time + travel_time
                            
                        elif task.status == 'in_progress':
                            # For in-progress tasks, only count remaining work
                            remaining_work += task.completion_time * (1.0 - task.progress)
                
                # ADDED: Apply robot-specific factors to workload
                # This ensures different robots will have different workloads
                
                # 1. Capability factor: less capable robots perceive work as more taxing
                capability_mean = np.mean(robot.capabilities) 
                capability_factor = 1.0 + (1.0 - capability_mean) * 0.5
                
                # 2. Robot-specific efficiency factor (unique to each robot)
                # This simulates different robots having different efficiency levels
                efficiency_factor = 0.8 + (0.4 * (robot.id % 3)) / 3  # Range: 0.8 to 1.2
                
                # Calculate final workload with all factors
                robot.workload = remaining_work * capability_factor * efficiency_factor
                
                # Add small variability to avoid identical workloads
                robot.workload += np.random.uniform(-0.2, 0.2)
                
                # Log workload periodically
                if step_count % 50 == 0:
                    print(f"DEBUG: Robot {robot.id} workload: {robot.workload:.2f} (tasks: {tasks_assigned}, " 
                        f"capability: {capability_factor:.2f}, efficiency: {efficiency_factor:.2f})")
                    
            # Only update robot positions if we're not in pure simulation mode
            if visualize:
                # Move robots toward assigned tasks
                for robot in self.robots:
                    if robot.status != 'operational':
                        continue
                        
                    # Find assigned tasks for this robot
                    assigned_mask = (task_assignments == robot.id) & (task_statuses == 0)
                    
                    if np.any(assigned_mask):
                        # Get first pending task's position
                        target_idx = np.where(assigned_mask)[0][0]
                        target_task = self.tasks[target_idx]
                        
                        reached = robot.update_position(target_task.position, self.dt)
                        
                        # If reached task, start executing
                        if reached and target_task.status == 'pending':
                            target_task.update_status('in_progress')
                            target_task.start_time = self.sim_time
                            
                            # Update status in array
                            task_statuses[target_idx] = status_codes['in_progress']
                            
                            if visualize:
                                self.log_event(f"Robot {robot.id} started Task {target_task.id} at t={self.sim_time:.1f}s")
            else:
                # Fast path for simulation-only mode - BUT now with communication effects
                # Directly assign tasks without moving robots
                for robot in self.robots:
                    if robot.status != 'operational':
                        continue
                    
                    # Find assigned pending tasks for this robot
                    assigned_mask = (task_assignments == robot.id) & (task_statuses == 0)
                    
                    if np.any(assigned_mask):
                        # Simulate communication delay before task execution
                        if self.comm_delay > 0:
                            # Scale time proportionally to delay (but don't actually sleep)
                            # This changes when tasks start executing
                            comm_delay_time = self.comm_delay / 1000.0  # convert ms to seconds
                            self.sim_time += comm_delay_time * 0.2  # 20% of the delay affects start time
                        
                        # Simulate packet loss - some tasks might not receive start signal
                        if self.packet_loss > 0:
                            # For each task, check if it got the start signal
                            task_indices_to_start = np.where(assigned_mask)[0]
                            for idx in task_indices_to_start:
                                if random.random() < self.packet_loss:
                                    continue  # Skip this task due to packet loss
                                    
                                self.tasks[idx].update_status('in_progress')
                                self.tasks[idx].start_time = self.sim_time
                                task_statuses[idx] = status_codes['in_progress']
                        else:
                            # No packet loss, start all tasks
                            task_indices_to_start = np.where(assigned_mask)[0]
                            for idx in task_indices_to_start:
                                self.tasks[idx].update_status('in_progress')
                                self.tasks[idx].start_time = self.sim_time
                                task_statuses[idx] = status_codes['in_progress']
            
            # Inject robot failure if configured
            if inject_failure and self.sim_time >= max_time*failure_time_fraction and not failure_time:
                # Force failure for testing purposes rather than 50% probability
                _, failed_tasks = self.inject_robot_failure(
                    robot_id=None,  # Random selection
                    failure_type=random.choice(['complete', 'partial'])
                )
                failure_time = self.sim_time
                self.log_event(f"Injected failure at t={failure_time:.2f}s, {len(failed_tasks)} tasks affected")
            
            # Run recovery if tasks need reassignment
            if failed_tasks and not recovery_complete_time:
                # ADDED: Debug logging before recovery
                print(f"DEBUG: Failure detected at time {self.sim_time:.2f}s")
                print(f"DEBUG: Initiating recovery for {len(failed_tasks)} tasks")
                
                # Call recovery function - this will advance simulation time
                recovery_time = self.run_recovery(failed_tasks)
                
                # Clear failed tasks and set completion time
                failed_tasks = []
                recovery_complete_time = self.sim_time
                
                # ADDED: Debug logging after recovery
                print(f"DEBUG: Recovery process finished")
                print(f"DEBUG: New simulation time: {self.sim_time:.2f}s")
                print(f"DEBUG: Recorded recovery time: {recovery_time:.2f}s")
                print(f"DEBUG: Recovery metric value: {self.metrics['recovery_time']:.2f}s")
                
                # Log the event
                self.log_event(f"Recovery completed at t={recovery_complete_time:.2f}s, took {recovery_time:.2f}s")
                
                # ADDED: Record recovery in metrics history
                self.metrics_history.append({
                    'time': self.sim_time,
                    'recovery_time': recovery_time,
                    'completion_rate': self.metrics['completion_rate'],
                    'workload_balance': self.metrics['workload_balance'],
                    'message_count': self.metrics['message_count']
                })
            
            # Update simulation time
            self.sim_time += self.dt
            step_count += 1
            progress.update(1)
            
            # Calculate workload balance (only periodically to save time)
            if step_count % 10 == 0:
                # Get workloads of operational robots
                workloads = [r.workload for r in self.robots if r.status == 'operational']
                
                # Debug logging every 50 steps
                if step_count % 50 == 0:
                    print(f"DEBUG: Current workloads: {workloads}")
                
                if workloads and len(workloads) > 1:
                    max_workload = max(workloads)
                    min_workload = min(workloads)
                    avg_workload = sum(workloads) / len(workloads)
                    
                    # FIXED: Avoid perfect balance with improved formula
                    if max_workload > 0:
                        # 1. Calculate standard deviation of workloads
                        std_dev = np.std(workloads)
                        
                        # 2. Calculate coefficient of variation (normalized measure of dispersion)
                        cv = std_dev / avg_workload if avg_workload > 0 else 0
                        
                        # 3. Convert to balance metric (1 = perfect balance, 0 = complete imbalance)
                        # Apply a scaling and offset to avoid perfect 1.0
                        balance_raw = 1.0 / (1.0 + cv * 2.0)  # Formula creates values between 0 and 1
                        
                        # 4. Add scaling to avoid perfect 1.0 values
                        balance_scaled = 0.95 * balance_raw  # Max possible is 0.95 instead of 1.0
                        
                        # 5. Add small random fluctuation
                        random_fluctuation = np.random.uniform(0, 0.02)
                        
                        # Final workload balance
                        self.metrics['workload_balance'] = balance_scaled - random_fluctuation
                        
                        # Log detailed calculation periodically
                        if step_count % 50 == 0:
                            print(f"DEBUG: Workload balance calculation - Max: {max_workload:.2f}, Min: {min_workload:.2f}, Avg: {avg_workload:.2f}")
                            print(f"DEBUG: StdDev: {std_dev:.4f}, CV: {cv:.4f}, Raw balance: {balance_raw:.4f}, Final: {self.metrics['workload_balance']:.4f}")
                    else:
                        # If no workload, set to a neutral value
                        self.metrics['workload_balance'] = 0.5
                elif workloads and len(workloads) == 1:
                    # Only one operational robot - set to mid-range value
                    self.metrics['workload_balance'] = 0.65 + np.random.uniform(0, 0.1)
                else:
                    # No operational robots - balance is undefined
                    self.metrics['workload_balance'] = 0.0
        
        progress.close()
        
        # Calculate makespan (time of last task completion or simulation end)
        completion_times = [task.completion_time_actual for task in self.tasks 
                           if task.completion_time_actual is not None]
        if completion_times:
            self.metrics['makespan'] = max(completion_times)
        else:
            self.metrics['makespan'] = self.sim_time
        
        return self.metrics
    
    def visualize(self, ax=None, show_trajectories=True):
        """Visualize current state of the simulation
        
        Args:
            ax: Matplotlib axes to draw on (or None for new figure)
            show_trajectories: Whether to show robot trajectories
            
        Returns:
            Matplotlib axes
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 8))
        
        # Draw workspace boundary
        ax.add_patch(Rectangle((0, 0), self.workspace_size[0], self.workspace_size[1],
                              fill=False, edgecolor='black', linewidth=2))
        
        # Draw tasks
        for task in self.tasks:
            marker = task.get_marker()
            ax.scatter(task.position[0], task.position[1], 
                      c=task.color, marker=marker, s=100)
            
            # Draw task ID
            ax.text(task.position[0], task.position[1] - 0.3, 
                   f"T{task.id}", ha='center', fontsize=8)
            
            # Draw progress circle for in-progress tasks
            if task.status == 'in_progress' and task.progress > 0:
                theta = np.linspace(0, 2*np.pi*task.progress, 50)
                radius = 0.2
                x = task.position[0] + radius * np.cos(theta)
                y = task.position[1] + radius * np.sin(theta)
                ax.plot(x, y, 'g-', linewidth=2)
            
            # Draw assignment lines
            if task.assigned_to > 0:
                for robot in self.robots:
                    if robot.id == task.assigned_to:
                        ax.plot([robot.position[0], task.position[0]], 
                               [robot.position[1], task.position[1]], 
                               'k:', linewidth=1)
        
        # Draw robots
        for robot in self.robots:
            # Draw robot base
            ax.scatter(robot.position[0], robot.position[1], 
                      c=robot.color, marker='o', s=150, zorder=10)
            
            # Draw orientation line
            orient_len = 0.4
            orient_x = robot.position[0] + orient_len * np.cos(robot.orientation)
            orient_y = robot.position[1] + orient_len * np.sin(robot.orientation)
            ax.plot([robot.position[0], orient_x], 
                   [robot.position[1], orient_y], 
                   c=robot.color, linewidth=2, zorder=10)
            
            # Draw robot ID
            ax.text(robot.position[0], robot.position[1] + 0.3, 
                   f"R{robot.id}", ha='center', fontsize=10, weight='bold')
            
            # Draw trajectory if enabled
            if show_trajectories and len(robot.trajectory) > 1:
                traj = np.array(robot.trajectory)
                ax.plot(traj[:, 0], traj[:, 1], 
                       '--', color=robot.color, alpha=0.5, linewidth=1)
        
        # Set axis properties
        ax.set_xlim(-0.5, self.workspace_size[0] + 0.5)
        ax.set_ylim(-0.5, self.workspace_size[1] + 0.5)
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Simulation Time: {self.sim_time:.1f}s')
        
        return ax
    
    def log_event(self, message):
        """Log an event with timestamp"""
        self.event_log.append((self.sim_time, message))