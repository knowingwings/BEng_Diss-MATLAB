function utils = kinematics_utils()
    % KINEMATICS_UTILS - Returns function handles for robot kinematics
    utils = struct(...
        'computeForwardKinematics', @computeForwardKinematics, ...
        'computeInverseKinematics', @computeInverseKinematics, ...
        'getTransformMatrix', @getTransformMatrix, ...
        'getDHParameters', @getDHParameters, ...
        'createJacobian', @createJacobian, ...
        'computeBaseKinematics', @computeBaseKinematics, ...
        'simulateJointMotion', @simulateJointMotion ...
    );
    
    % DH Parameters for OpenMANIPULATOR-X (4-DOF + gripper)
    % [a, alpha, d, theta] - Modified DH convention
    function params = getDHParameters()
        params = [...
            % a(m),  alpha(rad), d(m),    theta(rad)
            0,      pi/2,       0.077,    0;  % Joint 1
            0.130,  0,          0,        0;  % Joint 2
            0.124,  0,          0,        0;  % Joint 3
            0.126,  0,          0,        0;  % Joint 4
        ];
    end
    
    function T = getTransformMatrix(a, alpha, d, theta)
        % Create transformation matrix from DH parameters
        T = [cos(theta), -sin(theta)*cos(alpha), sin(theta)*sin(alpha),  a*cos(theta);
             sin(theta), cos(theta)*cos(alpha),  -cos(theta)*sin(alpha), a*sin(theta);
             0,          sin(alpha),             cos(alpha),             d;
             0,          0,                      0,                      1];
    end
    
    function [pose, T_matrices] = computeForwardKinematics(joint_angles, base_pose)
        % Compute forward kinematics for the manipulator
        % joint_angles: 4x1 vector of joint angles [rad]
        % base_pose: [x, y, z, roll, pitch, yaw] of robot base
        
        % Get DH parameters
        dh_params = getDHParameters();
        
        % Initialize transformation matrices
        T_matrices = cell(size(dh_params, 1) + 1, 1);
        
        % Base transform (from world to robot base)
        T_base = getBaseTransform(base_pose);
        T_matrices{1} = T_base;
        
        % Current transformation
        T_current = T_base;
        
        % Compute forward kinematics using DH parameters
        for i = 1:size(dh_params, 1)
            a = dh_params(i, 1);
            alpha = dh_params(i, 2);
            d = dh_params(i, 3);
            theta = joint_angles(i);
            
            % Calculate transform for this joint
            T_i = getTransformMatrix(a, alpha, d, theta);
            
            % Update current transform
            T_current = T_current * T_i;
            T_matrices{i+1} = T_current;
        end
        
        % Extract pose [x, y, z, roll, pitch, yaw] from final transform
        pos = T_current(1:3, 4);
        rot = rotm2eul(T_current(1:3, 1:3), 'ZYX');
        
        pose = [pos', rot];
    end
    
    function T = getBaseTransform(base_pose)
        % Create transformation matrix for the robot base
        % base_pose: [x, y, z, roll, pitch, yaw]
        x = base_pose(1);
        y = base_pose(2);
        z = base_pose(3);
        roll = base_pose(4);
        pitch = base_pose(5);
        yaw = base_pose(6);
        
        % Rotation matrices
        R_x = [1, 0, 0; 0, cos(roll), -sin(roll); 0, sin(roll), cos(roll)];
        R_y = [cos(pitch), 0, sin(pitch); 0, 1, 0; -sin(pitch), 0, cos(pitch)];
        R_z = [cos(yaw), -sin(yaw), 0; sin(yaw), cos(yaw), 0; 0, 0, 1];
        
        % Combined rotation matrix
        R = R_z * R_y * R_x;
        
        % Transformation matrix
        T = [R, [x; y; z]; 0, 0, 0, 1];
    end
    
    function joint_angles = computeInverseKinematics(target_pose, base_pose)
        % Compute inverse kinematics for the manipulator
        % target_pose: [x, y, z, roll, pitch, yaw] of end-effector
        % base_pose: [x, y, z, roll, pitch, yaw] of robot base
        
        % Get DH parameters
        dh_params = getDHParameters();
        
        % Transform target pose to robot base frame
        T_base_inv = inv(getBaseTransform(base_pose));
        T_target_world = getBaseTransform(target_pose);
        T_target_base = T_base_inv * T_target_world;
        
        % Extract position from transform
        x = T_target_base(1, 4);
        y = T_target_base(2, 4);
        z = T_target_base(3, 4);
        
        % Extract DH parameters
        a2 = dh_params(2, 1);
        a3 = dh_params(3, 1);
        a4 = dh_params(4, 1);
        d1 = dh_params(1, 3);
        
        % Calculate joint 1 (base rotation)
        theta1 = atan2(y, x);
        
        % Calculate distance to wrist center
        r = sqrt(x^2 + y^2);
        s = z - d1;
        
        % Simplified IK for this 4-DOF arm
        % Distance to wrist center
        D = sqrt(r^2 + s^2);
        
        % Joint 3 calculation using law of cosines
        cos_theta3 = (D^2 - a2^2 - a3^2) / (2 * a2 * a3);
        % Ensure within valid range
        cos_theta3 = min(max(cos_theta3, -1), 1);
        theta3 = acos(cos_theta3);
        
        % Joint 2 calculation
        theta2 = atan2(s, r) - atan2(a3 * sin(theta3), a2 + a3 * cos(theta3));
        
        % Joint 4 calculation (wrist orientation)
        % Extract desired orientation from target pose
        [~, pitch, ~] = quat2angle(eul2quat(target_pose(4:6), 'ZYX'), 'ZYX');
        theta4 = pitch - (theta2 + theta3);
        
        % Return joint angles
        joint_angles = [theta1; theta2; theta3; theta4];
        
        % Ensure joint angles are within limits
        joint_limits = [-pi/2, pi/2; -pi/2, pi/2; -pi/2, pi/2; -pi/2, pi/2];
        for i = 1:4
            joint_angles(i) = min(max(joint_angles(i), joint_limits(i,1)), joint_limits(i,2));
        end
    end
    
    function J = createJacobian(joint_angles, base_pose)
        % Create Jacobian matrix for velocity control
        % Get transformation matrices
        [~, T_matrices] = computeForwardKinematics(joint_angles, base_pose);
        
        % Calculate Jacobian columns
        J = zeros(6, 4); % 6 DOF x 4 joints
        
        % For each joint
        for i = 1:4
            % Position of the current joint
            if i == 1
                p_i = T_matrices{i}(1:3, 4);
            else
                p_i = T_matrices{i}(1:3, 4);
            end
            
            % Position of the end effector
            p_e = T_matrices{end}(1:3, 4);
            
            % Rotation axis of the joint
            z_i = T_matrices{i}(1:3, 3);
            
            % Linear velocity component
            J_v = cross(z_i, p_e - p_i);
            
            % Angular velocity component
            J_w = z_i;
            
            % Fill Jacobian column
            J(:, i) = [J_v; J_w];
        end
    end
    
    function [x_new, y_new, theta_new] = computeBaseKinematics(x, y, theta, v_l, v_r, dt)
        % Compute differential drive kinematics for the TurtleBot3 base
        % x, y, theta: Current pose
        % v_l, v_r: Left and right wheel velocities [m/s]
        % dt: Time step [s]
        
        % TurtleBot3 parameters
        wheel_radius = 0.033; % m
        wheel_separation = 0.287; % m
        
        % Convert wheel velocities to linear and angular velocities
        v = (v_r + v_l) / 2;
        omega = (v_r - v_l) / wheel_separation;
        
        % Update pose using differential drive kinematics
        if abs(omega) < 1e-6
            % Straight line motion
            x_new = x + v * cos(theta) * dt;
            y_new = y + v * sin(theta) * dt;
            theta_new = theta;
        else
            % Circular motion
            R = v / omega;
            x_new = x + R * (sin(theta + omega * dt) - sin(theta));
            y_new = y + R * (cos(theta) - cos(theta + omega * dt));
            theta_new = theta + omega * dt;
            
            % Normalize angle to [-pi, pi]
            theta_new = mod(theta_new + pi, 2*pi) - pi;
        end
    end
    
    function [joint_pos, link_pos] = simulateJointMotion(joint_angles, base_pose)
        % Simulate joint motion and return positions for visualization
        % joint_angles: Joint angles [rad]
        % base_pose: Base pose [x, y, z, roll, pitch, yaw]
        
        % Get transformation matrices
        [~, T_matrices] = computeForwardKinematics(joint_angles, base_pose);
        
        % Extract joint positions
        joint_pos = zeros(length(T_matrices), 3);
        for i = 1:length(T_matrices)
            joint_pos(i, :) = T_matrices{i}(1:3, 4)';
        end
        
        % Generate intermediate points along links for visualization
        n_points = 10; % Number of points per link
        link_pos = cell(length(T_matrices)-1, 1);
        
        for i = 1:length(T_matrices)-1
            start_pos = joint_pos(i, :);
            end_pos = joint_pos(i+1, :);
            
            % Interpolate positions
            link_pos{i} = zeros(n_points, 3);
            for j = 1:n_points
                t = (j-1)/(n_points-1);
                link_pos{i}(j, :) = (1-t) * start_pos + t * end_pos;
            end
        end
    end
end