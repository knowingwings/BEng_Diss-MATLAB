% Test script to check individual utilities
clear all;
close all;
clc;

% Add path with absolute path
current_dir = pwd;
addpath(fullfile(current_dir, '..', 'common'));

% Test environment_utils directly
disp('Testing environment_utils directly:');
env_utils = environment_utils();
disp('Fields in env_utils:');
disp(fieldnames(env_utils));

% Test each utility separately
try
    environment_utils();
    disp('environment_utils loaded successfully');
catch ME
    disp(['Error loading environment_utils: ' ME.message]);
end

try
    robot_utils();
    disp('robot_utils loaded successfully');
catch ME
    disp(['Error loading robot_utils: ' ME.message]);
end

try
    task_utils();
    disp('task_utils loaded successfully');
catch ME
    disp(['Error loading task_utils: ' ME.message]);
end

try
    auction_utils();
    disp('auction_utils loaded successfully');
catch ME
    disp(['Error loading auction_utils: ' ME.message]);
end