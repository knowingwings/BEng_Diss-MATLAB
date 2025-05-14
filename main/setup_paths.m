% setup_paths.m
% Script to properly set up all paths for the distributed auction algorithm

% Get the current directory
current_dir = pwd;

% Check if we're in the main directory or elsewhere
[~, current_folder] = fileparts(current_dir);

if strcmp(current_folder, 'main')
    % We're in the main directory
    fprintf('Current directory: main\n');
    base_dir = fullfile(current_dir, '..');
else
    % Assume we're in the project root
    fprintf('Current directory: %s\n', current_folder);
    base_dir = current_dir;
end

% Add the common directory
common_dir = fullfile(base_dir, 'common');
if exist(common_dir, 'dir')
    fprintf('Adding common directory: %s\n', common_dir);
    addpath(common_dir);
else
    fprintf('WARNING: Common directory not found: %s\n', common_dir);
    
    % Try to find common directory
    if exist('./common', 'dir')
        fprintf('Found common directory in current folder.\n');
        addpath('./common');
    elseif exist('../common', 'dir')
        fprintf('Found common directory in parent folder.\n');
        addpath('../common');
    else
        fprintf('ERROR: Could not find common directory. Please ensure it exists.\n');
        fprintf('Create it manually if needed and place the utility files there.\n');
    end
end

% Create test directory if it doesn't exist
test_dir = fullfile(base_dir, 'test');
if ~exist(test_dir, 'dir')
    fprintf('Creating test directory: %s\n', test_dir);
    mkdir(test_dir);
end
addpath(test_dir);

% Create results directory if it doesn't exist
results_dir = fullfile(base_dir, 'results');
if ~exist(results_dir, 'dir')
    fprintf('Creating results directory: %s\n', results_dir);
    mkdir(results_dir);
end
addpath(results_dir);

% Create figures directory if it doesn't exist
figures_dir = fullfile(base_dir, 'figures');
if ~exist(figures_dir, 'dir')
    fprintf('Creating figures directory: %s\n', figures_dir);
    mkdir(figures_dir);
end
addpath(figures_dir);

% Check if utility functions exist
required_utils = {'environment_utils', 'robot_utils', 'task_utils', 'auction_utils'};
all_found = true;

for i = 1:length(required_utils)
    util_name = required_utils{i};
    util_path = which(util_name);
    
    if isempty(util_path)
        fprintf('WARNING: %s.m not found in path.\n', util_name);
        all_found = false;
    else
        fprintf('Found %s: %s\n', util_name, util_path);
    end
end

if ~all_found
    fprintf('\nSome utility files are missing. Please ensure all required files are in the common directory.\n');
else
    fprintf('\nAll required utility files found.\n');
end

% Try to load utils_manager to verify it works
try
    utils = utils_manager();
    fprintf('Successfully loaded utils_manager.\n');
catch ME
    fprintf('ERROR loading utils_manager: %s\n', ME.message);
end

fprintf('\nPath setup complete!\n');