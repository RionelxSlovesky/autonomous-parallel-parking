% run_vehicle_motion_sanity_check.m
clear; clc; close all;

%% =========================
% Vehicle Motion Sanity Check
% =========================
% Purpose:
%   Verify the kinematic bicycle model and steering lag behavior before
%   rebuilding parking controllers or RL environments.
%
% What this script does:
%   Runs several simple motion tests:
%     1) forward straight
%     2) reverse straight
%     3) forward left turn
%     4) forward right turn
%     5) reverse left turn
%     6) reverse right turn
%
% Expected behavior:
%   - forward straight: moves along +x with no heading change
%   - reverse straight: moves backward along -x with no heading change
%   - forward left turn: curves left, heading increases
%   - forward right turn: curves right, heading decreases
%   - reverse left turn: backing up with left steer, heading decreases
%   - reverse right turn: backing up with right steer, heading increases

%% =========================
% Parameters
% =========================
Ts = 0.02;              % 50 Hz
NumSteps = 250;         % 5 seconds
Wheelbase = 0.256;

SteerMin = -0.5;
SteerMax =  0.5;
TauSteer = 0.16;        % steering lag time constant
Vmax = 0.35;

EgoLength = 0.32;
EgoWidth  = 0.18;

%% =========================
% Test definitions
% =========================
tests = {
    struct("name","Forward Straight", "steerCmd",  0.00, "vCmd",  0.18)
    struct("name","Reverse Straight", "steerCmd",  0.00, "vCmd", -0.18)
    struct("name","Forward Left Turn", "steerCmd",  0.35, "vCmd",  0.18)
    struct("name","Forward Right Turn","steerCmd", -0.35, "vCmd",  0.18)
    struct("name","Reverse Left Turn", "steerCmd",  0.35, "vCmd", -0.18)
    struct("name","Reverse Right Turn","steerCmd", -0.35, "vCmd", -0.18)
};

allResults = cell(numel(tests),1);

%% =========================
% Run tests
% =========================
for i = 1:numel(tests)
    test = tests{i};
    result = runSingleTest(test.name, test.steerCmd, test.vCmd, ...
        Ts, NumSteps, Wheelbase, TauSteer, SteerMin, SteerMax, Vmax);
    allResults{i} = result;
end

%% =========================
% Combined trajectory plot
% =========================
figure('Name','Vehicle Motion Sanity Check - All Trajectories');
hold on; grid on; axis equal;

for i = 1:numel(allResults)
    r = allResults{i};
    plot(r.X, r.Y, 'LineWidth', 1.8, 'DisplayName', r.Name);
    plot(r.X(1), r.Y(1), 'o', 'HandleVisibility','off');
    plot(r.X(end), r.Y(end), 's', 'HandleVisibility','off');
end

xlabel('X (m)');
ylabel('Y (m)');
title('Kinematic Bicycle Model Sanity Check');
legend('Location','best');

%% =========================
% Individual plots
% =========================
for i = 1:numel(allResults)
    r = allResults{i};

    figure('Name', ['Trajectory - ' r.Name]);
    hold on; grid on; axis equal;
    plot(r.X, r.Y, 'b-', 'LineWidth', 1.8);
    plot(r.X(1), r.Y(1), 'go', 'MarkerSize', 8, 'LineWidth', 1.5);
    plot(r.X(end), r.Y(end), 'ro', 'MarkerSize', 8, 'LineWidth', 1.5);

    % draw final vehicle pose
    [px, py] = carPolygon([r.X(end), r.Y(end)], r.PSI(end), EgoLength, EgoWidth);
    patch(px, py, [0.2 0.4 1.0], 'FaceAlpha', 0.35, 'EdgeColor', 'k');

    xlabel('X (m)');
    ylabel('Y (m)');
    title(['Trajectory - ' r.Name]);

    figure('Name', ['States - ' r.Name]);

    subplot(3,1,1);
    plot(r.PSI, 'LineWidth', 1.5); grid on;
    ylabel('\psi (rad)');
    title(r.Name);

    subplot(3,1,2);
    plot(r.V, 'LineWidth', 1.5); grid on;
    ylabel('v (m/s)');

    subplot(3,1,3);
    plot(r.DELTA, 'LineWidth', 1.5); grid on;
    ylabel('\delta (rad)');
    xlabel('Step');
end

%% =========================
% Simple animation for one test
% =========================
animateIdx = 3; % change this from 1..6 if you want
r = allResults{animateIdx};

figure('Name', ['Animation - ' r.Name]);
hold on; grid on; axis equal;
xlabel('X (m)');
ylabel('Y (m)');
title(['Animation - ' r.Name]);

margin = 0.4;
xlim([min(r.X)-margin, max(r.X)+margin]);
ylim([min(r.Y)-margin, max(r.Y)+margin]);

hTrail = plot(nan, nan, 'b-', 'LineWidth', 1.8);
hCar = patch(nan, nan, [0.2 0.4 1.0], 'FaceAlpha', 0.35, 'EdgeColor', 'k');

for k = 1:length(r.X)
    set(hTrail, 'XData', r.X(1:k), 'YData', r.Y(1:k));
    [px, py] = carPolygon([r.X(k), r.Y(k)], r.PSI(k), EgoLength, EgoWidth);
    set(hCar, 'XData', px, 'YData', py);
    drawnow;
    pause(0.02);
end

%% =========================
% Console summary
% =========================
fprintf('\n==== Vehicle Motion Sanity Check Summary ====\n');
for i = 1:numel(allResults)
    r = allResults{i};
    fprintf('%s\n', r.Name);
    fprintf('  Final x    : %.4f\n', r.X(end));
    fprintf('  Final y    : %.4f\n', r.Y(end));
    fprintf('  Final yaw  : %.4f rad\n', r.PSI(end));
    fprintf('  Final speed: %.4f m/s\n\n', r.V(end));
end

%% =========================
% Local functions
% =========================
function result = runSingleTest(name, steerCmdConst, vCmdConst, ...
    Ts, NumSteps, Wheelbase, TauSteer, SteerMin, SteerMax, Vmax)

    % Initial state
    x = 0.0;
    y = 0.0;
    psi = 0.0;
    v = 0.0;
    delta = 0.0;

    % Clamp commands
    steerCmdConst = min(max(steerCmdConst, SteerMin), SteerMax);
    vCmdConst = min(max(vCmdConst, -Vmax), Vmax);

    X = nan(NumSteps,1);
    Y = nan(NumSteps,1);
    PSI = nan(NumSteps,1);
    V = nan(NumSteps,1);
    DELTA = nan(NumSteps,1);

    for k = 1:NumSteps
        % Steering lag
        delta = delta + (Ts/TauSteer) * (steerCmdConst - delta);

        % Speed smoothing
        v = 0.85*v + 0.15*vCmdConst;

        % Kinematic bicycle update
        x = x + Ts * v * cos(psi);
        y = y + Ts * v * sin(psi);
        psi = wrapToPiLocal(psi + Ts * (v/Wheelbase) * tan(delta));

        X(k) = x;
        Y(k) = y;
        PSI(k) = psi;
        V(k) = v;
        DELTA(k) = delta;
    end

    result = struct();
    result.Name = name;
    result.X = X;
    result.Y = Y;
    result.PSI = PSI;
    result.V = V;
    result.DELTA = DELTA;
end

function [px, py] = carPolygon(center, yaw, L, W)
    c = cos(yaw);
    s = sin(yaw);
    R = [c -s; s c];

    corners = [ L/2,  W/2;
                L/2, -W/2;
               -L/2, -W/2;
               -L/2,  W/2]';

    worldCorners = R * corners + center(:);

    px = [worldCorners(1,:), worldCorners(1,1)];
    py = [worldCorners(2,:), worldCorners(2,1)];
end

function a = wrapToPiLocal(a)
    a = mod(a + pi, 2*pi) - pi;
end