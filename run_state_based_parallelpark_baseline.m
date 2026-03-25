% run_state_based_parallelpark_baseline.m
clear; clc; close all;

%% =========================
% State-Based Scripted Parallel Parking Baseline
% =========================
% Key update in this version:
%   - Adds an EARLY STOP GATE so the car stops once it is "good enough"
%     instead of continuing to over-park.
%
% Phase sequence:
%   1) forward_approach
%   2) reverse_turn_in
%   3) reverse_straighten
%   4) reverse_heading_correction
%   5) stop

%% =========================
% Parameters
% =========================
Ts = 0.02;
MaxSteps = 12000;

% Vehicle model
Wheelbase = 0.256;
SteerMin = -0.5;
SteerMax =  0.5;
TauSteer = 0.16;
Vmax = 0.25;

% Ego vehicle dimensions
EgoLength = 0.32;
EgoWidth  = 0.18;

% Parked vehicles
ParkedCarLength = 0.36;
ParkedCarWidth  = 0.20;

% Slot geometry
SlotLength = 0.90;
SlotDepth  = 0.36;

% World
Geofence = [-1.5 1.5 -0.8 1.0];
CurbY = -0.30;

% Target pose
TargetX = 0.0;
TargetY = CurbY + SlotDepth/2;
TargetYaw = 0.0;

% Parked car centers
gapHalf = SlotLength / 2;
carHalf = ParkedCarLength / 2;
FrontCarCenter = [ gapHalf + carHalf, TargetY];
RearCarCenter  = [-gapHalf - carHalf, TargetY];

% Strict success tolerances
GoalRadius = 0.10;
GoalYawTol = deg2rad(25);
GoalSpeedTol = 0.06;

% Early stop gate (so it stops before over-parking)
softStopPosTol = 0.2;
softStopYawTol = deg2rad(20);

%% =========================
% Initial pose
% =========================
x = 0.06;
y = 0.20;
psi = 0.00;
v = 0.0;
delta = 0.0;

%% =========================
% Phase parameters
% =========================
entryX = 0.7;                  % forward until x reaches this
reverseTurnYawTarget = -0.55;   % switch from turn-in to straighten
headingCorrectionTol = deg2rad(1.5);

% Commands by phase
steer_approach   =  0.00;  vref_approach  =  0.12;
steer_turnin     = -0.30;  vref_turnin    = -0.12;
steer_straight   =  0.90;  vref_straight  = -0.08;

% Small final correction commands
steer_correct_l  = -0.12;
steer_correct_r  =  0.18;
vref_correct     = -0.02;

steer_stop       =  0.00;  vref_stop      =  0.00;

phase = "forward_approach";

%% =========================
% Logging
% =========================
X = nan(MaxSteps,1);
Y = nan(MaxSteps,1);
PSI = nan(MaxSteps,1);
V = nan(MaxSteps,1);
DELTA = nan(MaxSteps,1);
STEER_CMD = nan(MaxSteps,1);
VREF = nan(MaxSteps,1);
PHASE_NAME = strings(MaxSteps,1);

success = false;
collision = false;
outOfBounds = false;

%% =========================
% Simulation loop
% =========================
for k = 1:MaxSteps

    posErr = hypot(x - TargetX, y - TargetY);
    yawErrSigned = wrapToPiLocal(TargetYaw - psi);
    yawErr = abs(yawErrSigned);

    % ----- Early stop gate -----
    % Once the car is already in a decent parking pose, stop motion
    % so it does not keep over-parking.
    if posErr < softStopPosTol && yawErr < softStopYawTol
        phase = "stop";
    end

    % ----- Phase transitions -----
    switch phase
        case "forward_approach"
            if x >= entryX
                phase = "reverse_turn_in";
            end

        case "reverse_turn_in"
            if psi <= reverseTurnYawTarget || y <= (TargetY + 0.10)
                phase = "reverse_straighten";
            end

        case "reverse_straighten"
            if posErr < 0.12 && yawErr < deg2rad(15)
                phase = "reverse_heading_correction";
            end

        case "reverse_heading_correction"
            if posErr < GoalRadius && yawErr < GoalYawTol && abs(v) < 0.08
                phase = "stop";
            end

        case "stop"
            % stay here
    end

    % ----- Command generation by phase -----
    switch phase
        case "forward_approach"
            steerCmd = steer_approach;
            vRef = vref_approach;

        case "reverse_turn_in"
            steerCmd = steer_turnin;
            vRef = vref_turnin;

        case "reverse_straighten"
            steerCmd = steer_straight;
            vRef = vref_straight;

        case "reverse_heading_correction"
            % Small final touch-up based on heading error sign
            if yawErrSigned > headingCorrectionTol
                steerCmd = steer_correct_l;
            elseif yawErrSigned < -headingCorrectionTol
                steerCmd = steer_correct_r;
            else
                steerCmd = 0.0;
            end
            vRef = vref_correct;

        case "stop"
            steerCmd = steer_stop;
            vRef = vref_stop;
    end

    % Clamp steering
    steerCmd = min(max(steerCmd, SteerMin), SteerMax);

    % Speed controller
    % Stronger braking once stop is triggered
    if phase == "stop"
        vCmd = 2.5 * (vRef - v);
    else
        vCmd = 1.5 * (vRef - v);
    end
    vCmd = min(max(vCmd, -Vmax), Vmax);

    % Steering lag
    delta = delta + (Ts/TauSteer) * (steerCmd - delta);

    % Speed smoothing
    v = 0.85*v + 0.15*vCmd;

    % Kinematic bicycle update
    x = x + Ts * v * cos(psi);
    y = y + Ts * v * sin(psi);
    psi = wrapToPiLocal(psi + Ts * (v/Wheelbase) * tan(delta));

    % Checks
    egoBox = getEgoAABB(x, y, EgoLength, EgoWidth);
    frontBox = getRectAABB(FrontCarCenter, ParkedCarLength, ParkedCarWidth);
    rearBox  = getRectAABB(RearCarCenter,  ParkedCarLength, ParkedCarWidth);

    collision = aabbOverlap(egoBox, frontBox) || ...
                aabbOverlap(egoBox, rearBox) || ...
                (egoBox(3) < CurbY);

    outOfBounds = (x < Geofence(1)) || (x > Geofence(2)) || ...
                  (y < Geofence(3)) || (y > Geofence(4));

    posErr = hypot(x - TargetX, y - TargetY);
    yawErr = abs(wrapToPiLocal(TargetYaw - psi));
    success = (posErr < GoalRadius) && ...
              (yawErr < GoalYawTol) && ...
              (abs(v) < GoalSpeedTol);

    % Log
    X(k) = x;
    Y(k) = y;
    PSI(k) = psi;
    V(k) = v;
    DELTA(k) = delta;
    STEER_CMD(k) = steerCmd;
    VREF(k) = vRef;
    PHASE_NAME(k) = phase;

    if success || collision || outOfBounds
        X = X(1:k);
        Y = Y(1:k);
        PSI = PSI(1:k);
        V = V(1:k);
        DELTA = DELTA(1:k);
        STEER_CMD = STEER_CMD(1:k);
        VREF = VREF(1:k);
        PHASE_NAME = PHASE_NAME(1:k);
        break;
    end
end

%% =========================
% Summary
% =========================
fprintf('\n==== State-Based Scripted Parking Summary ====\n');
fprintf('Steps: %d\n', length(X));
fprintf('Success: %d\n', success);
fprintf('Collision: %d\n', collision);
fprintf('OutOfBounds: %d\n', outOfBounds);
fprintf('Final x: %.4f\n', X(end));
fprintf('Final y: %.4f\n', Y(end));
fprintf('Final yaw: %.4f rad\n', PSI(end));
fprintf('Final speed: %.4f m/s\n', V(end));
fprintf('Final pos err: %.4f\n', hypot(X(end)-TargetX, Y(end)-TargetY));
fprintf('Final yaw err: %.4f\n', abs(wrapToPiLocal(TargetYaw - PSI(end))));
fprintf('Final phase: %s\n', PHASE_NAME(end));

%% =========================
% Plot trajectory
% =========================
figure('Name','State-Based Scripted Parallel Parking');
hold on; grid on; axis equal;

drawCar(RearCarCenter, 0, ParkedCarLength, ParkedCarWidth, [0.7 0.7 0.7]);
drawCar(FrontCarCenter, 0, ParkedCarLength, ParkedCarWidth, [0.7 0.7 0.7]);
drawCar([TargetX, TargetY], TargetYaw, EgoLength, EgoWidth, [0.2 0.8 0.2]);

plot(X, Y, 'b.-', 'LineWidth', 1.5, 'MarkerSize', 10);
plot(X(1), Y(1), 'go', 'MarkerSize', 10, 'LineWidth', 2);
plot(X(end), Y(end), 'ro', 'MarkerSize', 10, 'LineWidth', 2);

plot([Geofence(1), Geofence(2)], [CurbY, CurbY], 'k--', 'LineWidth', 1.5);
rectangle('Position',[-SlotLength/2 CurbY SlotLength SlotDepth], ...
    'EdgeColor',[0 0.5 0], 'LineStyle','--', 'LineWidth',1.5);

xlabel('X (m)');
ylabel('Y (m)');
title('State-Based Scripted Parallel Parking');

%% =========================
% Plot states and commands
% =========================
figure('Name','State-Based States and Commands');

subplot(4,1,1);
plot(V, 'LineWidth', 1.5); grid on;
ylabel('v (m/s)');
title('Vehicle States and Commands');

subplot(4,1,2);
plot(DELTA, 'LineWidth', 1.5); grid on;
ylabel('\delta (rad)');

subplot(4,1,3);
plot(STEER_CMD, 'LineWidth', 1.5); grid on;
ylabel('steer cmd');

subplot(4,1,4);
plot(VREF, 'LineWidth', 1.5); grid on;
ylabel('v ref');
xlabel('Step');

%% =========================
% Animation
% =========================
figure('Name','State-Based Parking Animation');
hold on; grid on; axis equal;
xlim([Geofence(1), Geofence(2)]);
ylim([Geofence(3), Geofence(4)]);
xlabel('X (m)');
ylabel('Y (m)');
title('State-Based Parking Animation');

drawCar(RearCarCenter, 0, ParkedCarLength, ParkedCarWidth, [0.7 0.7 0.7]);
drawCar(FrontCarCenter, 0, ParkedCarLength, ParkedCarWidth, [0.7 0.7 0.7]);
drawCar([TargetX, TargetY], TargetYaw, EgoLength, EgoWidth, [0.2 0.8 0.2]);

plot([Geofence(1), Geofence(2)], [CurbY, CurbY], 'k--', 'LineWidth', 1.5);
rectangle('Position',[-SlotLength/2 CurbY SlotLength SlotDepth], ...
    'EdgeColor',[0 0.5 0], 'LineStyle','--', 'LineWidth',1.5);

hTrail = plot(nan, nan, 'b-', 'LineWidth', 1.5);
hCar = patch(nan, nan, [0.2 0.4 1.0], 'FaceAlpha', 0.5);

for k = 1:length(X)
    set(hTrail, 'XData', X(1:k), 'YData', Y(1:k));
    [px, py] = carPolygon([X(k), Y(k)], PSI(k), EgoLength, EgoWidth);
    set(hCar, 'XData', px, 'YData', py);
    drawnow;
    pause(0.03);
end

%% =========================
% Local functions
% =========================
function box = getEgoAABB(x, y, L, W)
    halfL = L/2;
    halfW = W/2;
    box = [x-halfL, x+halfL, y-halfW, y+halfW];
end

function box = getRectAABB(center, L, W)
    halfL = L/2;
    halfW = W/2;
    box = [center(1)-halfL, center(1)+halfL, center(2)-halfW, center(2)+halfW];
end

function tf = aabbOverlap(a, b)
    tf = (a(1) <= b(2)) && (a(2) >= b(1)) && ...
         (a(3) <= b(4)) && (a(4) >= b(3));
end

function drawCar(center, yaw, L, W, colorVal)
    [px, py] = carPolygon(center, yaw, L, W);
    patch(px, py, colorVal, 'FaceAlpha', 0.4, 'EdgeColor', 'k');
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