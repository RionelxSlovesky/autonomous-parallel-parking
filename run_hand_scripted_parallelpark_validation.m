% run_hand_scripted_parallelpark_validation.m
clear; clc; close all;

%% =========================
% Hand-Scripted Parallel Parking Validation
% =========================
% Goal:
%   Verify that the parking geometry is physically feasible using a
%   manually scripted maneuver with fixed control phases.
%
% Phases:
%   1) Forward straight approach
%   2) Reverse turn-in
%   3) Reverse straighten
%   4) Stop
%
% This script uses:
%   - kinematic bicycle model
%   - steering lag
%   - simple collision checks
%   - visualization + animation

%% =========================
% Parameters
% =========================
Ts = 0.02;
MaxSteps = 5000;

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
SlotLength = 0.72;
SlotDepth  = 0.30;

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

% Success tolerances
GoalRadius = 0.08;
GoalYawTol = deg2rad(10);
GoalSpeedTol = 0.05;

%% =========================
% Initial pose
% =========================
x = 0.60;
y = 0.20;
psi = 0.00;
v = 0.0;
delta = 0.0;

%% =========================
% Hand-scripted phase plan
% =========================
% You can tune these 6 numbers first before changing anything else.
N1 = 140;   % forward approach duration
N2 = 240;   % reverse turn-in duration
N3 = 220;   % reverse straighten duration
N4 =  80;   % stop/settle duration

steer1 =  0.00;  vref1 =  0.12;   % forward straight
steer2 = -0.48;  vref2 = -0.10;   % reverse into slot
steer3 =  0.46;  vref3 = -0.08;   % reverse straighten
steer4 =  0.00;  vref4 =  0.00;   % stop

phaseBreaks = [N1, N1+N2, N1+N2+N3, N1+N2+N3+N4];

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

    % ----- phase selection -----
    if k <= phaseBreaks(1)
        steerCmd = steer1;
        vRef = vref1;
        phaseName = "forward_approach";
    elseif k <= phaseBreaks(2)
        steerCmd = steer2;
        vRef = vref2;
        phaseName = "reverse_turn_in";
    elseif k <= phaseBreaks(3)
        steerCmd = steer3;
        vRef = vref3;
        phaseName = "reverse_straighten";
    elseif k <= phaseBreaks(4)
        steerCmd = steer4;
        vRef = vref4;
        phaseName = "stop";
    else
        steerCmd = 0.0;
        vRef = 0.0;
        phaseName = "done";
    end

    % clamp steering
    steerCmd = min(max(steerCmd, SteerMin), SteerMax);

    % simple P speed controller
    vCmd = 1.5 * (vRef - v);
    vCmd = min(max(vCmd, -Vmax), Vmax);

    % steering lag
    delta = delta + (Ts/TauSteer) * (steerCmd - delta);

    % speed smoothing
    v = 0.85*v + 0.15*vCmd;

    % kinematic bicycle update
    x = x + Ts * v * cos(psi);
    y = y + Ts * v * sin(psi);
    psi = wrapToPiLocal(psi + Ts * (v/Wheelbase) * tan(delta));

    % checks
    egoBox = getEgoAABB(x, y, EgoLength, EgoWidth);
    frontBox = getRectAABB(FrontCarCenter, ParkedCarLength, ParkedCarWidth);
    rearBox  = getRectAABB(RearCarCenter,  ParkedCarLength, ParkedCarWidth);

    collision = aabbOverlap(egoBox, frontBox) || aabbOverlap(egoBox, rearBox) || (egoBox(3) < CurbY);
    outOfBounds = (x < Geofence(1)) || (x > Geofence(2)) || ...
                  (y < Geofence(3)) || (y > Geofence(4));

    posErr = hypot(x - TargetX, y - TargetY);
    yawErr = abs(wrapToPiLocal(TargetYaw - psi));
    success = (posErr < GoalRadius) && (yawErr < GoalYawTol) && (abs(v) < GoalSpeedTol);

    % log
    X(k) = x;
    Y(k) = y;
    PSI(k) = psi;
    V(k) = v;
    DELTA(k) = delta;
    STEER_CMD(k) = steerCmd;
    VREF(k) = vRef;
    PHASE_NAME(k) = phaseName;

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
fprintf('\n==== Hand-Scripted Parking Validation Summary ====\n');
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

%% =========================
% Plot trajectory
% =========================
figure('Name','Hand-Scripted Parallel Parking Validation');
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
title('Hand-Scripted Parallel Parking Validation');

%% =========================
% Plot states and controls
% =========================
figure('Name','States and Commands');

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
figure('Name','Hand-Scripted Parking Animation');
hold on; grid on; axis equal;
xlim([Geofence(1), Geofence(2)]);
ylim([Geofence(3), Geofence(4)]);
xlabel('X (m)');
ylabel('Y (m)');
title('Hand-Scripted Parking Animation');

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