% test_hard_region_expert_single_start_debug.m
clear; clc; close all;

%% =========================
% Test a hard-region expert on one failed start (debug version)
% =========================
% This version tells you EXACTLY why "collision" happened.
% It separates:
%   - hitFront
%   - hitRear
%   - hitCurb
%
% It also draws the actual AABB collision boxes so you can compare
% the real collision logic to the rotated car drawing.

%% Parameters
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

% Success tolerances
GoalRadius = 0.10;
GoalYawTol = deg2rad(25);
GoalSpeedTol = 0.06;

% Early stop gate
softStopPosTol = 0.2;
softStopYawTol = deg2rad(20);

%% =========================
% Hard-region test start
% =========================
x = 0.04;
y = 0.16;
psi = deg2rad(-3);
v = 0.0;
delta = 0.0;

%% =========================
% Modified hard-region expert parameters
% =========================
entryX = 0.75;
reverseTurnYawTarget = -0.55;
headingCorrectionTol = deg2rad(1.5);

steer_approach   =  0.00;  vref_approach  =  0.12;
steer_turnin     = -0.24;  vref_turnin    = -0.12;
steer_straight   =  0.75;  vref_straight  = -0.08;

steer_correct_l  = -0.12;
steer_correct_r  =  0.18;
vref_correct     = -0.02;

steer_stop       =  0.00;  vref_stop      =  0.00;

phase = "forward_approach";

%% Logging
X = nan(MaxSteps,1);
Y = nan(MaxSteps,1);
PSI = nan(MaxSteps,1);
V = nan(MaxSteps,1);
DELTA = nan(MaxSteps,1);
STEER_CMD = nan(MaxSteps,1);
VREF = nan(MaxSteps,1);
PHASE_NAME = strings(MaxSteps,1);

% Debug logs
hitFrontLog = false(MaxSteps,1);
hitRearLog  = false(MaxSteps,1);
hitCurbLog  = false(MaxSteps,1);

success = false;
collision = false;
outOfBounds = false;

% Final debug boxes
finalEgoBox = [];
finalFrontBox = [];
finalRearBox = [];

%% Simulation loop
for k = 1:MaxSteps

    posErr = hypot(x - TargetX, y - TargetY);
    yawErrSigned = wrapToPiLocal(TargetYaw - psi);
    yawErr = abs(yawErrSigned);

    % Early stop gate
    if posErr < softStopPosTol && yawErr < softStopYawTol
        phase = "stop";
    end

    % Phase transitions
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
            % remain here
    end

    % Commands by phase
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

    steerCmd = min(max(steerCmd, SteerMin), SteerMax);

    if phase == "stop"
        vCmd = 2.5 * (vRef - v);
    else
        vCmd = 1.5 * (vRef - v);
    end
    vCmd = min(max(vCmd, -Vmax), Vmax);

    % Dynamics
    delta = delta + (Ts/TauSteer) * (steerCmd - delta);
    v = 0.85*v + 0.15*vCmd;

    x = x + Ts * v * cos(psi);
    y = y + Ts * v * sin(psi);
    psi = wrapToPiLocal(psi + Ts * (v/Wheelbase) * tan(delta));

    % Checks
    egoBox = getEgoAABB(x, y, EgoLength, EgoWidth);
    frontBox = getRectAABB(FrontCarCenter, ParkedCarLength, ParkedCarWidth);
    rearBox  = getRectAABB(RearCarCenter,  ParkedCarLength, ParkedCarWidth);

    hitFront = aabbOverlap(egoBox, frontBox);
    hitRear  = aabbOverlap(egoBox, rearBox);
    hitCurb  = (egoBox(3) < CurbY);

    collision = hitFront || hitRear || hitCurb;

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

    hitFrontLog(k) = hitFront;
    hitRearLog(k)  = hitRear;
    hitCurbLog(k)  = hitCurb;

    if success || collision || outOfBounds
        X = X(1:k);
        Y = Y(1:k);
        PSI = PSI(1:k);
        V = V(1:k);
        DELTA = DELTA(1:k);
        STEER_CMD = STEER_CMD(1:k);
        VREF = VREF(1:k);
        PHASE_NAME = PHASE_NAME(1:k);

        hitFrontLog = hitFrontLog(1:k);
        hitRearLog  = hitRearLog(1:k);
        hitCurbLog  = hitCurbLog(1:k);

        finalEgoBox = egoBox;
        finalFrontBox = frontBox;
        finalRearBox = rearBox;
        break;
    end
end

%% Summary
fprintf('\n==== Hard-Region Expert Single-Start Debug Test ====\n');
fprintf('Start x: %.4f\n', 0.04);
fprintf('Start y: %.4f\n', 0.16);
fprintf('Start psi (deg): %.2f\n', -3.0);
fprintf('Steps: %d\n', length(X));
fprintf('Success: %d\n', success);
fprintf('Collision: %d\n', collision);
fprintf('OutOfBounds: %d\n', outOfBounds);
fprintf('Final x: %.4f\n', X(end));
fprintf('Final y: %.4f\n', Y(end));
fprintf('Final yaw: %.4f rad (%.2f deg)\n', PSI(end), rad2deg(PSI(end)));
fprintf('Final speed: %.4f m/s\n', V(end));
fprintf('Final pos err: %.4f\n', hypot(X(end)-TargetX, Y(end)-TargetY));
fprintf('Final yaw err: %.4f rad (%.2f deg)\n', ...
    abs(wrapToPiLocal(TargetYaw - PSI(end))), ...
    rad2deg(abs(wrapToPiLocal(TargetYaw - PSI(end)))));
fprintf('Final phase: %s\n', PHASE_NAME(end));

fprintf('\nCollision breakdown:\n');
fprintf('hitFront: %d\n', hitFrontLog(end));
fprintf('hitRear : %d\n', hitRearLog(end));
fprintf('hitCurb : %d\n', hitCurbLog(end));

%% Plot trajectory with collision boxes
figure('Name','Hard-Region Expert Single-Start Debug');
hold on; grid on; axis equal;

drawCar(RearCarCenter, 0, ParkedCarLength, ParkedCarWidth, [0.7 0.7 0.7]);
drawCar(FrontCarCenter, 0, ParkedCarLength, ParkedCarWidth, [0.7 0.7 0.7]);
drawCar([TargetX, TargetY], TargetYaw, EgoLength, EgoWidth, [0.2 0.8 0.2]);

plot(X, Y, 'b.-', 'LineWidth', 1.5, 'MarkerSize', 10);
plot(X(1), Y(1), 'go', 'MarkerSize', 8, 'LineWidth', 1.5);
plot(X(end), Y(end), 'ro', 'MarkerSize', 8, 'LineWidth', 1.5);

plot([Geofence(1), Geofence(2)], [CurbY, CurbY], 'k--', 'LineWidth', 1.5);
rectangle('Position',[-SlotLength/2 CurbY SlotLength SlotDepth], ...
    'EdgeColor',[0 0.5 0], 'LineStyle','--', 'LineWidth',1.5);

% Draw actual collision boxes from the final state
if ~isempty(finalEgoBox)
    drawAABB(finalEgoBox, [1 0 0]);      % red = ego collision box
    drawAABB(finalFrontBox, [1 1 0]);    % yellow = front parked car collision box
    drawAABB(finalRearBox,  [1 1 0]);    % yellow = rear parked car collision box
end

xlabel('X (m)');
ylabel('Y (m)');
title('Hard-Region Expert Single-Start Debug');

%% Plot controls
figure('Name','Hard-Region Expert Controls');

subplot(3,1,1);
plot(V, 'LineWidth', 1.5); grid on;
ylabel('v');

subplot(3,1,2);
plot(STEER_CMD, 'LineWidth', 1.5); grid on;
ylabel('steer');

subplot(3,1,3);
plot(VREF, 'LineWidth', 1.5); grid on;
ylabel('v ref');
xlabel('Step');

%% Plot collision flags
figure('Name','Collision Flags');
stairs(double(hitFrontLog), 'LineWidth', 1.5); hold on;
stairs(double(hitRearLog),  'LineWidth', 1.5);
stairs(double(hitCurbLog),  'LineWidth', 1.5);
grid on;
xlabel('Step');
ylabel('Triggered');
legend('hitFront', 'hitRear', 'hitCurb');
title('Collision Cause Over Time');

%% Animation
figure('Name','Hard-Region Expert Animation Debug');
hold on; grid on; axis equal;
xlim([Geofence(1), Geofence(2)]);
ylim([Geofence(3), Geofence(4)]);
xlabel('X (m)');
ylabel('Y (m)');
title('Hard-Region Expert Animation Debug');

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

%% Local functions
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

function drawAABB(box, colorVal)
    rectangle('Position', [box(1), box(3), box(2)-box(1), box(4)-box(3)], ...
        'EdgeColor', colorVal, 'LineWidth', 1.5, 'LineStyle', '--');
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