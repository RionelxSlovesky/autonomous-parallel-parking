% test_state_based_parallelpark_nearby_starts.m
clear; clc; close all;

%% =========================
% Test succeeded state-based parking controller on nearby start poses
% =========================

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

% ===== Use the succeeded code's success tolerances =====
GoalRadius = 0.10;
GoalYawTol = deg2rad(25);
GoalSpeedTol = 0.06;

% ===== Use the succeeded code's early stop gate =====
softStopPosTol = 0.2;
softStopYawTol = deg2rad(20);

%% Controller parameters from the succeeded code
entryX = 0.7;
reverseTurnYawTarget = -0.55;
headingCorrectionTol = deg2rad(1.5);

steer_approach   =  0.00;  vref_approach  =  0.12;
steer_turnin     = -0.30;  vref_turnin    = -0.12;
steer_straight   =  0.90;  vref_straight  = -0.08;

steer_correct_l  = -0.12;
steer_correct_r  =  0.18;
vref_correct     = -0.02;

steer_stop       =  0.00;  vref_stop      =  0.00;

%% Nearby start poses around the working one
% Base successful start:
% x = 0.06; y = 0.20; psi = 0.00
starts = [
    0.06, 0.20, deg2rad( 0);
    0.03, 0.18, deg2rad(-2);
    0.09, 0.22, deg2rad( 2);
    0.05, 0.17, deg2rad( 3);
    0.08, 0.21, deg2rad(-3);
    0.04, 0.19, deg2rad( 1);
    0.07, 0.23, deg2rad(-1);
    0.02, 0.20, deg2rad( 0);
    0.10, 0.20, deg2rad( 0);
    0.06, 0.16, deg2rad( 0);
    0.06, 0.24, deg2rad( 0);
];

numStarts = size(starts,1);
results = struct([]);

bestIdx = 1;
bestScore = inf;

%% Run all starts
for s = 1:numStarts

    x = starts(s,1);
    y = starts(s,2);
    psi = starts(s,3);
    v = 0.0;
    delta = 0.0;

    phase = "forward_approach";

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

    for k = 1:MaxSteps

        posErr = hypot(x - TargetX, y - TargetY);
        yawErrSigned = wrapToPiLocal(TargetYaw - psi);
        yawErr = abs(yawErrSigned);

        % Early stop gate from succeeded code
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
                % stay here
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

    finalPosErr = hypot(X(end) - TargetX, Y(end) - TargetY);
    finalYawErr = abs(wrapToPiLocal(TargetYaw - PSI(end)));

    score = finalPosErr + 0.5*finalYawErr;
    if collision
        score = score + 100;
    elseif outOfBounds
        score = score + 200;
    elseif ~success
        score = score + 10;
    end

    results(s).start = starts(s,:);
    results(s).success = success;
    results(s).collision = collision;
    results(s).outOfBounds = outOfBounds;
    results(s).finalPosErr = finalPosErr;
    results(s).finalYawErr = finalYawErr;
    results(s).steps = length(X);
    results(s).X = X;
    results(s).Y = Y;
    results(s).PSI = PSI;
    results(s).V = V;
    results(s).DELTA = DELTA;
    results(s).STEER_CMD = STEER_CMD;
    results(s).VREF = VREF;
    results(s).PHASE_NAME = PHASE_NAME;
    results(s).score = score;

    if score < bestScore
        bestScore = score;
        bestIdx = s;
    end
end

%% Summary table
fprintf('\n==== Nearby Start Pose Robustness Test (Succeeded Controller) ====\n');
fprintf('Idx     x       y      psi(deg)   succ   coll   oob   posErr   yawErr(deg)   steps\n');

numSuccess = 0;
for s = 1:numStarts
    r = results(s);
    numSuccess = numSuccess + r.success;
    fprintf('%2d   %6.3f  %6.3f   %7.2f      %d      %d      %d    %7.4f    %7.2f      %4d\n', ...
        s, r.start(1), r.start(2), rad2deg(r.start(3)), ...
        r.success, r.collision, r.outOfBounds, ...
        r.finalPosErr, rad2deg(r.finalYawErr), r.steps);
end

fprintf('\nSuccesses: %d / %d\n', numSuccess, numStarts);
fprintf('Best trajectory index: %d\n', bestIdx);

%% Plot all trajectories
figure('Name','Nearby Start Robustness - All Trajectories');
hold on; grid on; axis equal;

drawCar(RearCarCenter, 0, ParkedCarLength, ParkedCarWidth, [0.7 0.7 0.7]);
drawCar(FrontCarCenter, 0, ParkedCarLength, ParkedCarWidth, [0.7 0.7 0.7]);
drawCar([TargetX, TargetY], TargetYaw, EgoLength, EgoWidth, [0.2 0.8 0.2]);

plot([Geofence(1), Geofence(2)], [CurbY, CurbY], 'k--', 'LineWidth', 1.5);
rectangle('Position',[-SlotLength/2 CurbY SlotLength SlotDepth], ...
    'EdgeColor',[0 0.5 0], 'LineStyle','--', 'LineWidth',1.5);

for s = 1:numStarts
    r = results(s);
    if r.success
        plot(r.X, r.Y, 'g-', 'LineWidth', 1.8);
    else
        plot(r.X, r.Y, 'r-', 'LineWidth', 1.2);
    end
    plot(r.X(1), r.Y(1), 'ko', 'MarkerSize', 5, 'HandleVisibility','off');
end

xlabel('X (m)');
ylabel('Y (m)');
title('Nearby Start Pose Robustness Test');
legend('Rear parked car','Front parked car','Target pose','Location','best');

%% Plot best run
r = results(bestIdx);

figure('Name','Best Nearby Start Run');
hold on; grid on; axis equal;

drawCar(RearCarCenter, 0, ParkedCarLength, ParkedCarWidth, [0.7 0.7 0.7]);
drawCar(FrontCarCenter, 0, ParkedCarLength, ParkedCarWidth, [0.7 0.7 0.7]);
drawCar([TargetX, TargetY], TargetYaw, EgoLength, EgoWidth, [0.2 0.8 0.2]);

plot([Geofence(1), Geofence(2)], [CurbY, CurbY], 'k--', 'LineWidth', 1.5);
rectangle('Position',[-SlotLength/2 CurbY SlotLength SlotDepth], ...
    'EdgeColor',[0 0.5 0], 'LineStyle','--', 'LineWidth',1.5);

plot(r.X, r.Y, 'b.-', 'LineWidth', 1.5, 'MarkerSize', 10);
plot(r.X(1), r.Y(1), 'go', 'MarkerSize', 8, 'LineWidth', 1.5);
plot(r.X(end), r.Y(end), 'ro', 'MarkerSize', 8, 'LineWidth', 1.5);

xlabel('X (m)');
ylabel('Y (m)');
title(sprintf('Best Nearby Start Run (Index %d)', bestIdx));

%% Save results
save('nearby_start_pose_results.mat', 'results');
disp('Saved results to nearby_start_pose_results.mat');

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