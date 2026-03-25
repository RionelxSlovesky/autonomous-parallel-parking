% evaluate_behavior_cloning_wider_neighborhood.m
clear; clc; close all;

%% =========================
% Evaluate BC policy on a wider neighborhood
% =========================

%% Load trained BC policy
load('bc_parallelpark_policy.mat', 'bcPolicy');

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

% Success tolerances (same as current BC evaluation)
GoalRadius = 0.10;
GoalYawTol = deg2rad(25);
GoalSpeedTol = 0.06;

%% Wider neighborhood of starts
% Slightly wider than the earlier local test
starts = [
    0.00, 0.16, deg2rad(-4);
    0.00, 0.20, deg2rad( 0);
    0.00, 0.24, deg2rad( 4);

    0.04, 0.16, deg2rad(-3);
    0.04, 0.20, deg2rad( 0);
    0.04, 0.24, deg2rad( 3);

    0.08, 0.16, deg2rad(-2);
    0.08, 0.20, deg2rad( 0);
    0.08, 0.24, deg2rad( 2);

    0.12, 0.16, deg2rad(-3);
    0.12, 0.20, deg2rad( 0);
    0.12, 0.24, deg2rad( 3);

    0.16, 0.18, deg2rad(-2);
    0.16, 0.22, deg2rad( 2);
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

    X = nan(MaxSteps,1);
    Y = nan(MaxSteps,1);
    PSI = nan(MaxSteps,1);
    V = nan(MaxSteps,1);
    DELTA = nan(MaxSteps,1);
    STEER_CMD = nan(MaxSteps,1);
    VCMD = nan(MaxSteps,1);

    success = false;
    collision = false;
    outOfBounds = false;

    for k = 1:MaxSteps

        % Build BC state
        state = buildState(x, y, psi, v, delta, TargetX, TargetY, TargetYaw);

        % Predict action from BC policy
        action = bc_policy_action(state, bcPolicy);

        steerCmd = action(1);
        vCmd = action(2);

        % Clamp actions
        steerCmd = min(max(steerCmd, SteerMin), SteerMax);
        vCmd     = min(max(vCmd, -Vmax), Vmax);

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
        VCMD(k) = vCmd;

        if success || collision || outOfBounds
            X = X(1:k);
            Y = Y(1:k);
            PSI = PSI(1:k);
            V = V(1:k);
            DELTA = DELTA(1:k);
            STEER_CMD = STEER_CMD(1:k);
            VCMD = VCMD(1:k);
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
    results(s).VCMD = VCMD;
    results(s).score = score;

    if score < bestScore
        bestScore = score;
        bestIdx = s;
    end
end

%% Summary table
fprintf('\n==== Behavior Cloning Evaluation: Wider Neighborhood ====\n');
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

fprintf('\nBC Wider-Neighborhood Successes: %d / %d\n', numSuccess, numStarts);
fprintf('Best BC trajectory index: %d\n', bestIdx);

%% Plot all trajectories
figure('Name','BC Wider Neighborhood - All Trajectories');
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
title('Behavior Cloning - Wider Neighborhood');
legend('Rear parked car','Front parked car','Target pose','Location','best');

%% Plot best run
r = results(bestIdx);

figure('Name','Best BC Wider-Neighborhood Run');
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
title(sprintf('Best BC Wider-Neighborhood Run (Index %d)', bestIdx));

%% Save results
save('bc_parallelpark_wider_eval_results.mat', 'results');
disp('Saved bc_parallelpark_wider_eval_results.mat');

%% =========================
% Local functions
% =========================
function action = bc_policy_action(state, bcPolicy)
    state = state(:)';  % row
    stateN = (state - bcPolicy.xMean) ./ bcPolicy.xStd;
    actionN = predict(bcPolicy.net, stateN);
    action = actionN .* bcPolicy.yStd + bcPolicy.yMean;
    action = action(:);
end

function s = buildState(x, y, psi, v, delta, TargetX, TargetY, TargetYaw)
    dx = TargetX - x;
    dy = TargetY - y;

    c = cos(TargetYaw);
    s2 = sin(TargetYaw);

    d_long =  c*dx + s2*dy;
    d_lat  = -s2*dx + c*dy;
    e_yaw  = wrapToPiLocal(TargetYaw - psi);

    s = [x; y; psi; v; delta; d_long; d_lat; e_yaw];
end

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