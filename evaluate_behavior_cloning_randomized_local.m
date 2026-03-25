% evaluate_behavior_cloning_randomized_local.m
clear; clc; close all;

%% =========================
% Randomized local evaluation of BC parking policy
% =========================
% Purpose:
%   Evaluate the behavior cloning policy on randomly sampled local starts
%   around the neighborhood where it was trained/tested.
%
% Output:
%   - success / collision / out-of-bounds / timeout counts
%   - average final errors
%   - trajectory plot
%   - saved results file
%
% Requirements:
%   - bc_parallelpark_policy.mat
%   - helper files on MATLAB path:
%       carPolygon.m
%       projectPolygon.m
%       polygonsOverlapSAT.m
%       egoHitsCurb.m
%       checkParkingCollision.m

%% Load trained BC policy
load('bc_parallelpark_policy.mat', 'bcPolicy');

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

% Success tolerances
GoalRadius = 0.10;
GoalYawTol = deg2rad(25);
GoalSpeedTol = 0.06;

%% =========================
% Randomized local test set
% =========================
rng(42);   % reproducible

numTests = 50;

xRange = [0.00, 0.16];
yRange = [0.16, 0.24];
psiDegRange = [-4, 4];

starts = zeros(numTests, 3);
for i = 1:numTests
    starts(i,1) = xRange(1) + (xRange(2)-xRange(1))*rand;
    starts(i,2) = yRange(1) + (yRange(2)-yRange(1))*rand;
    starts(i,3) = deg2rad(psiDegRange(1) + (psiDegRange(2)-psiDegRange(1))*rand);
end

%% Results struct
results = repmat(struct( ...
    'start', [], ...
    'success', [], ...
    'collision', [], ...
    'outOfBounds', [], ...
    'timeout', [], ...
    'finalPosErr', [], ...
    'finalYawErr', [], ...
    'steps', [], ...
    'X', [], ...
    'Y', [], ...
    'PSI', [], ...
    'V', [], ...
    'DELTA', [], ...
    'STEER_CMD', [], ...
    'VCMD', [], ...
    'score', []), numTests, 1);

bestIdx = 1;
bestScore = inf;

%% =========================
% Run all tests
% =========================
for s = 1:numTests

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
    timeout = false;

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

        % Rotated collision check
        [collision, ~, ~, ~, ~, ~, ~] = ...
            checkParkingCollision([x, y], psi, EgoLength, EgoWidth, ...
                                  FrontCarCenter, RearCarCenter, ...
                                  ParkedCarLength, ParkedCarWidth, CurbY);

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

    if ~(success || collision || outOfBounds)
        timeout = true;
        X = X(1:MaxSteps);
        Y = Y(1:MaxSteps);
        PSI = PSI(1:MaxSteps);
        V = V(1:MaxSteps);
        DELTA = DELTA(1:MaxSteps);
        STEER_CMD = STEER_CMD(1:MaxSteps);
        VCMD = VCMD(1:MaxSteps);
    end

    finalPosErr = hypot(X(end) - TargetX, Y(end) - TargetY);
    finalYawErr = abs(wrapToPiLocal(TargetYaw - PSI(end)));

    score = finalPosErr + 0.5*finalYawErr;
    if collision
        score = score + 100;
    elseif outOfBounds
        score = score + 200;
    elseif timeout
        score = score + 50;
    elseif ~success
        score = score + 10;
    end

    results(s).start = starts(s,:);
    results(s).success = success;
    results(s).collision = collision;
    results(s).outOfBounds = outOfBounds;
    results(s).timeout = timeout;
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

%% =========================
% Summary metrics
% =========================
successCount = sum([results.success]);
collisionCount = sum([results.collision]);
oobCount = sum([results.outOfBounds]);
timeoutCount = sum([results.timeout]);

finalPosErrs = [results.finalPosErr];
finalYawErrs = rad2deg([results.finalYawErr]);

fprintf('\n==== Randomized Local BC Evaluation ====\n');
fprintf('Num tests         : %d\n', numTests);
fprintf('Successes         : %d\n', successCount);
fprintf('Collisions        : %d\n', collisionCount);
fprintf('Out of bounds     : %d\n', oobCount);
fprintf('Timeouts          : %d\n', timeoutCount);
fprintf('Success rate      : %.2f %%\n', 100*successCount/numTests);
fprintf('Mean pos err      : %.4f\n', mean(finalPosErrs));
fprintf('Median pos err    : %.4f\n', median(finalPosErrs));
fprintf('Mean yaw err (deg): %.2f\n', mean(finalYawErrs));
fprintf('Median yaw err    : %.2f\n', median(finalYawErrs));
fprintf('Best trajectory index: %d\n', bestIdx);

%% Detailed table
fprintf('\nIdx     x       y      psi(deg)   succ   coll   oob   tout   posErr   yawErr(deg)   steps\n');
for s = 1:numTests
    r = results(s);
    fprintf('%2d   %6.3f  %6.3f   %7.2f      %d      %d      %d      %d   %7.4f    %7.2f      %4d\n', ...
        s, r.start(1), r.start(2), rad2deg(r.start(3)), ...
        r.success, r.collision, r.outOfBounds, r.timeout, ...
        r.finalPosErr, rad2deg(r.finalYawErr), r.steps);
end

%% =========================
% Plot all trajectories
% =========================
figure('Name','BC Randomized Local Evaluation - All Trajectories');
hold on; grid on; axis equal;

drawCar(RearCarCenter, 0, ParkedCarLength, ParkedCarWidth, [0.7 0.7 0.7]);
drawCar(FrontCarCenter, 0, ParkedCarLength, ParkedCarWidth, [0.7 0.7 0.7]);
drawCar([TargetX, TargetY], TargetYaw, EgoLength, EgoWidth, [0.2 0.8 0.2]);

plot([Geofence(1), Geofence(2)], [CurbY, CurbY], 'k--', 'LineWidth', 1.5);
rectangle('Position',[-SlotLength/2 CurbY SlotLength SlotDepth], ...
    'EdgeColor',[0 0.5 0], 'LineStyle','--', 'LineWidth',1.5);

for s = 1:numTests
    r = results(s);
    if r.success
        plot(r.X, r.Y, 'g-', 'LineWidth', 1.2);
    elseif r.collision
        plot(r.X, r.Y, 'r-', 'LineWidth', 1.0);
    elseif r.outOfBounds
        plot(r.X, r.Y, 'm-', 'LineWidth', 1.0);
    else
        plot(r.X, r.Y, 'b-', 'LineWidth', 1.0);
    end
end

xlabel('X (m)');
ylabel('Y (m)');
title('Behavior Cloning - Randomized Local Evaluation');

%% =========================
% Plot start points colored by outcome
% =========================
figure('Name','Randomized Starts by Outcome');
hold on; grid on; axis equal;

for s = 1:numTests
    r = results(s);
    if r.success
        plot(r.start(1), r.start(2), 'go', 'MarkerSize', 7, 'LineWidth', 1.5);
    elseif r.collision
        plot(r.start(1), r.start(2), 'ro', 'MarkerSize', 7, 'LineWidth', 1.5);
    elseif r.outOfBounds
        plot(r.start(1), r.start(2), 'mo', 'MarkerSize', 7, 'LineWidth', 1.5);
    else
        plot(r.start(1), r.start(2), 'bo', 'MarkerSize', 7, 'LineWidth', 1.5);
    end
end

xlabel('Start x');
ylabel('Start y');
title('Randomized Local Starts by Outcome');
legend('Success','Collision','OutOfBounds','Timeout','Location','best');

%% =========================
% Plot best run
% =========================
r = results(bestIdx);

figure('Name','Best Randomized Local BC Run');
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
title(sprintf('Best Randomized Local BC Run (Index %d)', bestIdx));

%% Save results
save('bc_parallelpark_randomized_local_eval_results.mat', 'results', 'starts');
disp('Saved bc_parallelpark_randomized_local_eval_results.mat');

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

function drawCar(center, yaw, L, W, colorVal)
    poly = carPolygon(center, yaw, L, W);
    p = [poly; poly(1,:)];
    patch(p(:,1), p(:,2), colorVal, 'FaceAlpha', 0.4, 'EdgeColor', 'k');
end

function a = wrapToPiLocal(a)
    a = mod(a + pi, 2*pi) - pi;
end