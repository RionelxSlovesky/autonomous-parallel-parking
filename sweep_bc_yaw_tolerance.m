% sweep_bc_yaw_tolerance_fast.m
clear; clc; close all;

%% Load trained BC policy
load('bc_parallelpark_policy.mat', 'bcPolicy');

%% Parameters
Ts = 0.02;
MaxSteps = 4000;   % reduced from 12000

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

% Keep these fixed during the sweep
GoalRadius = 0.10;
GoalSpeedTol = 0.06;

%% Randomized local starts (smaller test set)
rng(42);
numTests = 20;

xRange = [0.00, 0.16];
yRange = [0.16, 0.24];
psiDegRange = [-4, 4];

starts = zeros(numTests, 3);
for i = 1:numTests
    starts(i,1) = xRange(1) + (xRange(2)-xRange(1))*rand;
    starts(i,2) = yRange(1) + (yRange(2)-yRange(1))*rand;
    starts(i,3) = deg2rad(psiDegRange(1) + (psiDegRange(2)-psiDegRange(1))*rand);
end

%% Shorter yaw tolerance list
yawTolDegList = [25 20 15 10];
numTol = numel(yawTolDegList);

sweepResults = repmat(struct( ...
    'yawTolDeg', [], ...
    'successCount', [], ...
    'successRate', [], ...
    'collisionCount', [], ...
    'oobCount', [], ...
    'timeoutCount', [], ...
    'meanPosErr', [], ...
    'medianPosErr', [], ...
    'meanYawErrDeg', [], ...
    'medianYawErrDeg', []), numTol, 1);

%% Sweep loop
for t = 1:numTol

    GoalYawTol = deg2rad(yawTolDegList(t));

    successCount = 0;
    collisionCount = 0;
    oobCount = 0;
    timeoutCount = 0;

    posErrs = zeros(numTests,1);
    yawErrsDeg = zeros(numTests,1);

    for s = 1:numTests

        x = starts(s,1);
        y = starts(s,2);
        psi = starts(s,3);
        v = 0.0;
        delta = 0.0;

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

            % Early accept / stop near target to save time
            if posErr < 0.12 && yawErr < GoalYawTol && abs(v) < 0.08
                success = true;
                break;
            end

            % Normal success condition
            success = (posErr < GoalRadius) && ...
                      (yawErr < GoalYawTol) && ...
                      (abs(v) < GoalSpeedTol);

            if success || collision || outOfBounds
                break;
            end
        end

        if ~(success || collision || outOfBounds)
            timeout = true;
        end

        finalPosErr = hypot(x - TargetX, y - TargetY);
        finalYawErrDeg = rad2deg(abs(wrapToPiLocal(TargetYaw - psi)));

        posErrs(s) = finalPosErr;
        yawErrsDeg(s) = finalYawErrDeg;

        successCount = successCount + success;
        collisionCount = collisionCount + collision;
        oobCount = oobCount + outOfBounds;
        timeoutCount = timeoutCount + timeout;
    end

    sweepResults(t).yawTolDeg = yawTolDegList(t);
    sweepResults(t).successCount = successCount;
    sweepResults(t).successRate = 100 * successCount / numTests;
    sweepResults(t).collisionCount = collisionCount;
    sweepResults(t).oobCount = oobCount;
    sweepResults(t).timeoutCount = timeoutCount;
    sweepResults(t).meanPosErr = mean(posErrs);
    sweepResults(t).medianPosErr = median(posErrs);
    sweepResults(t).meanYawErrDeg = mean(yawErrsDeg);
    sweepResults(t).medianYawErrDeg = median(yawErrsDeg);
end

%% Print summary
fprintf('\n==== Fast BC Yaw Tolerance Sweep ====\n');
fprintf('YawTol(deg)   Success   Rate(%%)   Coll   OOB   Timeout   MeanPosErr   MeanYawErr(deg)\n');

for t = 1:numTol
    r = sweepResults(t);
    fprintf('%10.1f   %7d   %7.2f   %4d   %3d   %7d     %8.4f        %8.2f\n', ...
        r.yawTolDeg, r.successCount, r.successRate, ...
        r.collisionCount, r.oobCount, r.timeoutCount, ...
        r.meanPosErr, r.meanYawErrDeg);
end

%% Plot success rate vs yaw tolerance
figure('Name','Fast BC Success Rate vs Yaw Tolerance');
plot([sweepResults.yawTolDeg], [sweepResults.successRate], '-o', 'LineWidth', 1.8);
grid on;
xlabel('Yaw tolerance (deg)');
ylabel('Success rate (%)');
title('Fast Behavior Cloning Success Rate vs Yaw Tolerance');

%% Save results
save('bc_parallelpark_yaw_tolerance_sweep_fast.mat', 'sweepResults', 'starts');
disp('Saved bc_parallelpark_yaw_tolerance_sweep_fast.mat');

%% Local functions
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

function a = wrapToPiLocal(a)
    a = mod(a + pi, 2*pi) - pi;
end