% evaluate_and_compare_bc_vs_sac_final_alignment.m
clear; clc; close all;

%% ============================================================
% Evaluate and compare:
%   1) BC baseline policy
%   2) SAC final-alignment-only policy
%
% Loads:
%   - bc_parallelpark_policy.mat
%   - sac_final_alignment_only.mat
%
% Runs:
%   A) randomized local evaluation
%   B) yaw-tolerance sweep
%
% Output:
%   - printed comparison tables
%   - plots
%   - saved comparison file
%
% Requirements:
%   - helper files on MATLAB path:
%       carPolygon.m
%       projectPolygon.m
%       polygonsOverlapSAT.m
%       egoHitsCurb.m
%       checkParkingCollision.m
%% ============================================================

%% ----------------------------
% Load policies
% ----------------------------
load('bc_parallelpark_policy.mat', 'bcPolicy');
load('sac_final_alignment_only.mat', 'agent', 'cfg');

%% ----------------------------
% Shared evaluation settings
% ----------------------------
rng(42);

numTests = 50;
MaxSteps = 300;          % use alignment-only horizon for fair comparison

% We evaluate near-target alignment region only
xRange = [-0.10 0.10];
yRange = [cfg.TargetY - 0.08, cfg.TargetY + 0.08];
psiRangeDeg = [-20 20];

starts = zeros(numTests,3);
for i = 1:numTests
    starts(i,1) = xRange(1) + (xRange(2)-xRange(1))*rand;
    starts(i,2) = yRange(1) + (yRange(2)-yRange(1))*rand;
    starts(i,3) = deg2rad(psiRangeDeg(1) + (psiRangeDeg(2)-psiRangeDeg(1))*rand);
end

GoalRadius = 0.08;
GoalSpeedTol = 0.04;

yawTolDegList = [20 15 12 10];

%% ============================================================
% PART 1: Randomized local evaluation at 15 deg
%% ============================================================
mainYawTolDeg = 15;
mainYawTol = deg2rad(mainYawTolDeg);

bcResultsMain  = runEvalBC(bcPolicy, cfg, starts, GoalRadius, mainYawTol, GoalSpeedTol, MaxSteps);
sacResultsMain = runEvalSAC(agent,   cfg, starts, GoalRadius, mainYawTol, GoalSpeedTol, MaxSteps);

printEvalSummary(sprintf('BC Randomized Local @ %d deg', mainYawTolDeg), bcResultsMain);
printEvalSummary(sprintf('SAC Randomized Local @ %d deg', mainYawTolDeg), sacResultsMain);

plotCompareTrajectories(bcResultsMain, sacResultsMain, cfg, ...
    sprintf('BC vs SAC Randomized Local @ %d deg', mainYawTolDeg));

%% ============================================================
% PART 2: Yaw-tolerance sweep
%% ============================================================
bcSweep  = repmat(makeSweepStruct(), numel(yawTolDegList), 1);
sacSweep = repmat(makeSweepStruct(), numel(yawTolDegList), 1);

for t = 1:numel(yawTolDegList)
    yawTolDeg = yawTolDegList(t);
    yawTol = deg2rad(yawTolDeg);

    bcRes  = runEvalBC(bcPolicy, cfg, starts, GoalRadius, yawTol, GoalSpeedTol, MaxSteps);
    sacRes = runEvalSAC(agent,   cfg, starts, GoalRadius, yawTol, GoalSpeedTol, MaxSteps);

    bcSweep(t)  = summarizeSweep(yawTolDeg, bcRes);
    sacSweep(t) = summarizeSweep(yawTolDeg, sacRes);
end

%% Print sweep comparison
fprintf('\n==== BC vs SAC Final-Alignment Comparison: Yaw Tolerance Sweep ====\n');
fprintf('YawTol   BC_Succ  BC_Rate  BC_Coll  BC_Tout   SAC_Succ  SAC_Rate  SAC_Coll  SAC_Tout\n');
for t = 1:numel(yawTolDegList)
    b = bcSweep(t);
    s = sacSweep(t);
    fprintf('%6.1f   %7d  %7.2f  %7d  %7d   %8d  %8.2f  %8d  %8d\n', ...
        b.yawTolDeg, b.successCount, b.successRate, b.collisionCount, b.timeoutCount, ...
        s.successCount, s.successRate, s.collisionCount, s.timeoutCount);
end

%% Plot success-rate comparison
figure('Name','BC vs SAC Success Rate vs Yaw Tolerance');
plot([bcSweep.yawTolDeg],  [bcSweep.successRate],  '-o', 'LineWidth', 1.8); hold on;
plot([sacSweep.yawTolDeg], [sacSweep.successRate], '-s', 'LineWidth', 1.8);
grid on;
xlabel('Yaw tolerance (deg)');
ylabel('Success rate (%)');
title('BC vs SAC Final-Alignment Comparison');
legend('BC','SAC','Location','best');

%% Plot mean yaw error comparison
figure('Name','BC vs SAC Mean Yaw Error');
plot([bcSweep.yawTolDeg],  [bcSweep.meanYawErrDeg],  '-o', 'LineWidth', 1.8); hold on;
plot([sacSweep.yawTolDeg], [sacSweep.meanYawErrDeg], '-s', 'LineWidth', 1.8);
grid on;
xlabel('Yaw tolerance (deg)');
ylabel('Mean final yaw error (deg)');
title('BC vs SAC Mean Final Yaw Error');
legend('BC','SAC','Location','best');

%% Plot mean position error comparison
figure('Name','BC vs SAC Mean Position Error');
plot([bcSweep.yawTolDeg],  [bcSweep.meanPosErr],  '-o', 'LineWidth', 1.8); hold on;
plot([sacSweep.yawTolDeg], [sacSweep.meanPosErr], '-s', 'LineWidth', 1.8);
grid on;
xlabel('Yaw tolerance (deg)');
ylabel('Mean final position error');
title('BC vs SAC Mean Final Position Error');
legend('BC','SAC','Location','best');

%% Best-run plots at 15 deg
plotBestRun(bcResultsMain,  cfg, sprintf('Best BC Run @ %d deg', mainYawTolDeg));
plotBestRun(sacResultsMain, cfg, sprintf('Best SAC Run @ %d deg', mainYawTolDeg));

%% Save
save('bc_vs_sac_final_alignment_comparison.mat', ...
    'starts', 'bcResultsMain', 'sacResultsMain', 'bcSweep', 'sacSweep');
disp('Saved bc_vs_sac_final_alignment_comparison.mat');

%% ============================================================
% Local functions
%% ============================================================

function s = makeSweepStruct()
    s = struct( ...
        'yawTolDeg', [], ...
        'successCount', [], ...
        'successRate', [], ...
        'collisionCount', [], ...
        'oobCount', [], ...
        'timeoutCount', [], ...
        'meanPosErr', [], ...
        'medianPosErr', [], ...
        'meanYawErrDeg', [], ...
        'medianYawErrDeg', []);
end

function s = summarizeSweep(yawTolDeg, results)
    s = makeSweepStruct();
    s.yawTolDeg = yawTolDeg;
    s.successCount = sum([results.success]);
    s.successRate = 100 * s.successCount / numel(results);
    s.collisionCount = sum([results.collision]);
    s.oobCount = sum([results.outOfBounds]);
    s.timeoutCount = sum([results.timeout]);
    s.meanPosErr = mean([results.finalPosErr]);
    s.medianPosErr = median([results.finalPosErr]);
    s.meanYawErrDeg = mean(rad2deg([results.finalYawErr]));
    s.medianYawErrDeg = median(rad2deg([results.finalYawErr]));
end

function results = makeResultStruct(n)
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
        'score', []), n, 1);
end

function results = runEvalBC(bcPolicy, cfg, starts, GoalRadius, GoalYawTol, GoalSpeedTol, MaxSteps)
    n = size(starts,1);
    results = makeResultStruct(n);

    for s = 1:n
        x = starts(s,1); y = starts(s,2); psi = starts(s,3);
        v = 0.0; delta = 0.0;

        X = nan(MaxSteps,1); Y = nan(MaxSteps,1); PSI = nan(MaxSteps,1);
        V = nan(MaxSteps,1); DELTA = nan(MaxSteps,1);
        STEER_CMD = nan(MaxSteps,1); VCMD = nan(MaxSteps,1);

        success = false; collision = false; outOfBounds = false; timeout = false;

        for k = 1:MaxSteps
            obs = buildState(x, y, psi, v, delta, cfg.TargetX, cfg.TargetY, cfg.TargetYaw);
            action = bc_policy_action(obs, bcPolicy);

            steerCmd = min(max(action(1), cfg.SteerMin), cfg.SteerMax);
            vCmd     = min(max(action(2), -cfg.Vmax), cfg.Vmax);

            delta = delta + (cfg.Ts/cfg.TauSteer) * (steerCmd - delta);
            v = 0.85*v + 0.15*vCmd;

            x = x + cfg.Ts * v * cos(psi);
            y = y + cfg.Ts * v * sin(psi);
            psi = wrapToPiLocal(psi + cfg.Ts * (v/cfg.Wheelbase) * tan(delta));

            [collision, ~, ~, ~, ~, ~, ~] = ...
                checkParkingCollision([x, y], psi, cfg.EgoLength, cfg.EgoWidth, ...
                                      cfg.FrontCarCenter, cfg.RearCarCenter, ...
                                      cfg.ParkedCarLength, cfg.ParkedCarWidth, cfg.CurbY);

            outOfBounds = (x < cfg.Geofence(1)) || (x > cfg.Geofence(2)) || ...
                          (y < cfg.Geofence(3)) || (y > cfg.Geofence(4));

            posErr = hypot(x - cfg.TargetX, y - cfg.TargetY);
            yawErr = abs(wrapToPiLocal(cfg.TargetYaw - psi));

            success = (posErr < GoalRadius) && ...
                      (yawErr < GoalYawTol) && ...
                      (abs(v) < GoalSpeedTol);

            X(k)=x; Y(k)=y; PSI(k)=psi; V(k)=v; DELTA(k)=delta;
            STEER_CMD(k)=steerCmd; VCMD(k)=vCmd;

            if success || collision || outOfBounds
                X = X(1:k); Y = Y(1:k); PSI = PSI(1:k); V = V(1:k);
                DELTA = DELTA(1:k); STEER_CMD = STEER_CMD(1:k); VCMD = VCMD(1:k);
                break;
            end
        end

        if ~(success || collision || outOfBounds)
            timeout = true;
            X = X(1:MaxSteps); Y = Y(1:MaxSteps); PSI = PSI(1:MaxSteps); V = V(1:MaxSteps);
            DELTA = DELTA(1:MaxSteps); STEER_CMD = STEER_CMD(1:MaxSteps); VCMD = VCMD(1:MaxSteps);
        end

        results(s) = packResult(starts(s,:), success, collision, outOfBounds, timeout, ...
            X, Y, PSI, V, DELTA, STEER_CMD, VCMD, cfg);
    end
end

function results = runEvalSAC(agent, cfg, starts, GoalRadius, GoalYawTol, GoalSpeedTol, MaxSteps)
    n = size(starts,1);
    results = makeResultStruct(n);

    for s = 1:n
        x = starts(s,1); y = starts(s,2); psi = starts(s,3);
        v = 0.0; delta = 0.0;

        X = nan(MaxSteps,1); Y = nan(MaxSteps,1); PSI = nan(MaxSteps,1);
        V = nan(MaxSteps,1); DELTA = nan(MaxSteps,1);
        STEER_CMD = nan(MaxSteps,1); VCMD = nan(MaxSteps,1);

        success = false; collision = false; outOfBounds = false; timeout = false;

        for k = 1:MaxSteps
            obs = buildState(x, y, psi, v, delta, cfg.TargetX, cfg.TargetY, cfg.TargetYaw);

            action = getAction(agent, {obs});
            if iscell(action), action = action{1}; end
            action = squeeze(action);

            steerCmd = min(max(action(1), cfg.SteerMin), cfg.SteerMax);
            vCmd     = min(max(action(2), -cfg.Vmax), cfg.Vmax);

            delta = delta + (cfg.Ts/cfg.TauSteer) * (steerCmd - delta);
            v = 0.85*v + 0.15*vCmd;

            x = x + cfg.Ts * v * cos(psi);
            y = y + cfg.Ts * v * sin(psi);
            psi = wrapToPiLocal(psi + cfg.Ts * (v/cfg.Wheelbase) * tan(delta));

            [collision, ~, ~, ~, ~, ~, ~] = ...
                checkParkingCollision([x, y], psi, cfg.EgoLength, cfg.EgoWidth, ...
                                      cfg.FrontCarCenter, cfg.RearCarCenter, ...
                                      cfg.ParkedCarLength, cfg.ParkedCarWidth, cfg.CurbY);

            outOfBounds = (x < cfg.Geofence(1)) || (x > cfg.Geofence(2)) || ...
                          (y < cfg.Geofence(3)) || (y > cfg.Geofence(4));

            posErr = hypot(x - cfg.TargetX, y - cfg.TargetY);
            yawErr = abs(wrapToPiLocal(cfg.TargetYaw - psi));

            success = (posErr < GoalRadius) && ...
                      (yawErr < GoalYawTol) && ...
                      (abs(v) < GoalSpeedTol);

            X(k)=x; Y(k)=y; PSI(k)=psi; V(k)=v; DELTA(k)=delta;
            STEER_CMD(k)=steerCmd; VCMD(k)=vCmd;

            if success || collision || outOfBounds
                X = X(1:k); Y = Y(1:k); PSI = PSI(1:k); V = V(1:k);
                DELTA = DELTA(1:k); STEER_CMD = STEER_CMD(1:k); VCMD = VCMD(1:k);
                break;
            end
        end

        if ~(success || collision || outOfBounds)
            timeout = true;
            X = X(1:MaxSteps); Y = Y(1:MaxSteps); PSI = PSI(1:MaxSteps); V = V(1:MaxSteps);
            DELTA = DELTA(1:MaxSteps); STEER_CMD = STEER_CMD(1:MaxSteps); VCMD = VCMD(1:MaxSteps);
        end

        results(s) = packResult(starts(s,:), success, collision, outOfBounds, timeout, ...
            X, Y, PSI, V, DELTA, STEER_CMD, VCMD, cfg);
    end
end

function r = packResult(startPose, success, collision, outOfBounds, timeout, X, Y, PSI, V, DELTA, STEER_CMD, VCMD, cfg)
    finalPosErr = hypot(X(end) - cfg.TargetX, Y(end) - cfg.TargetY);
    finalYawErr = abs(wrapToPiLocal(cfg.TargetYaw - PSI(end)));

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

    r = struct();
    r.start = startPose;
    r.success = success;
    r.collision = collision;
    r.outOfBounds = outOfBounds;
    r.timeout = timeout;
    r.finalPosErr = finalPosErr;
    r.finalYawErr = finalYawErr;
    r.steps = numel(X);
    r.X = X;
    r.Y = Y;
    r.PSI = PSI;
    r.V = V;
    r.DELTA = DELTA;
    r.STEER_CMD = STEER_CMD;
    r.VCMD = VCMD;
    r.score = score;
end

function printEvalSummary(titleStr, results)
    fprintf('\n==== %s ====\n', titleStr);

    succ = sum([results.success]);
    coll = sum([results.collision]);
    oob  = sum([results.outOfBounds]);
    tout = sum([results.timeout]);
    pos  = [results.finalPosErr];
    yawd = rad2deg([results.finalYawErr]);

    fprintf('Successes     : %d / %d\n', succ, numel(results));
    fprintf('Collisions    : %d\n', coll);
    fprintf('OutOfBounds   : %d\n', oob);
    fprintf('Timeouts      : %d\n', tout);
    fprintf('Success rate  : %.2f %%\n', 100*succ/numel(results));
    fprintf('Mean pos err  : %.4f\n', mean(pos));
    fprintf('Median pos err: %.4f\n', median(pos));
    fprintf('Mean yaw err  : %.2f deg\n', mean(yawd));
    fprintf('Median yaw err: %.2f deg\n', median(yawd));
end

function plotCompareTrajectories(bcResults, sacResults, cfg, figName)
    figure('Name', figName);

    subplot(1,2,1); hold on; grid on; axis equal;
    baseScene(cfg);
    for s = 1:numel(bcResults)
        plotResultTraj(bcResults(s));
    end
    title('BC');

    subplot(1,2,2); hold on; grid on; axis equal;
    baseScene(cfg);
    for s = 1:numel(sacResults)
        plotResultTraj(sacResults(s));
    end
    title('SAC');

    sgtitle(figName);
end

function plotResultTraj(r)
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

function plotBestRun(results, cfg, figName)
    [~, bestIdx] = min([results.score]);
    r = results(bestIdx);

    figure('Name', figName); hold on; grid on; axis equal;
    baseScene(cfg);

    plot(r.X, r.Y, 'b.-', 'LineWidth', 1.5, 'MarkerSize', 10);
    plot(r.X(1), r.Y(1), 'go', 'MarkerSize', 8, 'LineWidth', 1.5);
    plot(r.X(end), r.Y(end), 'ro', 'MarkerSize', 8, 'LineWidth', 1.5);

    xlabel('X (m)');
    ylabel('Y (m)');
    title(figName);
end

function plotStartOutcomeMap(results, figName)
    figure('Name', figName); hold on; grid on; axis equal;
    for s = 1:numel(results)
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
    title(figName);
    legend('Success','Collision','OutOfBounds','Timeout','Location','best');
end

function baseScene(cfg)
    drawCar(cfg.RearCarCenter, 0, cfg.ParkedCarLength, cfg.ParkedCarWidth, [0.7 0.7 0.7]);
    drawCar(cfg.FrontCarCenter, 0, cfg.ParkedCarLength, cfg.ParkedCarWidth, [0.7 0.7 0.7]);
    drawCar([cfg.TargetX, cfg.TargetY], cfg.TargetYaw, cfg.EgoLength, cfg.EgoWidth, [0.2 0.8 0.2]);
    plot([cfg.Geofence(1), cfg.Geofence(2)], [cfg.CurbY, cfg.CurbY], 'k--', 'LineWidth', 1.5);
    rectangle('Position',[-cfg.SlotLength/2 cfg.CurbY cfg.SlotLength cfg.SlotDepth], ...
        'EdgeColor',[0 0.5 0], 'LineStyle','--', 'LineWidth',1.5);
    xlabel('X (m)');
    ylabel('Y (m)');
end

function action = bc_policy_action(state, bcPolicy)
    state = state(:)';
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