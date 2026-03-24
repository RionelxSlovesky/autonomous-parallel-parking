clear; clc; close all;

%% Load trained agent
load("sac_parallelpark_path_agent.mat","agent");

%% Create environment
env = QCarParallelParkingPathEnv();

[refX, refY] = env.getReferencePath();

%% Publicly reconstruct target + parked car geometry
targetX = 0.0;
targetY = env.CurbY + env.SlotDepth/2;
targetYaw = 0.0;

gapHalf = env.SlotLength / 2;
carHalf = env.ParkedCarLength / 2;
rearCarCenter  = [-(gapHalf + carHalf), targetY];
frontCarCenter = [ (gapHalf + carHalf), targetY];

x1 = -env.SlotLength/2;
y1 = env.CurbY;

%% Run multiple rollouts
numEpisodes = 5;

for ep = 1:numEpisodes
    obs = reset(env);
    done = false;
    stepCount = 0;

    X = [];
    Y = [];
    PSI = [];
    V = [];
    REW = [];
    cteLog = [];
    heLog = [];
    pathIdxLog = [];

    lastInfo = struct();

    while ~done && stepCount < env.MaxSteps
        stepCount = stepCount + 1;

        act = getAction(agent, obs);
        act = forceNumericAction(act);

        [obs, reward, done, info] = step(env, act);

        X(end+1,1) = info.x;
        Y(end+1,1) = info.y;
        PSI(end+1,1) = info.psi;
        V(end+1,1) = info.v;
        REW(end+1,1) = reward;
        cteLog(end+1,1) = info.CrossTrackError;
        heLog(end+1,1) = info.HeadingError;
        pathIdxLog(end+1,1) = info.PathIndex;

        lastInfo = info;
    end

    fprintf('\n==== Episode %d ====\n', ep);
    fprintf('Steps: %d\n', stepCount);
    fprintf('Success: %d\n', lastInfo.Success);
    fprintf('Collision: %d\n', lastInfo.Collision);
    fprintf('OutOfBounds: %d\n', lastInfo.OutOfBounds);
    fprintf('Final CrossTrackError: %.4f\n', lastInfo.CrossTrackError);
    fprintf('Final HeadingError: %.4f\n', lastInfo.HeadingError);
    fprintf('Total reward: %.4f\n', sum(REW));

    %% Figure 1: trajectory
    figure('Name', sprintf('Path Tracking Trajectory Episode %d', ep));
    hold on; grid on; axis equal;

    drawCar(rearCarCenter, 0, env.ParkedCarLength, env.ParkedCarWidth, [0.7 0.7 0.7]);
    drawCar(frontCarCenter, 0, env.ParkedCarLength, env.ParkedCarWidth, [0.7 0.7 0.7]);
    drawCar([targetX, targetY], targetYaw, env.EgoLength, env.EgoWidth, [0.2 0.8 0.2]);

    plot(refX, refY, 'm--', 'LineWidth', 2);
    plot(X, Y, 'b.-', 'LineWidth', 1.5, 'MarkerSize', 10);

    plot([env.Geofence(1), env.Geofence(2)], [env.CurbY, env.CurbY], 'k--', 'LineWidth', 1.5);
    rectangle('Position',[x1 y1 env.SlotLength env.SlotDepth], ...
        'EdgeColor',[0 0.5 0], 'LineStyle','--', 'LineWidth',1.5);

    plot(X(1), Y(1), 'go', 'MarkerSize', 10, 'LineWidth', 2);
    plot(X(end), Y(end), 'ro', 'MarkerSize', 10, 'LineWidth', 2);

    xlabel('X (m)');
    ylabel('Y (m)');
    title(sprintf('Path Tracking Trajectory Episode %d', ep));

    %% Figure 2: errors
    figure('Name', sprintf('Tracking Errors Episode %d', ep));
    plot(cteLog, 'LineWidth', 1.5); hold on;
    plot(heLog, 'LineWidth', 1.5);
    grid on;
    xlabel('Step');
    ylabel('Error');
    legend('CrossTrackError','HeadingError');
    title(sprintf('Tracking Errors Episode %d', ep));

    %% Figure 3: reward
    figure('Name', sprintf('Reward Episode %d', ep));
    plot(REW, 'LineWidth', 1.5);
    grid on;
    xlabel('Step');
    ylabel('Reward');
    title(sprintf('Step Reward Episode %d', ep));

    %% Figure 4: path index progression
    figure('Name', sprintf('Path Index Episode %d', ep));
    plot(pathIdxLog, 'LineWidth', 1.5);
    grid on;
    xlabel('Step');
    ylabel('Path Index');
    title(sprintf('Path Progression Episode %d', ep));

    %% Figure 5: animation
    figure('Name', sprintf('Animation Episode %d', ep));
    hold on; grid on; axis equal;
    xlim([env.Geofence(1), env.Geofence(2)]);
    ylim([env.Geofence(3), env.Geofence(4)]);
    xlabel('X (m)');
    ylabel('Y (m)');
    title(sprintf('Parking Path Tracking Animation Episode %d', ep));

    drawCar(rearCarCenter, 0, env.ParkedCarLength, env.ParkedCarWidth, [0.7 0.7 0.7]);
    drawCar(frontCarCenter, 0, env.ParkedCarLength, env.ParkedCarWidth, [0.7 0.7 0.7]);
    drawCar([targetX, targetY], targetYaw, env.EgoLength, env.EgoWidth, [0.2 0.8 0.2]);

    plot(refX, refY, 'm--', 'LineWidth', 2);
    plot([env.Geofence(1), env.Geofence(2)], [env.CurbY, env.CurbY], 'k--', 'LineWidth', 1.5);
    rectangle('Position',[x1 y1 env.SlotLength env.SlotDepth], ...
        'EdgeColor',[0 0.5 0], 'LineStyle','--', 'LineWidth',1.5);

    hTrail = plot(nan, nan, 'b-', 'LineWidth', 1.5);
    hCar = patch(nan, nan, [0.2 0.4 1.0], 'FaceAlpha', 0.5);

    for k = 1:length(X)
        set(hTrail, 'XData', X(1:k), 'YData', Y(1:k));
        [px, py] = carPolygon([X(k), Y(k)], PSI(k), env.EgoLength, env.EgoWidth);
        set(hCar, 'XData', px, 'YData', py);
        drawnow;
        pause(0.03);
    end
end

%% Helpers
function a = forceNumericAction(a)
    if iscell(a)
        a = a{1};
    end
    if isa(a, 'dlarray')
        a = extractdata(a);
    end
    a = double(a(:));
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