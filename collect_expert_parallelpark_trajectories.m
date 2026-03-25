% collect_expert_parallelpark_trajectories.m
clear; clc; close all;

%% =========================
% Collect expert trajectories from successful scripted runs
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

% Success tolerances (same as your succeeded controller)
GoalRadius = 0.10;
GoalYawTol = deg2rad(25);
GoalSpeedTol = 0.06;

% Early stop gate
softStopPosTol = 0.2;
softStopYawTol = deg2rad(20);

%% Controller parameters from succeeded code
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

%% Nearby starts
starts = [
    0.060, 0.200, deg2rad( 0);
    0.030, 0.180, deg2rad(-2);
    0.090, 0.220, deg2rad( 2);
    0.050, 0.170, deg2rad( 3);
    0.080, 0.210, deg2rad(-3);
    0.040, 0.190, deg2rad( 1);
    0.070, 0.230, deg2rad(-1);
    0.020, 0.200, deg2rad( 0);
    0.100, 0.200, deg2rad( 0);
    0.060, 0.160, deg2rad( 0);
    0.060, 0.240, deg2rad( 0);
];

numStarts = size(starts,1);

%% Dataset containers
expertEpisodes = {};
summary = struct([]);
episodeCount = 0;

%% Run all starts
for s = 1:numStarts

    x = starts(s,1);
    y = starts(s,2);
    psi = starts(s,3);
    v = 0.0;
    delta = 0.0;

    phase = "forward_approach";

    % Per-episode logs
    states = [];
    actions = [];
    next_states = [];
    dones = [];
    phases = strings(0,1);

    success = false;
    collision = false;
    outOfBounds = false;

    for k = 1:MaxSteps

        % Build current state BEFORE action
        state = buildState(x, y, psi, v, delta, TargetX, TargetY, TargetYaw);

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

        action = [steerCmd; vCmd];

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

        done = success || collision || outOfBounds;

        next_state = buildState(x, y, psi, v, delta, TargetX, TargetY, TargetYaw);

        % Log transition
        states = [states; state(:)']; %#ok<AGROW>
        actions = [actions; action(:)']; %#ok<AGROW>
        next_states = [next_states; next_state(:)']; %#ok<AGROW>
        dones = [dones; done]; %#ok<AGROW>
        phases = [phases; phase]; %#ok<AGROW>

        if done
            break;
        end
    end

    % Save only successful episodes
    if success
        episodeCount = episodeCount + 1;

        ep = struct();
        ep.start = starts(s,:);
        ep.states = states;
        ep.actions = actions;
        ep.next_states = next_states;
        ep.dones = dones;
        ep.phases = phases;
        ep.length = size(states,1);

        expertEpisodes{end+1} = ep; %#ok<AGROW>
    end

    summary(s).start = starts(s,:);
    summary(s).success = success;
    summary(s).collision = collision;
    summary(s).outOfBounds = outOfBounds;
    summary(s).numTransitions = size(states,1);
end

%% Print summary
fprintf('\n==== Expert Trajectory Collection Summary ====\n');
fprintf('Successful expert episodes collected: %d / %d\n', episodeCount, numStarts);

for s = 1:numStarts
    fprintf('Start %2d: success=%d, collision=%d, outOfBounds=%d, transitions=%d\n', ...
        s, summary(s).success, summary(s).collision, summary(s).outOfBounds, summary(s).numTransitions);
end

%% Build flattened dataset too
all_states = [];
all_actions = [];
all_next_states = [];
all_dones = [];
all_episode_ids = [];
all_phase_ids = strings(0,1);

for i = 1:numel(expertEpisodes)
    ep = expertEpisodes{i};
    n = size(ep.states,1);

    all_states = [all_states; ep.states]; %#ok<AGROW>
    all_actions = [all_actions; ep.actions]; %#ok<AGROW>
    all_next_states = [all_next_states; ep.next_states]; %#ok<AGROW>
    all_dones = [all_dones; ep.dones]; %#ok<AGROW>
    all_episode_ids = [all_episode_ids; i*ones(n,1)]; %#ok<AGROW>
    all_phase_ids = [all_phase_ids; ep.phases]; %#ok<AGROW>
end

expertDataset = struct();
expertDataset.states = all_states;
expertDataset.actions = all_actions;
expertDataset.next_states = all_next_states;
expertDataset.dones = all_dones;
expertDataset.episode_ids = all_episode_ids;
expertDataset.phase_ids = all_phase_ids;

%% Save
save('expert_parallelpark_trajectories.mat', 'expertEpisodes', 'expertDataset', 'summary');
disp('Saved expert_parallelpark_trajectories.mat');

%% Optional: quick plot of successful starts
figure('Name','Successful Expert Start Poses');
hold on; grid on; axis equal;

drawCar(RearCarCenter, 0, ParkedCarLength, ParkedCarWidth, [0.7 0.7 0.7]);
drawCar(FrontCarCenter, 0, ParkedCarLength, ParkedCarWidth, [0.7 0.7 0.7]);
drawCar([TargetX, TargetY], TargetYaw, EgoLength, EgoWidth, [0.2 0.8 0.2]);

plot([Geofence(1), Geofence(2)], [CurbY, CurbY], 'k--', 'LineWidth', 1.5);
rectangle('Position',[-SlotLength/2 CurbY SlotLength SlotDepth], ...
    'EdgeColor',[0 0.5 0], 'LineStyle','--', 'LineWidth',1.5);

for i = 1:numel(expertEpisodes)
    ep = expertEpisodes{i};
    plot(ep.states(:,1), ep.states(:,2), 'b-');
end

xlabel('X (m)');
ylabel('Y (m)');
title('Successful Expert Trajectories');

%% =========================
% Local functions
% =========================
function s = buildState(x, y, psi, v, delta, TargetX, TargetY, TargetYaw)
    dx = TargetX - x;
    dy = TargetY - y;

    c = cos(TargetYaw);
    s2 = sin(TargetYaw);

    d_long =  c*dx + s2*dy;
    d_lat  = -s2*dx + c*dy;
    e_yaw  = wrapToPiLocal(TargetYaw - psi);

    % State representation for imitation later
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