% train_rl_final_alignment_only.m
clear; clc;

%% ============================================================
% RL fine-tuning for FINAL ALIGNMENT ONLY
% ============================================================
% Purpose:
%   Train RL only on the last part of parking:
%   - car starts near the slot / near target
%   - RL improves final yaw + small position error
%   - BC already solved the full maneuver, so RL only refines alignment
%
% Output:
%   sac_final_alignment_only.mat
%
% Requirements:
%   - Reinforcement Learning Toolbox
%   - Deep Learning Toolbox
%   - helper files on path:
%       carPolygon.m
%       projectPolygon.m
%       polygonsOverlapSAT.m
%       egoHitsCurb.m
%       checkParkingCollision.m

%% ----------------------------
% Specs
% ----------------------------
obsDim = 8;
actDim = 2;

obsInfo = rlNumericSpec([obsDim 1], ...
    'LowerLimit', -inf(obsDim,1), ...
    'UpperLimit',  inf(obsDim,1));
obsInfo.Name = "obs";

actInfo = rlNumericSpec([actDim 1], ...
    'LowerLimit', [-0.5; -0.20], ...
    'UpperLimit', [ 0.5;  0.20]);
actInfo.Name = "action";

%% ----------------------------
% Environment config
% ----------------------------
cfg = makeEnvConfig();

env = rlFunctionEnv(obsInfo, actInfo, ...
    @(action,loggedSignals) stepFcn(action, loggedSignals, cfg), ...
    @() resetFcn(cfg));

%% ----------------------------
% Actor
% ----------------------------
actorLG = buildSimpleActor(obsDim, actDim);

actor = rlContinuousGaussianActor(actorLG, obsInfo, actInfo, ...
    "ActionMeanOutputNames","mean", ...
    "ActionStandardDeviationOutputNames","std");

%% ----------------------------
% Critics
% ----------------------------
critic1 = makeCritic(obsDim, actDim, obsInfo, actInfo, "critic1");
critic2 = makeCritic(obsDim, actDim, obsInfo, actInfo, "critic2");

%% ----------------------------
% SAC options
% ----------------------------
agentOpts = rlSACAgentOptions;
agentOpts.SampleTime = cfg.Ts;
agentOpts.DiscountFactor = 0.99;
agentOpts.MiniBatchSize = 128;
agentOpts.ExperienceBufferLength = 1e5;
agentOpts.TargetSmoothFactor = 5e-3;

agent = rlSACAgent(actor, [critic1 critic2], agentOpts);

%% ----------------------------
% Training options
% ----------------------------
trainOpts = rlTrainingOptions;
trainOpts.MaxEpisodes = 400;
trainOpts.MaxStepsPerEpisode = cfg.MaxSteps;
trainOpts.ScoreAveragingWindowLength = 20;
trainOpts.Plots = "training-progress";
trainOpts.Verbose = true;
trainOpts.StopTrainingCriteria = "EpisodeCount";
trainOpts.StopTrainingValue = trainOpts.MaxEpisodes;

%% ----------------------------
% Train
% ----------------------------
trainingStats = train(agent, env, trainOpts);

%% ----------------------------
% Save
% ----------------------------
save('sac_final_alignment_only.mat', 'agent', 'trainingStats', 'cfg');
disp('Saved sac_final_alignment_only.mat');

%% ============================================================
% Local functions
% ============================================================

function cfg = makeEnvConfig()
    cfg.Ts = 0.02;
    cfg.MaxSteps = 300;

    cfg.Wheelbase = 0.256;
    cfg.SteerMin = -0.5;
    cfg.SteerMax =  0.5;
    cfg.TauSteer = 0.16;
    cfg.Vmax = 0.20;

    cfg.EgoLength = 0.32;
    cfg.EgoWidth  = 0.18;

    cfg.ParkedCarLength = 0.36;
    cfg.ParkedCarWidth  = 0.20;

    cfg.SlotLength = 0.90;
    cfg.SlotDepth  = 0.36;

    cfg.Geofence = [-1.5 1.5 -0.8 1.0];
    cfg.CurbY = -0.30;

    cfg.TargetX = 0.0;
    cfg.TargetY = cfg.CurbY + cfg.SlotDepth/2;
    cfg.TargetYaw = 0.0;

    gapHalf = cfg.SlotLength / 2;
    carHalf = cfg.ParkedCarLength / 2;
    cfg.FrontCarCenter = [ gapHalf + carHalf, cfg.TargetY ];
    cfg.RearCarCenter  = [-gapHalf - carHalf, cfg.TargetY ];

    % Start near target only
    cfg.xRange = [-0.10 0.10];
    cfg.yRange = [cfg.TargetY - 0.08, cfg.TargetY + 0.08];
    cfg.psiRange = deg2rad([-20 20]);

    % Tighter final goal than BC's coarse success
    cfg.GoalRadius = 0.08;
    cfg.GoalYawTol = deg2rad(15);
    cfg.GoalSpeedTol = 0.04;
end

function [initialObservation, loggedSignals] = resetFcn(cfg)
    loggedSignals = struct();
    loggedSignals.StepCount = 0;

    loggedSignals.x = cfg.xRange(1) + (cfg.xRange(2)-cfg.xRange(1))*rand;
    loggedSignals.y = cfg.yRange(1) + (cfg.yRange(2)-cfg.yRange(1))*rand;
    loggedSignals.psi = cfg.psiRange(1) + (cfg.psiRange(2)-cfg.psiRange(1))*rand;

    loggedSignals.v = 0.0;
    loggedSignals.delta = 0.0;

    loggedSignals.prevPosErr = hypot(loggedSignals.x - cfg.TargetX, ...
                                     loggedSignals.y - cfg.TargetY);
    loggedSignals.prevYawErr = abs(wrapToPiLocal(cfg.TargetYaw - loggedSignals.psi));

    initialObservation = buildState( ...
        loggedSignals.x, loggedSignals.y, loggedSignals.psi, ...
        loggedSignals.v, loggedSignals.delta, ...
        cfg.TargetX, cfg.TargetY, cfg.TargetYaw);

    loggedSignals.Observation = initialObservation;
end

function [nextObs,reward,isDone,loggedSignals] = stepFcn(action, loggedSignals, cfg)
    loggedSignals.StepCount = loggedSignals.StepCount + 1;

    steerCmd = min(max(action(1), cfg.SteerMin), cfg.SteerMax);
    vCmd     = min(max(action(2), -cfg.Vmax), cfg.Vmax);

    % Dynamics
    loggedSignals.delta = loggedSignals.delta + ...
        (cfg.Ts/cfg.TauSteer) * (steerCmd - loggedSignals.delta);

    loggedSignals.v = 0.85*loggedSignals.v + 0.15*vCmd;

    loggedSignals.x = loggedSignals.x + cfg.Ts * loggedSignals.v * cos(loggedSignals.psi);
    loggedSignals.y = loggedSignals.y + cfg.Ts * loggedSignals.v * sin(loggedSignals.psi);
    loggedSignals.psi = wrapToPiLocal(loggedSignals.psi + ...
        cfg.Ts * (loggedSignals.v/cfg.Wheelbase) * tan(loggedSignals.delta));

    nextObs = buildState( ...
        loggedSignals.x, loggedSignals.y, loggedSignals.psi, ...
        loggedSignals.v, loggedSignals.delta, ...
        cfg.TargetX, cfg.TargetY, cfg.TargetYaw);

    posErr = hypot(loggedSignals.x - cfg.TargetX, loggedSignals.y - cfg.TargetY);
    yawErr = abs(wrapToPiLocal(cfg.TargetYaw - loggedSignals.psi));

    [collision, ~, ~, ~, ~, ~, ~] = ...
        checkParkingCollision([loggedSignals.x, loggedSignals.y], loggedSignals.psi, ...
                              cfg.EgoLength, cfg.EgoWidth, ...
                              cfg.FrontCarCenter, cfg.RearCarCenter, ...
                              cfg.ParkedCarLength, cfg.ParkedCarWidth, cfg.CurbY);

    outOfBounds = (loggedSignals.x < cfg.Geofence(1)) || ...
                  (loggedSignals.x > cfg.Geofence(2)) || ...
                  (loggedSignals.y < cfg.Geofence(3)) || ...
                  (loggedSignals.y > cfg.Geofence(4));

    success = (posErr < cfg.GoalRadius) && ...
              (yawErr < cfg.GoalYawTol) && ...
              (abs(loggedSignals.v) < cfg.GoalSpeedTol);

    timeout = loggedSignals.StepCount >= cfg.MaxSteps;

    % ---------------- Reward: SIMPLIFIED ----------------
    posProgress = loggedSignals.prevPosErr - posErr;
    yawProgress = loggedSignals.prevYawErr - yawErr;

    reward = 0;

    % Main focus: final alignment
    reward = reward + 6*posProgress;
    reward = reward + 12*yawProgress;

    % Penalize residual error
    reward = reward - 0.6*posErr;
    reward = reward - 1.5*yawErr;

    % Small control penalty
    reward = reward - 0.02*(steerCmd^2);
    reward = reward - 0.01*(vCmd^2);

    % Big penalties for unsafe behavior
    if collision
        reward = reward - 80;
    end
    if outOfBounds
        reward = reward - 60;
    end
    if timeout
        reward = reward - 10;
    end

    % Strong success bonus
    if success
        reward = reward + 50;
    end

    isDone = success || collision || outOfBounds || timeout;

    loggedSignals.prevPosErr = posErr;
    loggedSignals.prevYawErr = yawErr;
    loggedSignals.Observation = nextObs;
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

function lg = buildSimpleActor(obsDim, actDim)
    lg = layerGraph();

    lg = addLayers(lg, featureInputLayer(obsDim,'Name','state','Normalization','none'));
    lg = addLayers(lg, fullyConnectedLayer(128,'Name','fc1'));
    lg = addLayers(lg, reluLayer('Name','relu1'));
    lg = addLayers(lg, fullyConnectedLayer(128,'Name','fc2'));
    lg = addLayers(lg, reluLayer('Name','relu2'));
    lg = addLayers(lg, fullyConnectedLayer(64,'Name','fc3'));
    lg = addLayers(lg, reluLayer('Name','relu3'));

    meanHead = fullyConnectedLayer(actDim,'Name','mean');

    std_fc = fullyConnectedLayer(64,'Name','std_fc');
    std_out = fullyConnectedLayer(actDim,'Name','std_raw');

    % small initial std
    std_fc.Weights = 0.01*randn(64,64);
    std_fc.Bias = zeros(64,1);
    std_out.Weights = 0.01*randn(actDim,64);
    std_out.Bias = -2*ones(actDim,1);

    lg = addLayers(lg, meanHead);
    lg = addLayers(lg, std_fc);
    lg = addLayers(lg, reluLayer('Name','std_relu'));
    lg = addLayers(lg, std_out);
    lg = addLayers(lg, softplusLayer('Name','std'));

    lg = connectLayers(lg,'state','fc1');
    lg = connectLayers(lg,'fc1','relu1');
    lg = connectLayers(lg,'relu1','fc2');
    lg = connectLayers(lg,'fc2','relu2');
    lg = connectLayers(lg,'relu2','fc3');
    lg = connectLayers(lg,'fc3','relu3');
    lg = connectLayers(lg,'relu3','mean');

    lg = connectLayers(lg,'relu3','std_fc');
    lg = connectLayers(lg,'std_fc','std_relu');
    lg = connectLayers(lg,'std_relu','std_raw');
    lg = connectLayers(lg,'std_raw','std');
end

function critic = makeCritic(obsDim, actDim, obsInfo, actInfo, prefix)
    lg = layerGraph();

    obsIn = featureInputLayer(obsDim,'Name',prefix+"_obs",'Normalization','none');
    actIn = featureInputLayer(actDim,'Name',prefix+"_act",'Normalization','none');
    concat = concatenationLayer(1,2,'Name',prefix+"_concat");

    trunk = [
        fullyConnectedLayer(128,'Name',prefix+"_fc1")
        reluLayer('Name',prefix+"_relu1")
        fullyConnectedLayer(128,'Name',prefix+"_fc2")
        reluLayer('Name',prefix+"_relu2")
        fullyConnectedLayer(64,'Name',prefix+"_fc3")
        reluLayer('Name',prefix+"_relu3")
        fullyConnectedLayer(1,'Name',prefix+"_q")
    ];

    lg = addLayers(lg, obsIn);
    lg = addLayers(lg, actIn);
    lg = addLayers(lg, concat);
    lg = addLayers(lg, trunk);

    lg = connectLayers(lg, prefix+"_obs", prefix+"_concat/in1");
    lg = connectLayers(lg, prefix+"_act", prefix+"_concat/in2");
    lg = connectLayers(lg, prefix+"_concat", prefix+"_fc1");

    critic = rlQValueFunction(lg, obsInfo, actInfo, ...
        "ObservationInputNames", prefix+"_obs", ...
        "ActionInputNames", prefix+"_act");
end

function a = wrapToPiLocal(a)
    a = mod(a + pi, 2*pi) - pi;
end