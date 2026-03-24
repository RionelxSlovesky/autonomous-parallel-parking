clear; clc;

%% Create environment
env = QCarParallelParkingPathEnv();

obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);

numObs = obsInfo.Dimension(1);
numAct = actInfo.Dimension(1);

%% =========================
% Build Actor (Gaussian)
% =========================
lgA = layerGraph();

lgA = addLayers(lgA, featureInputLayer(numObs, "Name","obs"));
lgA = addLayers(lgA, fullyConnectedLayer(128, "Name","fc1"));
lgA = addLayers(lgA, reluLayer("Name","relu1"));
lgA = addLayers(lgA, fullyConnectedLayer(128, "Name","fc2"));
lgA = addLayers(lgA, reluLayer("Name","relu2"));

meanHead = [
    fullyConnectedLayer(64, "Name","mean_fc1")
    reluLayer("Name","mean_relu1")
    fullyConnectedLayer(numAct, "Name","mean")
];

useSoftplus = exist("softplusLayer","file") == 2;
if useSoftplus
    stdHead = [
        fullyConnectedLayer(64, "Name","std_fc1")
        reluLayer("Name","std_relu1")
        fullyConnectedLayer(numAct, "Name","std_raw")
        softplusLayer("Name","std")
    ];
else
    stdHead = [
        fullyConnectedLayer(64, "Name","std_fc1")
        reluLayer("Name","std_relu1")
        fullyConnectedLayer(numAct, "Name","std_raw")
        reluLayer("Name","std")
    ];
end

lgA = addLayers(lgA, meanHead);
lgA = addLayers(lgA, stdHead);

lgA = connectLayers(lgA, "obs", "fc1");
lgA = connectLayers(lgA, "fc1", "relu1");
lgA = connectLayers(lgA, "relu1", "fc2");
lgA = connectLayers(lgA, "fc2", "relu2");
lgA = connectLayers(lgA, "relu2", "mean_fc1");
lgA = connectLayers(lgA, "relu2", "std_fc1");

actor = rlContinuousGaussianActor(lgA, obsInfo, actInfo, ...
    "ActionMeanOutputNames", "mean", ...
    "ActionStandardDeviationOutputNames", "std");

%% =========================
% Build Critics (Twin Q)
% =========================
critic1 = makeCritic(numObs, numAct, obsInfo, actInfo, "critic1");
critic2 = makeCritic(numObs, numAct, obsInfo, actInfo, "critic2");

%% =========================
% SAC Agent Options
% =========================
agentOpts = rlSACAgentOptions;
agentOpts.SampleTime = env.Ts;
agentOpts.DiscountFactor = 0.99;
agentOpts.MiniBatchSize = 128;
agentOpts.ExperienceBufferLength = 1e5;
agentOpts.TargetSmoothFactor = 5e-3;

agent = rlSACAgent(actor, [critic1 critic2], agentOpts);

%% =========================
% Training Options
% =========================
trainOpts = rlTrainingOptions;
trainOpts.MaxEpisodes = 400;
trainOpts.MaxStepsPerEpisode = env.MaxSteps;
trainOpts.ScoreAveragingWindowLength = 30;
trainOpts.Plots = "training-progress";
trainOpts.Verbose = true;

% Force training to continue until MaxEpisodes
trainOpts.StopTrainingCriteria = "EpisodeCount";
trainOpts.StopTrainingValue = trainOpts.MaxEpisodes;

%% Train
trainingStats = train(agent, env, trainOpts);

%% Save
save("sac_parallelpark_path_agent.mat", "agent", "trainingStats");
disp("Saved: sac_parallelpark_path_agent.mat");

%% ========= Local function: Critic builder =========
function critic = makeCritic(numObs, numAct, obsInfo, actInfo, namePrefix)
    lgC = layerGraph();

    obsIn = featureInputLayer(numObs, "Name", namePrefix + "_obs");
    actIn = featureInputLayer(numAct, "Name", namePrefix + "_act");
    concat = concatenationLayer(1, 2, "Name", namePrefix + "_concat");

    trunk = [
        fullyConnectedLayer(128, "Name", namePrefix + "_fc1")
        reluLayer("Name", namePrefix + "_relu1")
        fullyConnectedLayer(128, "Name", namePrefix + "_fc2")
        reluLayer("Name", namePrefix + "_relu2")
        fullyConnectedLayer(1, "Name", namePrefix + "_q")
    ];

    lgC = addLayers(lgC, obsIn);
    lgC = addLayers(lgC, actIn);
    lgC = addLayers(lgC, concat);
    lgC = addLayers(lgC, trunk);

    lgC = connectLayers(lgC, namePrefix + "_obs", namePrefix + "_concat/in1");
    lgC = connectLayers(lgC, namePrefix + "_act", namePrefix + "_concat/in2");
    lgC = connectLayers(lgC, namePrefix + "_concat", namePrefix + "_fc1");

    critic = rlQValueFunction(lgC, obsInfo, actInfo, ...
        "ObservationInputNames", namePrefix + "_obs", ...
        "ActionInputNames", namePrefix + "_act");
end