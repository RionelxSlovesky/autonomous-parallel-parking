% train_behavior_cloning_parallelpark.m
clear; clc; close all;

%% =========================
% Behavior Cloning for Parallel Parking
% =========================
% Trains a neural network to imitate the expert controller from:
%   expert_parallelpark_trajectories.mat
%
% Input state:
%   [x, y, psi, v, delta, d_long, d_lat, e_yaw]
%
% Output action:
%   [steerCmd, vCmd]

%% Load expert dataset
load('expert_parallelpark_trajectories.mat', 'expertDataset');

X = expertDataset.states;    % N x 8
Y = expertDataset.actions;   % N x 2

fprintf('Loaded dataset:\n');
fprintf('  Num samples: %d\n', size(X,1));
fprintf('  State dim  : %d\n', size(X,2));
fprintf('  Action dim : %d\n', size(Y,2));

%% Shuffle
rng(42);
idx = randperm(size(X,1));
X = X(idx,:);
Y = Y(idx,:);

%% Train/validation split
N = size(X,1);
Ntrain = round(0.85 * N);

XTrain = X(1:Ntrain,:);
YTrain = Y(1:Ntrain,:);

XVal = X(Ntrain+1:end,:);
YVal = Y(Ntrain+1:end,:);

%% Normalize inputs and outputs
xMean = mean(XTrain,1);
xStd  = std(XTrain,0,1);
xStd(xStd < 1e-8) = 1.0;

yMean = mean(YTrain,1);
yStd  = std(YTrain,0,1);
yStd(yStd < 1e-8) = 1.0;

XTrainN = (XTrain - xMean) ./ xStd;
XValN   = (XVal   - xMean) ./ xStd;

YTrainN = (YTrain - yMean) ./ yStd;
YValN   = (YVal   - yMean) ./ yStd;

%% Convert to format expected by trainNetwork
% featureInputLayer expects observations in rows for tabular data
% regression output can also stay row-wise in recent MATLAB versions
% To be safe, use datastore-like arrays via direct call.

%% Network architecture
layers = [
    featureInputLayer(size(XTrainN,2), 'Name','state', 'Normalization','none')

    fullyConnectedLayer(128, 'Name','fc1')
    reluLayer('Name','relu1')

    fullyConnectedLayer(128, 'Name','fc2')
    reluLayer('Name','relu2')

    fullyConnectedLayer(64, 'Name','fc3')
    reluLayer('Name','relu3')

    fullyConnectedLayer(size(YTrainN,2), 'Name','action')
    regressionLayer('Name','regressionoutput')
];

%% Training options
options = trainingOptions('adam', ...
    'MaxEpochs', 150, ...
    'MiniBatchSize', 128, ...
    'InitialLearnRate', 1e-3, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {XValN, YValN}, ...
    'ValidationFrequency', 25, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

%% Train network
net = trainNetwork(XTrainN, YTrainN, layers, options);

%% Evaluate
YPredValN = predict(net, XValN);
YPredVal = YPredValN .* yStd + yMean;

mseSteer = mean((YPredVal(:,1) - YVal(:,1)).^2);
mseVcmd  = mean((YPredVal(:,2) - YVal(:,2)).^2);
maeSteer = mean(abs(YPredVal(:,1) - YVal(:,1)));
maeVcmd  = mean(abs(YPredVal(:,2) - YVal(:,2)));

fprintf('\n==== Behavior Cloning Validation Metrics ====\n');
fprintf('Steer MSE: %.6f\n', mseSteer);
fprintf('Vcmd  MSE: %.6f\n', mseVcmd);
fprintf('Steer MAE: %.6f\n', maeSteer);
fprintf('Vcmd  MAE: %.6f\n', maeVcmd);

%% Save trained policy + normalization stats
bcPolicy = struct();
bcPolicy.net = net;
bcPolicy.xMean = xMean;
bcPolicy.xStd = xStd;
bcPolicy.yMean = yMean;
bcPolicy.yStd = yStd;

save('bc_parallelpark_policy.mat', 'bcPolicy');
disp('Saved bc_parallelpark_policy.mat');

%% Quick diagnostic plots
figure('Name','Behavior Cloning Validation Diagnostics');

subplot(2,2,1);
plot(YVal(:,1), YPredVal(:,1), '.');
grid on;
xlabel('True steer');
ylabel('Pred steer');
title('Steering Prediction');

subplot(2,2,2);
plot(YVal(:,2), YPredVal(:,2), '.');
grid on;
xlabel('True vCmd');
ylabel('Pred vCmd');
title('Speed Command Prediction');

subplot(2,2,3);
histogram(YVal(:,1) - YPredVal(:,1), 40);
grid on;
title('Steering Error Histogram');

subplot(2,2,4);
histogram(YVal(:,2) - YPredVal(:,2), 40);
grid on;
title('vCmd Error Histogram');