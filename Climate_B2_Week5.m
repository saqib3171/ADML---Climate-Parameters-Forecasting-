clc;close all;clear all;

%% Load Data
trainData = readtable('DailyDelhiClimateTrain.csv');
testData = readtable('DailyDelhiClimateTest.csv');
trainData.date = datetime(trainData.date, 'InputFormat', 'yyyy-MM-dd');
testData.date = datetime(testData.date, 'InputFormat', 'yyyy-MM-dd');
varNames = {'meantemp', 'humidity', 'wind_speed', 'meanpressure'};
combined_Data = [trainData; testData]; 

%% Preprocess Data (Normalization, Outlier Handling, etc.)

% Normalize data
for i = 1:length(varNames)
    varName = varNames{i};
    % Finding minimum and maximum values from the training data
    minVal = min(trainData.(varName), [], 'omitnan');
    maxVal = max(trainData.(varName), [], 'omitnan');
    % Normalize both training and testing data
    trainData.(varName) = (trainData.(varName) - minVal) / (maxVal - minVal);
    testData.(varName) = (testData.(varName) - minVal) / (maxVal - minVal);
end

% Outlier Handling (replace with NaN and then impute)
% Handle outliers in training data
for i = 1:length(varNames)
    varName = varNames{i};
    WO_NaN = trainData{:, varName}(~isnan(trainData{:, varName})); % Remove NaN values
    Q1 = prctile(WO_NaN, 25);
    Q3 = prctile(WO_NaN, 75);
    IQR = Q3 - Q1;
    outlierInd = trainData{:, varName} < Q1 - 1.5 * IQR | trainData{:, varName} > Q3 + 1.5 * IQR;
    trainData{:, varName}(outlierInd) = NaN;
    % Impute NaNs with the median
    trainData{:, varName} = fillmissing(trainData{:, varName}, 'movmedian', 24);
end

% Handle outliers in test data
for i = 1:length(varNames)
    varName = varNames{i};
    WO_NaN = testData{:, varName}(~isnan(testData{:, varName})); % Remove NaN values
    Q1 = prctile(WO_NaN, 25);
    Q3 = prctile(WO_NaN, 75);
    IQR = Q3 - Q1;
    outlierInd = testData{:, varName} < Q1 - 1.5 * IQR | testData{:, varName} > Q3 + 1.5 * IQR;
    testData{:, varName}(outlierInd) = NaN;
    % Impute NaNs with the median
    testData{:, varName} = fillmissing(testData{:, varName}, 'movmedian', 24);
end

%% Calculate and Plot Correlation Matrix for Training Data
corrMatrixTrain = corr(trainData{:, varNames}, 'Rows', 'complete');
figure;
heatmap(trainData.Properties.VariableNames(varNames), ...
        trainData.Properties.VariableNames(varNames), ...
        corrMatrixTrain);
title('Correlation Matrix of Training Data Variables');
colormap('jet'); 
colorbar;

%% Decompose Training Data using trenddecomp
seasonalPeriod = 365;  % Annual seasonality

if length(trainData.meantemp) > 2 * seasonalPeriod
    [trainData.trend_meantemp, trainData.seasonal_meantemp, trainData.residual_meantemp] = trenddecomp(trainData.meantemp, 'stl', seasonalPeriod);
    trainData.meantemp_deseasonalized = trainData.meantemp - trainData.seasonal_meantemp;
else
    error('Training data length is too short for the specified seasonal period.');
end

% Use the seasonal component from the training data for the test data
testData.seasonal_meantemp = trainData.seasonal_meantemp(end-length(testData.meantemp)+1:end);
testData.meantemp_deseasonalized = testData.meantemp - testData.seasonal_meantemp;

%% Time Series Cross-Validation 
Folds = 5; % Number of folds
foldSize = floor(height(trainData) / Folds);
rmseFolds = zeros(Folds, 1); % To store RMSE for each fold

for fold = 1:Folds
    fprintf('Processing fold %d/%d...\n', fold, Folds);
    
    % Define train and validation sets
    val_Idx = (foldSize * (fold - 1) + 1):(foldSize * fold);
    train_Idx = setdiff(1:height(trainData), val_Idx);
    
    foldTrainData = trainData(train_Idx, :);
    foldValData = trainData(val_Idx, :);

    % Feature Selection - 'meantemp' as input and 'humidity' as output
    XTrain = foldTrainData.meantemp_deseasonalized(1:end-1)';
    YTrain = foldTrainData.humidity(2:end)';
    XVal = foldValData.meantemp_deseasonalized(1:end-1)';
    YVal = foldValData.humidity(2:end)';
    
    % Reshape data for LSTM
    XTrain = reshape(XTrain, [1, numel(XTrain), 1]);
    YTrain = reshape(YTrain, [1, numel(YTrain), 1]);
    XVal = reshape(XVal, [1, numel(XVal), 1]);
    YVal = reshape(YVal, [1, numel(YVal), 1]);
    
%% LSTM Network Architecture
    numFeatures = 1; 
    numResponses = 1; 
    layer1 = 100; % first LSTM layer
    layer2 = 100; %  second LSTM layer
    
    layers = [ ...
        sequenceInputLayer(numFeatures)
        lstmLayer(layer1, 'OutputMode', 'sequence')
        dropoutLayer(0.3)
        lstmLayer(layer2)
        fullyConnectedLayer(numResponses)
        regressionLayer];
    
    % Hyperparameters Tuning
    options = trainingOptions('adam', ...
        'MaxEpochs', 100, ...
        'MiniBatchSize', 64, ...
        'InitialLearnRate', 0.01, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropPeriod', 50, ...
        'LearnRateDropFactor', 0.3, ...
        'L2Regularization', 0.0001, ...
        'GradientThreshold', 1, ...
        'Verbose', 0, ...
        'Plots', 'training-progress');
    
    % Train LSTM Network on Fold's Training Data
    net = trainNetwork(XTrain, YTrain, layers, options);
    
    % Predict and Evaluate on Fold's Validation Data
    YPred = predict(net, XVal, 'MiniBatchSize', 1);
    YPred = reshape(YPred, [numel(YPred), 1]);
    YVal = reshape(YVal, [numel(YVal), 1]);
    
    % Calculate RMSE
    rmse = sqrt(mean((YPred - YVal).^2));
    
    % Store the fold's RMSE
    rmseFolds(fold) = rmse;
end

%% Aggregate Performance Across Folds
mean_Rmse = mean(rmseFolds);
fprintf('Average RMSE across folds: %.2f\n', mean_Rmse);

%% Prepare Test Data for LSTM Prediction
% Extract features and targets from the test data
meantemp_Test = testData.meantemp_deseasonalized(1:end-1)'; 
humidity_Test = testData.humidity(2:end)';

% Reshape data for LSTM 
XTest = reshape(meantemp_Test, [1, numel(meantemp_Test), 1]);
YTest = reshape(humidity_Test, [1, numel(humidity_Test), 1]);

%% Predict Using the LSTM Network on Test Data
YPred_Test = predict(net, XTest, 'MiniBatchSize', 1);
YPred_Test = reshape(YPred_Test, [numel(YPred_Test), 1]);

YPredTest_with_seasonality = YPred_Test + testData.seasonal_meantemp(1:length(YPred_Test));

%% Evaluate Model Performance with Seasonality Component
rmseTestSeasonality = sqrt(mean((YPredTest_with_seasonality - testData.humidity(1:length(YPred_Test))).^2));
fprintf('Test RMSE with Seasonality: %.2f\n', rmseTestSeasonality);
  
    %% ---------------  Visualization -------------------------------

    %% Box Plots for Distribution of Variables
colors = {'blue', 'green', 'red', 'cyan'};
figure;

for i = 1:length(varNames)
    subplot(2,2,i);
    boxplot(combined_Data.(varNames{i}));
    title(['Box Plot of ', varNames{i}]);
    h = findobj(gca,'Tag','Box');
    patch(get(h,'XData'),get(h,'YData'),colors{i},'FaceAlpha',.5);
end

%% Histogram for the variables
figure;
numBins = 50;

for i = 1:length(varNames)
    subplot(2, 2, i);
    histogram(combined_Data.(varNames{i}), numBins);
    title(['Histogram of ', varNames{i}]);
    xlabel(varNames{i});
    ylabel('Frequency');
end
sgtitle('Histograms of Climate Variables');

%% Create autocorrelation plots
figure;

for i = 1:length(varNames)
    varName = varNames{i};

% Create a subplot for each autocorrelation plot
    subplot(2, 2, i);
    autocorr(combined_Data.(varName));
    title(['Autocorrelation of ', varName]);
end

%% Plotting Seasonal Component and Deseasonalized Data
%% Plotting All Components
figure;

% Plot Original Meantemp Data
subplot(4, 1, 1);
plot(trainData.date, trainData.meantemp, 'b');
title('Original Meantemp Data');
xlabel('Date');
ylabel('Meantemp');
grid on;

% Plot Trend Component
subplot(4, 1, 2);
plot(trainData.date, trainData.trend_meantemp, 'g');
title('Trend Component of Meantemp');
xlabel('Date');
ylabel('Trend');
grid on;

% Plot Seasonal Component
subplot(4, 1, 3);
plot(trainData.date, trainData.seasonal_meantemp, 'r');
title('Seasonal Component of Meantemp');
xlabel('Date');
ylabel('Seasonality');
grid on;

% Plot Residual Component
subplot(4, 1, 4);
plot(trainData.date, trainData.residual_meantemp, 'k');
title('Residual Component of Meantemp');
xlabel('Date');
ylabel('Residual');
grid on;
set(gcf, 'Position', [100, 100, 700, 500]); 

%% Visualization of Predictions vs. Actual Values with Seasonality
figure;
plot(testData.date(1:length(YPred_Test)), testData.humidity(1:length(YPred_Test)), 'b');
hold on;
plot(testData.date(1:length(YPred_Test)), YPredTest_with_seasonality, 'r');
hold off;
xlabel('Date');
ylabel('Humidity');
title('Test Data: Actual vs Predicted Humidity with Seasonality');
legend('Actual Humidity', 'Predicted Humidity');
grid on;

% Number of folds
Folds = length(rmseFolds);

% bar graph Cross-Validation RMSE 
figure;
bar(1:Folds, rmseFolds);
xlabel('Fold Number');
ylabel('RMSE');
title('Cross-Validation RMSE for Each Fold');
xticks(1:Folds);
grid on;
axis tight; 
