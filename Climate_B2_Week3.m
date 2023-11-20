clc;
close all;
clear all;

%% Load and combine datasets
trainData = readtable('DailyDelhiClimateTrain.csv');
testData = readtable('DailyDelhiClimateTest.csv');
combined_Data = [trainData; testData]; % For analysis purpose
combined_Data.date = datetime(combined_Data.date, 'InputFormat', 'yyyy-MM-dd');
varNames = {'meantemp', 'humidity', 'wind_speed', 'meanpressure'};

%% Missing Values Check
missingSummary = sum(ismissing(combined_Data));
disp('Summary of missing values in each column:');
disp(missingSummary);

%% Normalize the Data
for i = 1:length(varNames)
    varName = varNames{i};
    minVal = min(combined_Data.(varName));
    maxVal = max(combined_Data.(varName));
    % Apply Min-Max Scaling
    combined_Data.(varName) = (combined_Data.(varName) - minVal) / (maxVal - minVal);
end

%% Variables to check for outliers

% Loop through each variable to check for outliers
for i = 1:length(varNames)
    varName = varNames{i};
    Q1 = prctile(combined_Data.(varName), 25);
    Q3 = prctile(combined_Data.(varName), 75);
    IQR = Q3 - Q1;
    outlier_Inds = combined_Data.(varName) < Q1 - 1.5 * IQR | combined_Data.(varName) > Q3 + 1.5 * IQR;
    outliers = combined_Data.(varName)(outlier_Inds);
    disp(['Identified outliers in ', varName, ':']);
    disp(outliers);
end

%% Time Series Visualization
figure;
subplot(2,2,1);
plot(combined_Data.date, combined_Data.meantemp, 'r');
title('Mean Temperature Over Time');
ylabel('Temperature (C)');

subplot(2,2,2);
plot(combined_Data.date, combined_Data.humidity, 'b');
title('Humidity Over Time');
ylabel('Humidity (%)');

subplot(2,2,3);
plot(combined_Data.date, combined_Data.wind_speed, 'g');
title('Wind Speed Over Time');
ylabel('Wind Speed (km/h)');

subplot(2,2,4);
plot(combined_Data.date, combined_Data.meanpressure, 'm');
title('Mean Pressure Over Time');
ylabel('Pressure (hPa)');

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

figure;
 %% Create a PDF plot for each variable
for i = 1:length(varNames)
    varName = varNames{i};
    subplot(2, 2, i);
    [f, xi] = ksdensity(combined_Data.(varName));
    plot(xi, f);

% Set title for each subplot
    title(['PDF of ', varName]);
    xlabel(varName);
    ylabel('Density');
end
sgtitle('Probability Density Functions of Climate Variables');

%% Correlation Matrix
corrMatrix = corr(combined_Data{:, 2:5}, 'Rows', 'complete');
figure;
h = heatmap(combined_Data.Properties.VariableNames(2:5), combined_Data.Properties.VariableNames(2:5), corrMatrix);
title('Correlation Matrix of Variables');
colormap(h, 'jet'); 

%% decomposition

period = 365; % Annual period
add_Periods = [182, 90]; % Additional periods: Semi-annual and quarterly
variablesToDecompose = {'meantemp', 'humidity', 'wind_speed', 'meanpressure'};
for i = 1:length(varNames)
    varName = varNames{i};
    dateNums = datenum(combined_Data.date);
    [dateNums, ia] = unique(dateNums);
    uniqueVarData = combined_Data.(varName)(ia);

    % Loop through each period
    figure;
    for p = [period, add_Periods]
        if length(uniqueVarData) < 2*p
            warning(['Data length for ', varName, ' is too short for period: ', num2str(p)]);
            continue;
        end

        % Decompose 
        [td, sd, rd] = trenddecomp(uniqueVarData, 'stl', p);

        % Plotting the results for each period
        subplot(length([period, add_Periods]), 1, find([period, add_Periods] == p));
        plot(combined_Data.date(ia), sd);
        title(['Seasonal Component of ', varName, ' - Period: ', num2str(p)]);
    end
    sgtitle(['STL Decomposition of ', varName]);
end
    % Detecting Outliers in Residuals
    outlierThreshold = 3 * std(rd); % Example threshold: 3 standard deviations
    outlierInds = abs(rd) > outlierThreshold;
    combined_Data.(varName)(ia(outlierInds)) = NaN; % Replace outliers with NaN

   % Plotting the results after outlier removal
    figure;
    subplot(4,1,1);
    plot(combined_Data.date(ia), td, 'b');
    title(['Trend Component of ', varName]);

    subplot(4,1,2);
    plot(combined_Data.date(ia), sd, 'g');
    title(['Seasonal Component of ', varName]);

    subplot(4,1,3);
    plot(combined_Data.date(ia), rd, 'r');
    title(['Residuals of ', varName]);

    subplot(4,1,4);
    plot(combined_Data.date, combined_Data.(varName), 'k');
    title(['Cleaned ', varName]);

%% Create autocorrelation plots
figure;

for i = 1:length(varNames)
    varName = varNames{i};

% Create a subplot for each autocorrelation plot
    subplot(2, 2, i);
    autocorr(combined_Data.(varName));
    title(['Autocorrelation of ', varName]);
end

%% Histogram for the variables
figure;
numBins = 50; % No. of bins

for i = 1:length(varNames)
    subplot(2, 2, i);
    histogram(combined_Data.(varNames{i}), numBins);
    title(['Histogram of ', varNames{i}]);
    xlabel(varNames{i});
    ylabel('Frequency');
end
sgtitle('Histograms of Climate Variables');

