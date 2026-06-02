% SSM-based model reduction through oblique projection of an 
% Hydraulically Amplified Self-Healing Electrostatic (HASEL) actuator

clear 
close all

%% data extraction

fileNames_tot = {
    'HASEL_experiments/experiment_1_24-Nov-2024_16-44-20',
    'HASEL_experiments/experiment_2_24-Nov-2024_17-22-10',
    'HASEL_experiments/experiment_3_24-Nov-2024_18-00-00',
    'HASEL_experiments/experiment_4_24-Nov-2024_18-37-50',
    'HASEL_experiments/experiment_5_24-Nov-2024_19-15-40',
    'HASEL_experiments/experiment_6_24-Nov-2024_19-53-30',
    'HASEL_experiments/experiment_7_24-Nov-2024_20-31-19',
    'HASEL_experiments/experiment_8_24-Nov-2024_21-09-09'
};

fileNames_tot = flip(fileNames_tot);
V_bar_vect_tot = [3000:500:6500];

index_retained = 8;
fileNames = fileNames_tot(index_retained);
V_bar_vect = V_bar_vect_tot(index_retained);

yDataV = cell(length(V_bar_vect),9);

fp_VLower = cell(length(V_bar_vect),1);
M_VLower = cell(length(V_bar_vect),1);
R_VLower= cell(length(V_bar_vect),1);
Vtg_VLower = cell(length(V_bar_vect),1);

fp_VUpper = cell(length(V_bar_vect),1);
M_VUpper = cell(length(V_bar_vect),1);
R_VUpper= cell(length(V_bar_vect),1);
Vtg_VUpper = cell(length(V_bar_vect),1);

fp_V = cell(length(V_bar_vect),1);
M_V = cell(length(V_bar_vect),1);
R_V = cell(length(V_bar_vect),1);
Vtg_V = cell(length(V_bar_vect),1);

smooth_window = 4000;
smooth_window = 4050;

for vv = 1:size(fileNames,1)
    V_bar_curr = V_bar_vect(vv);
    file_curr = fileNames{vv};
    T = data_extraction(file_curr);
    
    % postprocess
    T = postProcess(T);

    % split trajectories and delay embed
    embedDim = 3;
    delaySteps = 4000;
%     sliceInt = [2 300];
    sliceInt = [0.1 300];
    % yDataTrunc is already delayed embedded
    [yDataTrunc, index_down, index_up] = analysis_step(T,V_bar_curr,embedDim,delaySteps,sliceInt,smooth_window);
    yDataTrunc_down = yDataTrunc(index_down,:);
    yDataTrunc_up = yDataTrunc(index_up,:);

end

dt = mean(diff(yDataTrunc{1,1}));

close all

xData_chosen = yDataTrunc_up;
xData_chosen_data = cell(3,2);
fp_chosen = xData_chosen(:,4);
fp_values = cell2mat(fp_chosen');

% define the fixed point as the mean of the fixed points 
fp = mean(fp_values,2);

% shift the data around the fp
for ii = 1:size(xData_chosen,1)
    xData_chosen_data{ii,1} = xData_chosen{ii,1} - xData_chosen{ii,1}(1);
    xData_chosen_data{ii,2} = xData_chosen{ii,2} - xData_chosen{ii,2}(:,end);

end

SSMDim = 1;
mfddim = SSMDim;
SSMOrder = 5;
ROMOrder = 5;
mfdorder = SSMOrder;
romorder = ROMOrder;


indTrain = [2];
indTest = [3 1];


% compute SSM with orthogonal projection

yData = xData_chosen_data;
yDataTrain = cell(length(indTrain), 2);
yDataTrain(:, 1) = xData_chosen_data(indTrain, 1);
yDataTrain(:, 2) = xData_chosen_data(indTrain, 2);

yDataTest = cell(length(indTest), 2);
yDataTest(:, 1) = xData_chosen_data(indTest, 1);
yDataTest(:, 2) = xData_chosen_data(indTest, 2);

% parametrization
[IMInfo, ~, ~] = IMGeometry(yData(indTrain,:), SSMDim, SSMOrder);
etaData = projectTrajectories(IMInfo, yData);

V_orth = IMInfo.parametrization.tangentSpaceAtOrigin;
P_orth = V_orth * transpose(V_orth);
orth_map = @(y) P_orth * y;
data_proj = transformTrajectories(orth_map, yDataTrain);

data_proj_test = transformTrajectories(orth_map, yDataTest);

% manifold reduced dynamics 
RDInfo = IMDynamicsFlow(etaData(indTrain,:),'R_PolyOrd', ROMOrder);

% advect
[yRec, etaRec, zRec, zData] = advect(IMInfo, RDInfo, yDataTest);

figure; hold on; grid on;
plot(yDataTest{2,1}, yDataTest{2,2}(1,:),'k','LineWidth',2,'DisplayName','Data');
plot(yRec{2,1}, yRec{2,2}(1,:),'r--','LineWidth',2,'DisplayName','Reconstructed data');
legend('Location','N','Interpreter','latex')
set(gca,'FontSize',20,'TickLabelInterpreter','Latex')
ylabel('Stroke [mm]','Interpreter','latex')
xlabel('time [s]','Interpreter','latex')

%% oblique projection from data

data_non_projected = yDataTrain;
 
t = horzcat(data_non_projected{:,1});
X = horzcat(data_non_projected{:,2});
t_red = t(round(0.00001*size(X,2)):end);
X_red = X(:,round(0.00001*size(X,2)):end);

[Xdot, Xnew, tnew] = ftd_data(data_non_projected);

X1 = X_red(:,1:end-1);
X2 = X_red(:,2:end);

[U,S,V] = svds(X1,4);
U = U(:,1:3*SSMDim);
S = S(1:3*SSMDim,1:3*SSMDim);
V = V(:,1:3*SSMDim);
S_tilde = U.'*X2*V*pinv(S);
[E,~,Lambda] = eigSorted(S_tilde);
lambda_A_from_S_tilde = 1/dt*log(Lambda);  
[~,index_sort] = sort(abs(real(lambda_A_from_S_tilde)));
lambda_trunc = lambda_A_from_S_tilde(index_sort);
E_sorted = E(:,index_sort);
V_trunc = U * E_sorted;
V_trunc_slow = V_trunc(:,1);
V_trunc_fast = V_trunc(:,2);
V_trunc_faster = V_trunc(:,3);

V_fast_SVD = [V_trunc_fast V_trunc_faster];
B_slow_SVD = null(V_fast_SVD.');

B0 = B_slow_SVD;
[nB,mB] = size(B0);

V_tg = V_trunc_slow;


R_tilde0 = lambda_trunc(1);


[nR,mR] = size(R_tilde0);
phi_0 = phi_mv(inv(transpose(B0)*V_tg) * transpose(B0) * Xnew, 2:ROMOrder);
N0 = zeros(size(R_tilde0,1), size(phi_0,1));
[nN,mN] = size(N0);

x0 = [B0(:); R_tilde0(:); N0(:)];
J = @(X) OblProjMat_nonlin(X,Xdot,Xnew,nB,mB,nR,mR,nN,mN,ROMOrder,V_tg);

% constraint 
nonlin_constraint = @(X) idemP(X,V_tg,nB,mB);

options = optimoptions('fminunc', 'Display', 'iter','MaxFunctionEvaluation',1e5);
[x_opt, fval] = fminunc(J,x0,options);



xB = x_opt(1:nB*mB);
B = reshape(xB,nB,mB);
xR_tilde = x_opt(1+nB*mB:nB*mB+nR*mR);
R_tilde = reshape(xR_tilde,nR,mR);
xN = x_opt(1+nB*mB+nR*mR:end);
N = reshape(xN,nN,mN);

P = V_tg * inv(transpose(B) * V_tg) * transpose(B);
Eta = P * X;

[aa,bb] = idemP(x_opt,V_tg,nB,mB);


R = R_tilde;
Rmap = @(t,xi) R*xi + N*phi_mv(xi, 2:ROMOrder);

yData_proj = cell(size(yData));
for ii = 1:size(yData,1)
    yData_proj{ii,1} = yData{ii,1};
    yData_proj{ii,2} = P * yData{ii,2};
end

% compute the reduced coordinates as the oblique projection of the
% coordinates in the full observable space onto the slow subspace 

[IMInfo_obl, ~, ~] = IMGeometry(yData_proj(indTrain,:), SSMDim, 1,'Ve',V_tg);
SSMChart_P = @(x) IMInfo_obl.parametrization.tangentSpaceAtOrigin'*P*x;
etaData = projectTrajectories(IMInfo_obl, yData_proj);


[IMInfo_obl, SSMChart, SSMFunction] = IMGeometry(yData(indTrain,:), SSMDim, SSMOrder, 'reducedCoordinates', etaData(indTrain,:));
IMInfo_obl.chart.map = SSMChart_P;

% manifold reduced dynamics 
RDInfo_obl = IMDynamicsFlow(etaData(indTrain,:),'R_PolyOrd', ROMOrder, 'style', 'normalform');

% advect
[yRec_obl, ~, ~, ~] = advect(IMInfo_obl, RDInfo_obl, yDataTest);


%% compare the prediction 

figure; hold on; grid on;
plot(yDataTest{1,1}, yDataTest{1,2}(1,:),'k','LineWidth',3,'DisplayName','Test data');
plot(yRec{1,1}, yRec{1,2}(1,:),':','color',[1.00,0.41,0.16],'LineWidth',3,'DisplayName','SSM model (orthogonal projection)');
plot(yRec_obl{1,1}, yRec_obl{1,2}(1,:),':','color',[0.07,0.62,1.00],'LineWidth',3,'DisplayName','SSM model (oblique projection)');
xlim([0 140])

legend('Location','N','Interpreter','latex')
set(gca,'FontSize',20,'TickLabelInterpreter','Latex')
ylabel('Stroke [mm]','Interpreter','latex')
xlabel('time [s]','Interpreter','latex')


% compute the errors
yRec_orth = cell(1,2);
yRec_orth{1,1} = yRec{1,1};
yRec_orth{1,2} = yRec{1,2};

yRec_Obl = cell(1,2);
yRec_Obl{1,1} = yRec_obl{1,1};
yRec_Obl{1,2} = yRec_obl{1,2};


data = cell(1,2);
data{1,1} = yDataTest{1,1};
data{1,2} = yDataTest{1,2};

normedTrajDist_orth = computeTrajectoryErrors(yRec_orth, data);
RRMSE_orth = mean(normedTrajDist_orth)

normedTrajDist_obl = computeTrajectoryErrors(yRec_Obl, data);
RRMSE_obl = mean(normedTrajDist_obl)


time = yRec{1,1};

% Raw instantaneous error vector
err_orth = yRec{1,2}(1,:) - yDataTest{1,2}(1,:);
err_obl  = yRec_obl{1,2}(1,:) - yDataTest{1,2}(1,:);

% Cumulative running RMS
cumRMS_orth = sqrt(cumsum(err_orth.^2) ./ (1:length(err_orth)));
cumRMS_obl  = sqrt(cumsum(err_obl.^2) ./ (1:length(err_obl)));

% Normalize by max magnitude of test trajectory
norm_factor = max(abs(yDataTest{1,2}(1,:))); 
cumRMS_orth = cumRMS_orth / norm_factor;
cumRMS_obl  = cumRMS_obl  / norm_factor;


figure; hold on; grid on;
legend('Location','N','Interpreter','latex')
set(gca,'FontSize',20,'TickLabelInterpreter','Latex')
ylabel('Stroke Error [$\%$]','Interpreter','latex')
xlabel('time [s]','Interpreter','latex')
xlim([0 140])
set(gca,'YScale','log'); 

plot(time, cumRMS_orth,'-','color',[1.00,0.41,0.16],'LineWidth',3,'DisplayName','SSM model (orthogonal projection)');
plot(time, cumRMS_obl,'-','color',[0.07,0.62,1.00],'LineWidth',3,'DisplayName','SSM model (oblique projection)');

NMTE_orth = RRMSE_orth; 
NMTE_obl  = RRMSE_obl; 

inset_pos = [0.6 0.4 0.25 0.35];
ax_inset = axes('Position', inset_pos); hold on;

bar_vals = [NMTE_orth, NMTE_obl];
hBar = bar(bar_vals,'FaceColor','flat');
hBar.CData(1,:) = [1.00,0.41,0.16]; 
hBar.CData(2,:) = [0.07,0.62,1.00]; 

for i = 1:2
    text(i, bar_vals(i)+0.001, ['$' sprintf('%.1f', bar_vals(i)*100) '$'], ...
        'HorizontalAlignment','center','FontSize',16,'Interpreter','latex','FontWeight','bold');
end

set(ax_inset, 'XTickLabel', [], 'YTickLabel', []);
grid(ax_inset, 'off');
set(ax_inset, 'Box', 'on', 'LineWidth', 1.5, 'Color','white');

ylabel('NMTE [$\%$]','FontSize',20,'Interpreter','latex');
ylim([0 max(bar_vals)*1.2]);


%% Additional functions

function exps=exponents(d,k)
    B = repmat({0:max(k)},1,d);
    A = combvec(B{:}).';
    exps = A(ismember(sum(A,2), k),:);
    [~,ind] = sort(sum(exps,2));
    exps = exps(ind,:);
end

function u = phi_mv(xi, r) % return monomials
    x = reshape(xi, 1, size(xi, 1), []);
    exps = exponents(size(xi, 1),r);
    u = reshape(prod(x.^exps, 2), size(exps, 1), []);
end

function J = OblProjMat_nonlin(x,Xdot,Xnew,nB,mB,nR,mR,nN,mN,ROMOrder,V)

xB = x(1:nB*mB);
B = reshape(xB,nB,mB);

B = B / sqrt(trace(B'*B)); 


xR_tilde = x(1+nB*mB:nB*mB+nR*mR);
R_tilde = reshape(xR_tilde,nR,mR);

xN = x(1+nB*mB+nR*mR:end);
N = reshape(xN,nN,mN);

LHS = transpose(B) * Xdot;
RHS = (transpose(B)*V) * R_tilde * inv(transpose(B)*V) * transpose(B) * Xnew + (transpose(B)*V) * N * phi_mv(inv(transpose(B)*V) * transpose(B) * Xnew, 2:ROMOrder);


Err = LHS - RHS;

J = norm(Err);

end

function [c,ceq] = idemP(x,V,nB,mB)
xB = x(1:nB*mB);
B = reshape(xB,nB,mB);

c = [];
ceq = det(V * inv(transpose(B) * V) * transpose(B) - eye(size(V,1)));

end

function [xDataTrunc, index_down, index_up] = analysis_step(T,V_bar,embedDim,delaySteps,sliceInt,smooth_window)

    % split trajectories
    [xData, fData, fDataMeasured, info] = organizeStepResponse(T,embedDim, delaySteps,smooth_window);
    
    for ii = 1:length(xData)
        xData{ii,3} = info.Voltages(ii);
    end
    
    % select relevant trajectories
    index_train = 3:2:size(xData,1);
    xDataTrain = cell(length(index_train),2);
    
    for ii = 1:length(index_train)
        xDataTrain{ii,1} = xData{index_train(ii),1};
        xDataTrain{ii,2} = xData{index_train(ii),2};
    end
    
    % truncate the trajectories 
    xDataTrunc = sliceTrajectories(xDataTrain, sliceInt);
    
    fDataTrunc = sliceTrajectories(fData, sliceInt);
    fDataMeasuredTrunc = sliceTrajectories(fDataMeasured, sliceInt);
    
    
    for ii = 1:length(index_train)
        xDataTrunc{ii,3} = xData{index_train(ii)-1,3};
    end
    
    figure; hold on; grid on 
    xlabel('Time [s]','Interpreter','latex')
    ylabel('Stroke [mm]','Interpreter','latex')
    set(gca,'FontSize',20,'TickLabelInterpreter','Latex')
    
    legendEntries = cell(1, size(xDataTrunc, 1));
    
    for ii = 1:size(xDataTrunc,1)  
        plot(xDataTrunc{ii,1}, xDataTrunc{ii,2}(1,:),'LineWidth',2)
    %     legendEntries{k} = ['V = ', num2str(xDataTrunc{ii, 3}/1000), ' kV'];
    end
    
    % visualize fixed points
    
    fp_vect = zeros(size(xDataTrunc,1),1);
    for pp = 1:size(xDataTrunc,1)
        fp = xDataTrunc{pp,2}(:,end);
        xDataTrunc{pp,4} = fp;
        fp_vect(pp) = fp(1);
    end
    index_down = 1:3;
    index_up = 4:length(fp_vect);
    fp_down = mean(fp_vect(1:3));
    fp_up = mean(fp_vect(4:end));
    fp_mid = (fp_down + fp_up)/2;
    delta_fp = fp_up - fp_mid;
    
    figure; hold on; grid on 
    xlabel('$\Delta V$ [kV]','Interpreter','latex')
    ylabel('$\bar{x}$ [mm]','Interpreter','latex')
    set(gca,'FontSize',20,'TickLabelInterpreter','Latex')
    
    for pp = 1:size(xDataTrunc,1)
        plot((xDataTrunc{pp,3} - V_bar)/1000,xDataTrunc{pp,4}(1),'*','MarkerSize',10,'LineWidth',2)
    end
    
    % Extract the third column as a vector
    Voltages_initial = cell2mat(xDataTrunc(:, 3));

    % Get the sorting indices
    [~, sortedIndices] = sort(Voltages_initial);

    % Reorder the rows of the cell array using the sorting indices
    xDataTrunc = xDataTrunc(sortedIndices, :);

    xlim([-2 2]);
    ylim([0 20])
    
    x_strip = [-0.5, 0.5, 0.5, -0.5]; % x-coordinates of the strip
    y_strip = [0, 0, 20, 20];         % y-coordinates spanning the y-limits
    fill(x_strip, y_strip, [0.8, 0.8, 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
    
    % plot the interpolating function 
    sigma = 0.2;
    f_interp = @(deltav) fp_mid + atan(deltav/sigma) * (2/pi) * delta_fp;
    deltav_vect = linspace(-2,2,1e2);
    
    plot(deltav_vect, f_interp(deltav_vect),'k--','LineWidth',2)

end

function T = data_extraction(fileNames)

allData = [];
datasetIndices = [];

loaded_data = load(fileNames);
data = loaded_data.data;
clear loaded_data

% Store the starting index of each dataset
datasetIndices = [datasetIndices; size(allData, 1) + 1];

% Calculate the laser sensor offset for this dataset
HV_factor = 1000; % for others
data(:,2) = data(:,2) * HV_factor; % CommandedVoltage
data(:,3) = data(:,3) * HV_factor; % MeasuredInputVoltage
data(:,5) = data(:,5) * 4 + 30; % Change the voltage reading of the laser sensor to absolute displacement (mm)
laser_sensor_offset = mean(data(1:1000, 5));

data(:,5) = data(:,5) - laser_sensor_offset;

% Ensure all position data are above zero
invalidIndices = find(data(:,5) <= 0);
invalidIndices(invalidIndices == 1) = [];
data(invalidIndices, 5) = 0;

allData = [allData; data]; % Concatenating along rows


T = array2table(allData, 'VariableNames', {'Time', 'CommandedVoltage', 'MeasuredInputVoltage', 'MeasuredCurrent', 'Position'});

end

function T_PP = postProcess(T)

% Filtering
% Filter the Position Data
windowLength = 100;
movingAverageFilter = ones(windowLength, 1) / windowLength;
T.PositionFiltered = conv(T.Position, movingAverageFilter, 'same');

% Plotting the filtered and non-filtered data
figure;
yyaxis left;
hold on;
plot(T.Time, T.PositionFiltered, 'LineWidth', 1.0); % Filtered data in red
ylabel('Position (mm)');
xlabel('Time');
title('Comparison of Position Data with Commanded Voltage');

yyaxis right;
plot(T.Time, T.CommandedVoltage/1000, 'LineWidth', 1.0); % Commanded voltage in green
ylabel('Commanded Voltage (kV)');

grid on;
legend('Filtered Position', 'Commanded Voltage');



T_PP = T;

end

function [ydot, ynew, tnew] = ftd_data(yData)
    ydot = []; ynew = []; tnew = [];
    for iTraj = 1:size(yData, 1)
        [yd, y, t] = finiteTimeDifference(yData{iTraj,2}, yData{iTraj,1}, 4);
        ydot = [ydot, yd]; ynew = [ynew, y]; tnew = [tnew, t];
    end
end

