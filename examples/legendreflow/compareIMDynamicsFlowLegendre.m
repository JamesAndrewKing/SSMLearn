clearvars
close all
clc

% Compare IMDynamicsFlow and IMDynamicsFlowLegendre on a 10D analytic
% nonlinear vector field. The nonlinear terms have their linear Taylor parts
% removed, so the comparison focuses on nonlinear reduced dynamics fits.

exampleDir = fileparts(mfilename('fullpath'));
repoRoot = fileparts(fileparts(exampleDir));
addpath(genpath(fullfile(repoRoot,'src')))

rng(4)

%% Parameters
stateDim = 10;
ROMOrders = 2:4;
nTrainTraj = 48;
nTestTraj = 8;
tTrain = linspace(0,0.45,121);
tTest = linspace(0,0.45,91);
noiseLevel = 0;
nValidation = 5000;

l_vals_flow = [0 logspace(-12,-3,16)];
l_vals_legendre = logspace(-8,-2,13);
% Power p in the Legendre smoothness penalty (1+|alpha|)^(2p).
% Set to 0 to penalize all Legendre features equally.
stateDegreePenalty = 1.5;

%% Analytic RHS
omega = 0.75 + 0.18*(1:stateDim/2) + 0.12*sin(1:stateDim/2);
A = zeros(stateDim);
for iPair = 1:stateDim/2
    idx = 2*iPair-1:2*iPair;
    A(idx,idx) = [-0.05-0.01*iPair, -omega(iPair); ...
        omega(iPair), -0.04-0.008*iPair];
end
for i = 1:stateDim
    for j = 1:stateDim
        if abs(i-j) == 2
            A(i,j) = A(i,j)+0.035*(-1)^(i+j);
        elseif abs(i-j) == 4
            A(i,j) = A(i,j)-0.018;
        end
    end
end

B = zeros(stateDim);
C = zeros(stateDim);
for i = 1:stateDim
    for j = 1:stateDim
        B(i,j) = 0.42*cos(0.58*i*j)/(1+abs(i-j)) ...
            + 0.18*sin(0.58*(i+j))/(i+j);
        C(i,j) = 0.42*cos(-0.37*i*j)/(1+abs(i-j)) ...
            + 0.18*sin(-0.37*(i+j))/(i+j);
    end
end
B = B/max(abs(eig(B)));
C = C/max(abs(eig(C)));

rhs = @(~,x) A*x ...
    + 1.40*(sin(B*x)-B*x) ...
    + 0.90*(tanh(C*x)-C*x) ...
    + 0.45*(exp(-0.30*x.^2)-1).*tanh(1.5*x) ...
    - 0.16*x.*(x.'*x);

odeOptsTrain = odeset('RelTol',1e-8,'AbsTol',1e-10, ...
    'MaxStep',median(diff(tTrain)));
odeOptsTest = odeset('RelTol',1e-8,'AbsTol',1e-10, ...
    'MaxStep',median(diff(tTest)));

%% Generate training trajectories
etaData = cell(nTrainTraj,2);
for iTraj = 1:nTrainTraj
    x0 = 2.0*(2*rand(stateDim,1)-1);
    [tSol,xSol] = ode45(rhs,tTrain,x0,odeOptsTrain);
    X = xSol.';
    X = X + noiseLevel*std(X,0,2).*randn(size(X));
    etaData{iTraj,1} = tSol.';
    etaData{iTraj,2} = X;
end

%% Regression samples for validation range
Xtrain = [];
for iTraj = 1:size(etaData,1)
    [~,Xi] = finiteTimeDifference(etaData{iTraj,2},etaData{iTraj,1},3);
    Xtrain = [Xtrain Xi]; %#ok<AGROW>
end

%% Validation cloud for vector-field error
xMin = min(Xtrain,[],2);
xMax = max(Xtrain,[],2);
xSpan = xMax-xMin;
xMin = xMin+0.05*xSpan;
xMax = xMax-0.05*xSpan;
validationPts = xMin + (xMax-xMin).*rand(stateDim,nValidation);
trueRhsVals = zeros(stateDim,nValidation);
for iPt = 1:nValidation
    trueRhsVals(:,iPt) = rhs(0,validationPts(:,iPt));
end

%% Test trajectories for NMTE
testIC = 1.4*(2*rand(stateDim,nTestTraj)-1);
yDataTest = cell(nTestTraj,2);
for iTraj = 1:nTestTraj
    [~,xTrue] = ode45(rhs,tTest,testIC(:,iTraj),odeOptsTest);
    yDataTest{iTraj,1} = tTest;
    yDataTest{iTraj,2} = xTrue.';
end

%% Fit and compare
rhsErrFlow = zeros(size(ROMOrders));
rhsErrLegendre = zeros(size(ROMOrders));
NMTEFlow = zeros(size(ROMOrders));
NMTELegendre = zeros(size(ROMOrders));
lambdaFlow = zeros(size(ROMOrders));
lambdaLegendre = zeros(size(ROMOrders));
nFeaturesFlow = zeros(size(ROMOrders));
nFeaturesLegendre = zeros(size(ROMOrders));

for iOrder = 1:numel(ROMOrders)
    ROMOrder = ROMOrders(iOrder);

    RDInfoFlow = IMDynamicsFlow(etaData, ...
        'R_PolyOrd', ROMOrder, ...
        'l_vals', l_vals_flow, ...
        'n_folds', 5, ...
        'fig_disp_nfp', -1);

    RDInfoLegendre = IMDynamicsFlowLegendre(etaData, ...
        'R_PolyOrd', ROMOrder, ...
        'l_vals', l_vals_legendre, ...
        'n_folds', 5, ...
        'state_degree_penalty', stateDegreePenalty, ...
        'fig_disp_nfp', -1);

    flowRhs = RDInfoFlow.reducedDynamics.map(validationPts);
    legendreRhs = RDInfoLegendre.reducedDynamics.map(validationPts);
    rhsErrFlow(iOrder) = norm(flowRhs-trueRhsVals,'fro')/norm(trueRhsVals,'fro');
    rhsErrLegendre(iOrder) = norm(legendreRhs-trueRhsVals,'fro')/norm(trueRhsVals,'fro');

    yRecFlow = cell(nTestTraj,2);
    yRecLegendre = cell(nTestTraj,2);
    for iTraj = 1:nTestTraj
        x0 = testIC(:,iTraj);
        xFlow = integrateModel(RDInfoFlow.reducedDynamics.map,tTest,x0,odeOptsTest);
        xLegendre = integrateModel(RDInfoLegendre.reducedDynamics.map,tTest,x0,odeOptsTest);

        yRecFlow{iTraj,1} = tTest;
        yRecFlow{iTraj,2} = xFlow.';
        yRecLegendre{iTraj,1} = tTest;
        yRecLegendre{iTraj,2} = xLegendre.';
    end

    normedTrajDistFlow = computeTrajectoryErrors(yRecFlow,yDataTest);
    normedTrajDistLegendre = computeTrajectoryErrors(yRecLegendre,yDataTest);
    NMTEFlow(iOrder) = mean(normedTrajDistFlow)*100;
    NMTELegendre(iOrder) = mean(normedTrajDistLegendre)*100;

    lambdaFlow(iOrder) = RDInfoFlow.reducedDynamics.l_opt;
    lambdaLegendre(iOrder) = RDInfoLegendre.reducedDynamics.l_opt;
    nFeaturesFlow(iOrder) = size(RDInfoFlow.reducedDynamics.coefficients,2);
    nFeaturesLegendre(iOrder) = size(RDInfoLegendre.reducedDynamics.coefficients,2);

    fprintf('order %d: RHS flow %.3e, Legendre %.3e | NMTE flow %.2f%%, Legendre %.2f%%\n', ...
        ROMOrder,rhsErrFlow(iOrder),rhsErrLegendre(iOrder), ...
        NMTEFlow(iOrder),NMTELegendre(iOrder))
end

%% Summary
[bestRhsFlow,bestRhsFlowIdx] = min(rhsErrFlow);
[bestRhsLegendre,bestRhsLegendreIdx] = min(rhsErrLegendre);
[bestNMTEFlow,bestNMTEFlowIdx] = min(NMTEFlow);
[bestNMTELegendre,bestNMTELegendreIdx] = min(NMTELegendre);

fprintf('\nAnalytic RHS benchmark\n')
fprintf('state dimension: %d, training trajectories: %d, noise level: %.1e\n', ...
    stateDim,nTrainTraj,noiseLevel)
fprintf('Legendre regression: columnScaledRidge with CV, state degree penalty %.2g\n', ...
    stateDegreePenalty)
fprintf('\nBest RHS error:\n')
fprintf('  IMDynamicsFlow:         %.3e at order %d\n', ...
    bestRhsFlow,ROMOrders(bestRhsFlowIdx))
fprintf('  IMDynamicsFlowLegendre: %.3e at order %d\n', ...
    bestRhsLegendre,ROMOrders(bestRhsLegendreIdx))
fprintf('\nBest NMTE:\n')
fprintf('  IMDynamicsFlow:         %.2f%% at order %d\n', ...
    bestNMTEFlow,ROMOrders(bestNMTEFlowIdx))
fprintf('  IMDynamicsFlowLegendre: %.2f%% at order %d\n\n', ...
    bestNMTELegendre,ROMOrders(bestNMTELegendreIdx))

%% Plot errors
figure('Color','w')
tiledlayout(1,2,'TileSpacing','compact','Padding','compact')

nexttile
semilogy(ROMOrders,rhsErrFlow,'o-','LineWidth',1.5)
hold on
semilogy(ROMOrders,rhsErrLegendre,'s-','LineWidth',1.5)
grid on
xlabel('polynomial order')
ylabel('relative RHS error')
legend({'IMDynamicsFlow','IMDynamicsFlowLegendre'},'Location','southwest')

nexttile
plotNMTEFlow = NMTEFlow;
plotNMTELegendre = NMTELegendre;
plotNMTEFlow(isinf(plotNMTEFlow)) = NaN;
plotNMTELegendre(isinf(plotNMTELegendre)) = NaN;
semilogy(ROMOrders,plotNMTEFlow,'o-','LineWidth',1.5)
hold on
semilogy(ROMOrders,plotNMTELegendre,'s-','LineWidth',1.5)
grid on
xlabel('polynomial order')
ylabel('NMTE (%)','Interpreter','none')
legend({'IMDynamicsFlow','IMDynamicsFlowLegendre'},'Location','southwest')

function x = integrateModel(R,tSamples,x0,opts)
try
    [~,x] = ode45(@(~,x) R(x),tSamples,x0,opts);
catch
    x = NaN(0,numel(x0));
end
end
