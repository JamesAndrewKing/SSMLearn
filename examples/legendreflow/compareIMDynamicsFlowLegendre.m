clearvars
close all
clc

% Compare monomial and Legendre polynomial flow fits on a smooth analytic
% vector field. The RHS is not a sparse polynomial, so the order sweep tests
% approximation and regularization rather than exact coefficient recovery.

exampleDir = fileparts(mfilename('fullpath'));
repoRoot = fileparts(fileparts(exampleDir));
addpath(genpath(fullfile(repoRoot,'src')))

rng(4)

stateDim = 12;
ROMOrders = 2:4;
nTrainTraj = 48;
nTestTraj = 8;
tTrain = linspace(0,0.45,121);
tTest = linspace(0,0.45,91);
noiseLevel = 0;
l_vals_flow = [0 logspace(-12,-3,16)];
l_vals_legendre = logspace(-8,-2,13);
stateDegreePenalty = 0.2;
nValidation = 5000;

etaData = generateData(nTrainTraj,tTrain,noiseLevel);
[Xtrain,~] = collectRegressionSamples(etaData);
legendreCenter = 0.5*(max(Xtrain,[],2)+min(Xtrain,[],2));
legendreScale = 0.5*(max(Xtrain,[],2)-min(Xtrain,[],2));
legendreScale(abs(legendreScale)<1e-12) = 1;
[validationPts,trueRhsVals] = validationCloud(Xtrain,nValidation);
testIC = 1.4*(2*rand(stateDim,nTestTraj)-1);

rhsErrFlow = zeros(size(ROMOrders));
rhsErrLegendre = zeros(size(ROMOrders));
rolloutErrFlow = zeros(size(ROMOrders));
rolloutErrLegendre = zeros(size(ROMOrders));
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
        'LegendreCenter', legendreCenter, ...
        'LegendreScale', legendreScale, ...
        'fig_disp_nfp', -1);

    rhsErrFlow(iOrder) = relativeError( ...
        RDInfoFlow.reducedDynamics.map(validationPts),trueRhsVals);
    rhsErrLegendre(iOrder) = relativeError( ...
        RDInfoLegendre.reducedDynamics.map(validationPts),trueRhsVals);

    [rolloutErrFlow(iOrder),rolloutErrLegendre(iOrder)] = rolloutErrors( ...
        RDInfoFlow.reducedDynamics.map,RDInfoLegendre.reducedDynamics.map, ...
        testIC,tTest);

    lambdaFlow(iOrder) = RDInfoFlow.reducedDynamics.l_opt;
    lambdaLegendre(iOrder) = RDInfoLegendre.reducedDynamics.l_opt;
    nFeaturesFlow(iOrder) = size(RDInfoFlow.reducedDynamics.coefficients,2);
    nFeaturesLegendre(iOrder) = size(RDInfoLegendre.reducedDynamics.coefficients,2);

    fprintf('order %d: RHS flow %.3e, Legendre %.3e | rollout flow %.3e, Legendre %.3e\n', ...
        ROMOrder,rhsErrFlow(iOrder),rhsErrLegendre(iOrder), ...
        rolloutErrFlow(iOrder),rolloutErrLegendre(iOrder))
end

fprintf('\nAnalytic RHS benchmark\n')
fprintf('state dimension: %d, training trajectories: %d, noise level: %.1e\n', ...
    stateDim,nTrainTraj,noiseLevel)
fprintf('Legendre regression: columnScaledRidge with CV, state degree penalty %.2g\n', ...
    stateDegreePenalty)
fprintf('\nBest RHS error:\n')
fprintf('  IMDynamicsFlow:         %.3e at order %d\n', ...
    min(rhsErrFlow),ROMOrders(argmin(rhsErrFlow)))
fprintf('  IMDynamicsFlowLegendre: %.3e at order %d\n', ...
    min(rhsErrLegendre),ROMOrders(argmin(rhsErrLegendre)))
fprintf('\nBest rollout error:\n')
fprintf('  IMDynamicsFlow:         %.3e at order %d\n', ...
    min(rolloutErrFlow),ROMOrders(argmin(rolloutErrFlow)))
fprintf('  IMDynamicsFlowLegendre: %.3e at order %d\n\n', ...
    min(rolloutErrLegendre),ROMOrders(argmin(rolloutErrLegendre)))

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
semilogy(ROMOrders,rolloutErrFlow,'o-','LineWidth',1.5)
hold on
semilogy(ROMOrders,rolloutErrLegendre,'s-','LineWidth',1.5)
grid on
xlabel('polynomial order')
ylabel('mean relative rollout error')
legend({'IMDynamicsFlow','IMDynamicsFlowLegendre'},'Location','southwest')

function etaData = generateData(nTraj,tSamples,noiseLevel)
etaData = cell(nTraj,2);
for iTraj = 1:nTraj
    x0 = 2.0*(2*rand(stateDimension,1)-1);
    [tSol,xSol] = ode45(@trueRhs,tSamples,x0,odeOptions(tSamples));
    X = xSol.';
    X = X+noiseLevel*std(X,0,2).*randn(size(X));
    etaData{iTraj,1} = tSol.';
    etaData{iTraj,2} = X;
end
end

function dx = trueRhs(~,x)
k = stateDimension;
A = linearCouplingMatrix(k);
B = analyticMixingMatrix(k,0.58);
C = analyticMixingMatrix(k,-0.37);
z = B*x;
dx = A*x ...
    + 1.40*(sin(z)-z) ...
    + 0.90*(tanh(C*x)-C*x) ...
    + 0.45*(exp(-0.30*x.^2)-1).*signSafe(x) ...
    - 0.16*x.*(x.'*x);
end

function A = linearCouplingMatrix(k)
omega = 0.75 + 0.18*(1:k/2) + 0.12*sin(1:k/2);
A = zeros(k);
for iPair = 1:k/2
    idx = 2*iPair-1:2*iPair;
    A(idx,idx) = [-0.05-0.01*iPair, -omega(iPair); ...
        omega(iPair), -0.04-0.008*iPair];
end
for i = 1:k
    for j = 1:k
        if abs(i-j) == 2
            A(i,j) = A(i,j)+0.035*(-1)^(i+j);
        elseif abs(i-j) == 4
            A(i,j) = A(i,j)-0.018;
        end
    end
end
end

function M = analyticMixingMatrix(k,phase)
M = zeros(k);
for i = 1:k
    for j = 1:k
        M(i,j) = 0.42*cos(phase*i*j)/(1+abs(i-j)) ...
            + 0.18*sin(phase*(i+j))/(i+j);
    end
end
M = M / max(abs(eig(M)));
end

function y = signSafe(x)
% Smooth odd factor so the exponential term is analytic and zero at x=0.
y = tanh(1.5*x);
end

function [X,dXdt] = collectRegressionSamples(etaData)
X = [];
dXdt = [];
for iTraj = 1:size(etaData,1)
    [dXi,Xi] = finiteTimeDifference(etaData{iTraj,2},etaData{iTraj,1},3);
    X = [X Xi]; %#ok<AGROW>
    dXdt = [dXdt dXi]; %#ok<AGROW>
end
end

function [gridPts,trueVals] = validationCloud(Xtrain,nPts)
pad = 0.05;
xMin = min(Xtrain,[],2);
xMax = max(Xtrain,[],2);
xSpan = xMax-xMin;
xMin = xMin+pad*xSpan;
xMax = xMax-pad*xSpan;
gridPts = xMin + (xMax-xMin).*rand(numel(xMin),nPts);
trueVals = zeros(size(gridPts));
for iPt = 1:size(gridPts,2)
    trueVals(:,iPt) = trueRhs(0,gridPts(:,iPt));
end
end

function err = relativeError(approx,trueVals)
err = norm(approx-trueVals,'fro')/norm(trueVals,'fro');
end

function [errFlow,errLegendre] = rolloutErrors(RFlow,RLegendre,testIC,tSamples)
nTraj = size(testIC,2);
errFlow = zeros(nTraj,1);
errLegendre = zeros(nTraj,1);
for iTraj = 1:nTraj
    x0 = testIC(:,iTraj);
    opts = odeOptions(tSamples);
    [~,xTrue] = ode45(@trueRhs,tSamples,x0,opts);
    xFlow = integrateModel(RFlow,tSamples,x0,opts);
    xLegendre = integrateModel(RLegendre,tSamples,x0,opts);
    if size(xFlow,1) == numel(tSamples)
        errFlow(iTraj) = relativeError(xFlow.',xTrue.');
    else
        errFlow(iTraj) = NaN;
    end
    if size(xLegendre,1) == numel(tSamples)
        errLegendre(iTraj) = relativeError(xLegendre.',xTrue.');
    else
        errLegendre(iTraj) = NaN;
    end
end
errFlow = mean(errFlow,'omitnan');
errLegendre = mean(errLegendre,'omitnan');
end

function x = integrateModel(R,tSamples,x0,opts)
try
    [~,x] = ode45(@(~,x) R(x),tSamples,x0,opts);
catch
    x = NaN(0,numel(x0));
end
end

function k = stateDimension
k = 12;
end

function idx = argmin(x)
[~,idx] = min(x);
end

function opts = odeOptions(tSamples)
opts = odeset('RelTol',1e-8,'AbsTol',1e-10, ...
    'MaxStep',median(diff(tSamples)));
end
