clearvars
close all
clc

% Compare IMDynamicsFlowParaCon and IMDynamicsFlowParaConLegendre on a
% parameter-dependent analytic vector field. The parameter is constant along
% each trajectory, matching the ParaCon data layout. The origin is fixed
% only at p = 0; for p ~= 0 the vector field has a parameter-dependent
% offset.

exampleDir = fileparts(mfilename('fullpath'));
repoRoot = fileparts(fileparts(exampleDir));
addpath(genpath(fullfile(repoRoot,'src')))

rng(8)

%% Parameters
stateDim = 4;
paramDim = 2;
stateOrder = 5;
paramOrder = 3;
[p1Grid,p2Grid] = ndgrid(linspace(-0.85,0.85,5),linspace(-0.7,0.7,5));
trainParams = [p1Grid(:).'; p2Grid(:).'];
nICPerParam = 2;
testParams = [-0.55 0.0 0.55; ...
               0.45 -0.5 0.05];
tTrain = linspace(0,0.45,81);
tTest = linspace(0,0.45,81);
nValidation = 4000;

% Powers in the split Legendre smoothness penalty
% (1+|alpha_x|)^(2p_x) * (1+|alpha_p|)^(2p_p).
% Set either power to 0 to penalize that direction uniformly.
lVals = 1e-8;
nFolds = 0;
stateDegreePenalty = 3;
parameterDegreePenalty = 3;

%% Analytic parameter-dependent RHS
A0 = zeros(stateDim);
for iPair = 1:stateDim/2
    idx = 2*iPair-1:2*iPair;
    omega = 0.75+0.17*iPair;
    A0(idx,idx) = [-0.07-0.01*iPair, -omega; ...
        omega, -0.06-0.01*iPair];
end
A1 = 0.08*toeplitz(cos(1:stateDim));
A2 = 0.07*toeplitz(sin(1:stateDim));
B = 0.45*cos((1:stateDim).'.*(1:stateDim)/stateDim);
G = 0.35*cos((1:stateDim).'.*(1:paramDim)/(stateDim+paramDim));

rhs = @(~,x,p) (A0+p(1)*A1+p(2)*A2)*x ...
    + G*[sin(2.2*p(1))+0.5*p(1)*p(2); ...
         tanh(1.7*p(2))+0.3*p(1)^2] ...
    + 0.9*sin(B*x) ...
    + 0.35*p(1)*cos(1.2*x) ...
    - 0.2*x.*(x.'*x);

odeOptsTrain = odeset('RelTol',1e-8,'AbsTol',1e-10,...
    'MaxStep',median(diff(tTrain)));
odeOptsTest = odeset('RelTol',1e-8,'AbsTol',1e-10,...
    'MaxStep',median(diff(tTest)));

%% Training data
nTrainTraj = size(trainParams,2)*nICPerParam;
xpData = cell(nTrainTraj,3);
iTraj = 0;
for iParam = 1:size(trainParams,2)
    p = trainParams(:,iParam);
    for iIC = 1:nICPerParam
        iTraj = iTraj+1;
        x0 = 1.2*(2*rand(stateDim,1)-1);
        [tSol,xSol] = ode45(@(t,x) rhs(t,x,p),tTrain,x0,odeOptsTrain);
        xpData{iTraj,1} = tSol.';
        xpData{iTraj,2} = xSol.';
        xpData{iTraj,3} = p;
    end
end

%% Validation cloud for vector-field error
Xtrain = [];
Ptrain = [];
for iTraj = 1:size(xpData,1)
    [~,Xi] = finiteTimeDifference(xpData{iTraj,2},xpData{iTraj,1},3);
    Xtrain = [Xtrain Xi];
    Ptrain = [Ptrain repmat(xpData{iTraj,3},1,size(Xi,2))];
end
xMin = min(Xtrain,[],2); xMax = max(Xtrain,[],2);
pMin = min(trainParams,[],2); pMax = max(trainParams,[],2);
validationX = xMin + (xMax-xMin).*rand(stateDim,nValidation);
validationP = pMin + (pMax-pMin).*rand(paramDim,nValidation);
trueRhsVals = zeros(stateDim,nValidation);
for iPt = 1:nValidation
    trueRhsVals(:,iPt) = rhs(0,validationX(:,iPt),validationP(:,iPt));
end

%% Test trajectories at held-out parameter values
nTestTraj = size(testParams,2)*3;
yDataTest = cell(nTestTraj,2);
testP = zeros(paramDim,nTestTraj);
iTraj = 0;
for iParam = 1:size(testParams,2)
    p = testParams(:,iParam);
    for iIC = 1:3
        iTraj = iTraj+1;
        x0 = 1.0*(2*rand(stateDim,1)-1);
        [~,xTrue] = ode45(@(t,x) rhs(t,x,p),tTest,x0,odeOptsTest);
        yDataTest{iTraj,1} = tTest;
        yDataTest{iTraj,2} = xTrue.';
        testP(:,iTraj) = p;
    end
end

%% Fit and compare
foldState = rng;
RDInfoParaCon = IMDynamicsFlowParaCon(xpData,...
    'Rs_PolyOrd',stateOrder,...
    'Rp_PolyOrd',paramOrder,...
    'origin_fixed',false,...
    'l_vals',lVals,...
    'n_folds',nFolds);

rng(foldState)
RDInfoLegendre = IMDynamicsFlowParaConLegendre(xpData,...
    'Rs_PolyOrd',stateOrder,...
    'Rp_PolyOrd',paramOrder,...
    'origin_fixed',false,...
    'l_vals',lVals,...
    'n_folds',nFolds,...
    'state_degree_penalty',stateDegreePenalty,...
    'parameter_degree_penalty',parameterDegreePenalty,...
    'fig_disp_nfp',-1);

monoRhs = zeros(size(trueRhsVals));
legendreRhs = zeros(size(trueRhsVals));
for iPt = 1:nValidation
    monoRhs(:,iPt) = RDInfoParaCon.reducedDynamics.map(...
        validationX(:,iPt),validationP(:,iPt));
    legendreRhs(:,iPt) = RDInfoLegendre.reducedDynamics.map(...
        validationX(:,iPt),validationP(:,iPt));
end
rhsErrMono = norm(monoRhs-trueRhsVals,'fro')/norm(trueRhsVals,'fro');
rhsErrLegendre = norm(legendreRhs-trueRhsVals,'fro')/norm(trueRhsVals,'fro');

yRecMono = cell(nTestTraj,2);
yRecLegendre = cell(nTestTraj,2);
for iTraj = 1:nTestTraj
    x0 = yDataTest{iTraj,2}(:,1);
    p = testP(:,iTraj);
    xMono = integrateModel(@(x) RDInfoParaCon.reducedDynamics.map(x,p),...
        tTest,x0,odeOptsTest);
    xLegendre = integrateModel(@(x) RDInfoLegendre.reducedDynamics.map(x,p),...
        tTest,x0,odeOptsTest);
    yRecMono{iTraj,1} = tTest;
    yRecMono{iTraj,2} = xMono.';
    yRecLegendre{iTraj,1} = tTest;
    yRecLegendre{iTraj,2} = xLegendre.';
end

NMTEParaCon = mean(computeTrajectoryErrors(yRecMono,yDataTest))*100;
NMTELegendre = mean(computeTrajectoryErrors(yRecLegendre,yDataTest))*100;

fprintf('\nParametric Legendre benchmark\n')
fprintf('state dimension: %d, parameter dimension: %d\n',stateDim,paramDim)
fprintf('state order: %d, parameter order: %d\n',stateOrder,paramOrder)
fprintf('features: monomial %d, Legendre public monomial %d\n',...
    size(RDInfoParaCon.reducedDynamics.coefficients,2),...
    size(RDInfoLegendre.reducedDynamics.coefficients,2))
fprintf('lambda: monomial %.3e, Legendre %.3e\n',...
    RDInfoParaCon.reducedDynamics.l_opt,...
    RDInfoLegendre.reducedDynamics.l_opt)
fprintf('RHS error:  monomial %.3e, Legendre %.3e\n',...
    rhsErrMono,rhsErrLegendre)
fprintf('NMTE:       monomial %.2f%%, Legendre %.2f%%\n',...
    NMTEParaCon,NMTELegendre)
fprintf('|R(0,0)| check for Legendre: %.3e\n',...
    norm(RDInfoLegendre.reducedDynamics.map(zeros(stateDim,1),zeros(paramDim,1))))
fprintf('|R(0,p)| check for Legendre at p=[0.25;-0.4]: %.3e\n',...
    norm(RDInfoLegendre.reducedDynamics.map(zeros(stateDim,1),[0.25;-0.4])))

figure('Color','w')
tiledlayout(1,2,'TileSpacing','compact','Padding','compact')

nexttile
bar([rhsErrMono rhsErrLegendre])
set(gca,'XTickLabel',{'ParaCon','Legendre'})
ylabel('relative RHS error')
grid on

nexttile
bar([NMTEParaCon NMTELegendre])
set(gca,'XTickLabel',{'ParaCon','Legendre'})
ylabel('NMTE (%)','Interpreter','none')
grid on

function x = integrateModel(R,tSamples,x0,opts)
try
    [~,x] = ode45(@(~,x) R(x),tSamples,x0,opts);
catch
    x = NaN(0,numel(x0));
end
end
