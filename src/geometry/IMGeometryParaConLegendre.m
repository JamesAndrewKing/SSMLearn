function [IMInfo,V,IMChart] = IMGeometryParaConLegendre(yData,etaData,varargin)
%IMGeometryParaConLegendre Parametric geometry fit on a Legendre basis.
%
%   IMInfo = IMGeometryParaConLegendre(yData,etaData,...)
%   IMInfo = IMGeometryParaConLegendre(yData,SSMDim,M,...)
%
%   Parametric analogue of IMGeometryLegendre. If reduced coordinates are
%   supplied in etaData, the call is a drop-in replacement for
%   IMGeometryParaCon with a scaled Legendre regression basis:
%
%      yData{i,1}   time samples
%      yData{i,2}   observable trajectory
%      etaData{i,1} time samples
%      etaData{i,2} reduced trajectory
%      etaData{i,3} parameter vector for trajectory i, optional
%
%   If etaData is omitted, empty, or the second input is SSMDim, the chart
%   is selected by the original IMGeometry natural-chart workflow and
%   stored in IMInfo.chart. In this higher-level mode parameter vectors are
%   read from yData{i,3} or from the 'paramData' option.
%
%   The fitted map is V(eta,p). If origin_fixed is true, V(0,p)=0 is
%   imposed by linear equality constraints on the Legendre coefficients.

if nargin < 2
    etaData = [];
end

positionalSSMDim = [];
positionalOrder = [];
if isnumeric(etaData) && isscalar(etaData)
    positionalSSMDim = etaData;
    etaData = [];
    if ~isempty(varargin) && isnumeric(varargin{1}) && isscalar(varargin{1})
        positionalOrder = varargin{1};
        varargin = varargin(2:end);
    end
end

if rem(length(varargin),2) > 0
    error('Error on input arguments. Missing or extra arguments.')
end

opts = localOptions(varargin{:});
if ~isempty(positionalSSMDim)
    opts.SSMDim = positionalSSMDim;
end
if ~isempty(positionalOrder)
    opts.eta_order = positionalOrder;
    opts.Rs_PolyOrd = positionalOrder;
end
if isfield(opts,'Rs_PolyOrd') && ~isempty(opts.Rs_PolyOrd)
    opts.eta_order = opts.Rs_PolyOrd;
end
if isfield(opts,'Rp_PolyOrd') && ~isempty(opts.Rp_PolyOrd)
    opts.delta_order = opts.Rp_PolyOrd;
end

[etaData,chartInfo,IMChart] = completeReducedCoordinates(yData,etaData,opts);

paramData = struct([]);
for ii = 1:size(etaData,1)
    n_i = numel(etaData{ii,1});
    paramData(ii).theta = zeros(1,n_i);
    if size(etaData,2) >= 3 && ~isempty(etaData{ii,3})
        p_i = etaData{ii,3};
        if size(p_i,2) == 1
            paramData(ii).delta = repmat(p_i(:),1,n_i);
        else
            paramData(ii).delta = p_i;
        end
    else
        paramData(ii).delta = zeros(0,n_i);
    end
    if isfield(opts,'traj_weight') && ~isempty(opts.traj_weight)
        paramData(ii).traj_weight = opts.traj_weight;
    end
end

args = optionArgsForFourier(opts);
[IMInfoFourier,~] = IMGeometryParaConFourierLegendre(yData,etaData,paramData,args{:}, ...
    'K',0,'harmonic_penalty',0,'amplitude_gated_harmonics',false);

paramInfoFull = IMInfoFourier.parametrization;
Expmat = parametricGeometryExponents(paramInfoFull.dimensionState, ...
    paramInfoFull.dimensionParam,paramInfoFull.polynomialOrderState, ...
    paramInfoFull.polynomialOrderParam,opts.origin_fixed);
[keep,missing] = matchExponentRows(paramInfoFull.combined_exponents,Expmat);
if any(missing)
    error('Could not map Legendre geometry coefficients to monomial compatibility exponents.');
end
W_mono = paramInfoFull.coefficients(:,keep);
phi = @(z) monomialFeatures(z,Expmat);
V = @(eta,p) W_mono*phi([eta; repmat(p,1,size(eta,2))]);
DxV = @(eta,p) finiteDifferenceStateDerivative(V,eta,p);

paramInfo = struct('map',V,'coefficients',W_mono, ...
    'polynomialOrderState',paramInfoFull.polynomialOrderState, ...
    'polynomialOrderParam',paramInfoFull.polynomialOrderParam, ...
    'dimensionState',paramInfoFull.dimensionState, ...
    'dimensionParam',paramInfoFull.dimensionParam, ...
    'phi',phi,'exponents',Expmat, ...
    'l_opt',paramInfoFull.l_opt,'CV_error',paramInfoFull.CV_error, ...
    'derivativeState',DxV);
paramInfo.monomialCoefficientsFull = paramInfoFull.coefficients;
paramInfo.monomialPhiFull = @(z) paramInfoFull.phi(z(1:paramInfoFull.dimensionState,:), ...
    0,z(paramInfoFull.dimensionState+1:end,:));
paramInfo.monomialExponentsFull = paramInfoFull.combined_exponents;
if isfield(paramInfoFull,'legendreCoefficients')
    paramInfo.legendreMap = @(eta,p) paramInfoFull.legendreMap(eta,0,p);
    paramInfo.legendreCoefficients = paramInfoFull.legendreCoefficients;
    paramInfo.legendrePhi = @(z) paramInfoFull.legendrePhi( ...
        z(1:paramInfoFull.dimensionState,:),0, ...
        z(paramInfoFull.dimensionState+1:end,:));
    paramInfo.legendreToMonomial = paramInfoFull.legendreToMonomial;
    paramInfo.legendreCenter = paramInfoFull.legendreCenter;
    paramInfo.legendreScale = paramInfoFull.legendreScale;
    paramInfo.regressionBasis = 'legendre';
end
IMInfo = struct('chart',chartInfo,'parametrization',paramInfo);
end

function [etaData,chartInfo,IMChart] = completeReducedCoordinates(yData,etaData,opts)
IMChart = [];
chartInfo = struct();
if ~isempty(etaData)
    if ~isempty(opts.chart)
        IMChart = opts.chart;
        chartInfo = struct('map',IMChart);
    end
    return
end

if ~isempty(opts.reducedCoordinates)
    etaData = opts.reducedCoordinates;
    if ~isempty(opts.chart)
        IMChart = opts.chart;
        chartInfo = struct('map',IMChart);
    end
else
    if isempty(opts.chart)
        if isempty(opts.SSMDim)
            error('SSMDim is required when etaData or a chart is not supplied.');
        end
        graphArgs = optionArgsForGraph(opts);
        [~,IMChart,~] = IMGeometry(yData,opts.SSMDim,opts.eta_order,graphArgs{:});
    else
        IMChart = opts.chart;
    end
    etaData = reducedDataFromChart(yData,IMChart,opts);
    chartInfo = struct('map',IMChart,'polynomialOrder',1);
end
end

function etaData = reducedDataFromChart(yData,chart,opts)
etaData = cell(size(yData,1),3);
for ii = 1:size(yData,1)
    etaData{ii,1} = yData{ii,1};
    etaData{ii,2} = chart(yData{ii,2});
    etaData{ii,3} = parameterForTrajectory(yData,ii,opts);
end
end

function p = parameterForTrajectory(yData,ii,opts)
if ~isempty(opts.paramData)
    if iscell(opts.paramData)
        p = opts.paramData{ii};
    else
        p = opts.paramData(ii).parameter;
    end
elseif size(yData,2) >= 3 && ~isempty(yData{ii,3})
    p = yData{ii,3};
else
    p = zeros(0,1);
end
end

function Expmat = parametricGeometryExponents(k,l,Rs,Rp,origin_fixed)
ExpmatState = [];
for iOrd = 1:Rs
    ExpmatState = [ExpmatState; multivariateExponents(k,iOrd)]; %#ok<AGROW>
end
if l > 0
    ExpmatParam = zeros(1,l);
    for iOrd = 1:Rp
        ExpmatParam = [ExpmatParam; multivariateExponents(l,iOrd)]; %#ok<AGROW>
    end
    Expmat = [];
    for iRow = 1:size(ExpmatState,1)
        Expmat = [Expmat; repmat(ExpmatState(iRow,:), ...
            size(ExpmatParam,1),1) ExpmatParam]; %#ok<AGROW>
    end
    if ~origin_fixed
        Expmat = [zeros(size(ExpmatParam,1)-1,k) ExpmatParam(2:end,:); Expmat];
    end
else
    Expmat = ExpmatState;
end
end

function [idx,missing] = matchExponentRows(source,target)
idx = zeros(size(target,1),1);
missing = false(size(target,1),1);
for ii = 1:size(target,1)
    jj = find(ismember(source,target(ii,:),'rows'),1);
    if isempty(jj)
        missing(ii) = true;
    else
        idx(ii) = jj;
    end
end
end

function Phi = monomialFeatures(Z,Expmat)
if isvector(Z)
    Z = Z(:);
end
Phi = ones(size(Expmat,1),size(Z,2));
for iVar = 1:size(Expmat,2)
    Phi = Phi.*(Z(iVar,:).^Expmat(:,iVar));
end
end

function J = finiteDifferenceStateDerivative(V,eta,p)
eta = eta(:);
p = p(:);
h = 1e-6*(1 + norm(eta));
y0 = V(eta,p);
J = zeros(numel(y0),numel(eta));
for jj = 1:numel(eta)
    d = zeros(size(eta));
    d(jj) = h;
    J(:,jj) = (V(eta+d,p) - V(eta-d,p))/(2*h);
end
end

function opts = localOptions(varargin)
opts = struct('SSMDim',[],'chart',[],'reducedCoordinates',[], ...
    'paramData',[],'Rs_PolyOrd',[],'Rp_PolyOrd',[], ...
    'eta_order',3,'delta_order',2,'l_vals',1e-4,'n_folds',0, ...
    'fold_style','default','c1',0,'c2',0,'feature_limit',inf, ...
    'eta_center',[],'eta_scale',[],'delta_center',[],'delta_scale',[], ...
    'eta_exp',[],'d_exp',[],'regression','columnScaledRidge', ...
    'state_degree_penalty',0,'delta_degree_penalty',0, ...
    'origin_fixed',false,'fixed_points',struct(),'traj_weight',[], ...
    'Display','off','OptimalityTolerance',[], ...
    'MaxIter',[],'MaxFunctionEvaluations',[], ...
    'SpecifyObjectiveGradient',[],'SpecifyConstraintGradient',[], ...
    'CheckGradients',[],'V_e',[],'V_e0',[],'H0',[]);
for ii = 1:2:numel(varargin)
    opts.(varargin{ii}) = varargin{ii+1};
end
end

function E = multivariateExponents(n,order)
if n == 0
    E = zeros(1,0);
    return
end
if n == 1
    E = order;
    return
end
E = [];
for k = 0:order
    sub = multivariateExponents(n-1,order-k);
    E = [E; [k*ones(size(sub,1),1) sub]]; %#ok<AGROW>
end
end

function args = optionArgsForFourier(opts)
names = fieldnames(opts);
drop = ["SSMDim","chart","reducedCoordinates","paramData", ...
    "Rs_PolyOrd","Rp_PolyOrd","traj_weight", ...
    "Display","OptimalityTolerance","MaxIter","MaxFunctionEvaluations", ...
    "SpecifyObjectiveGradient","SpecifyConstraintGradient", ...
    "CheckGradients","V_e","V_e0","H0"];
args = {};
for ii = 1:numel(names)
    name = string(names{ii});
    if any(name == drop)
        continue
    end
    args(end+1:end+2) = {char(name),opts.(names{ii})};
end
end

function args = optionArgsForGraph(opts)
args = {'l',opts.l_vals,'c1',opts.c1,'c2',opts.c2};
optional = ["Display","OptimalityTolerance","MaxIter", ...
    "MaxFunctionEvaluations","SpecifyObjectiveGradient", ...
    "SpecifyConstraintGradient","CheckGradients","V_e","V_e0","H0"];
for ii = 1:numel(optional)
    name = char(optional(ii));
    if isfield(opts,name) && ~isempty(opts.(name))
        args(end+1:end+2) = {name,opts.(name)};
    end
end
end
