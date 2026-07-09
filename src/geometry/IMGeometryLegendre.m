function [IMInfo,IMChart,IMParam] = IMGeometryLegendre(yData,SSMDim,M,varargin)
%IMGeometryLegendre Geometry parametrization on a scaled Legendre basis.
%
%   [IMInfo,IMChart,IMParam] = IMGeometryLegendre(yData,SSMDim,M,...)
%
%   Legendre-basis analogue of IMGeometry. The standard custom-coordinate
%   use case is a drop-in replacement for IMGeometry:
%
%      'chart'              user chart y -> eta
%      'reducedCoordinates' user reduced coordinates eta
%
%   If no chart or reduced coordinates are supplied, the natural chart is
%   selected by the original IMGeometry/IMGeometryGraphT0 workflow, and the
%   parametrization is then refitted with IMGeometryParaConLegendre. If
%   origin_fixed is true, IMParam(0)=0 is imposed through a linear equality
%   constraint.

if rem(length(varargin),2) > 0 && length(varargin) > 1
    error('Error on input arguments. Missing or extra arguments.')
end

opts = localOptions(varargin{:});
opts.eta_order = M;
opts.delta_order = 0;

[Y,t] = unpackYData(yData);

natural_chart = false;
if ~isempty(opts.reducedCoordinates)
    Q = unpackReducedCoordinates(opts.reducedCoordinates);
elseif ~isempty(opts.chart)
    Q = opts.chart(Y);
else
    graphArgs = optionArgsForGraph(opts);
    [~,IMChartNatural,~] = IMGeometry(yData,SSMDim,M,graphArgs{:});
    opts.chart = IMChartNatural;
    Q = opts.chart(Y);
    natural_chart = true;
end

if isempty(opts.chart)
    IMChart = @(y) Q*pinv(Y)*y;
else
    IMChart = opts.chart;
end

etaData = {t,Q};
yCell = {t,Y};
args = optionArgsForParaCon(opts);
[IMInfoPara,~] = IMGeometryParaConLegendre(yCell,etaData,args{:});

paramInfoPara = IMInfoPara.parametrization;
degree = sum(paramInfoPara.exponents(:,1:SSMDim),2);
linear_idx = degree == 1;
nonlinear_idx = degree >= 2;
linearExponents = paramInfoPara.exponents(linear_idx,1:SSMDim);
[~,linear_order] = sortLinearExponents(linearExponents,SSMDim);
linear_cols = find(linear_idx);
linear_cols = linear_cols(linear_order);
tangent = paramInfoPara.coefficients(:,linear_cols);
nonlinearExponents = paramInfoPara.exponents(nonlinear_idx,1:SSMDim);
nonlinearCoefficients = paramInfoPara.coefficients(:,nonlinear_idx);
phi = @(eta) monomialFeatures(eta,nonlinearExponents);
IMParam = @(eta) tangent*eta + nonlinearCoefficients*phi(eta);
paramInfo = struct('map',IMParam, ...
    'polynomialOrder',M, ...
    'dimension',SSMDim, ...
    'tangentSpaceAtOrigin',tangent, ...
    'nonlinearCoefficients',nonlinearCoefficients, ...
    'phi',phi, ...
    'exponents',nonlinearExponents, ...
    'l',opts.l_vals, ...
    'c1',opts.c1, ...
    'c2',opts.c2);
if natural_chart
    paramInfo.mapOut = @(q) q;
end
paramInfo.monomialCoefficientsFull = paramInfoPara.coefficients;
paramInfo.monomialPhiFull = @(eta) paramInfoPara.phi(eta);
paramInfo.monomialExponentsFull = paramInfoPara.exponents(:,1:SSMDim);
if isfield(paramInfoPara,'legendreCoefficients')
    paramInfo.legendreMap = @(eta) paramInfoPara.legendreMap(eta,zeros(0,size(eta,2)));
    paramInfo.legendreCoefficients = paramInfoPara.legendreCoefficients;
    paramInfo.legendrePhi = @(eta) paramInfoPara.legendrePhi(eta);
    paramInfo.legendreToMonomial = paramInfoPara.legendreToMonomial;
    paramInfo.legendreCenter = paramInfoPara.legendreCenter;
    paramInfo.legendreScale = paramInfoPara.legendreScale;
    paramInfo.regressionBasis = 'legendre';
end

IMInfo = IMInfoPara;
if natural_chart
    IMInfo.chart = struct('map',IMChart,'polynomialOrder',1);
else
    IMInfo.chart = struct('map',IMChart);
end
IMInfo.parametrization = paramInfo;
end

function [degree1,order] = sortLinearExponents(exponents,k)
degree1 = eye(k);
order = zeros(k,1);
for jj = 1:k
    idx = find(ismember(exponents,degree1(jj,:),'rows'),1);
    if isempty(idx)
        error('Could not find linear monomial exponent for state coordinate %d.',jj);
    end
    order(jj) = idx;
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

function [Y,t] = unpackYData(yData)
if iscell(yData)
    Y = [];
    t = [];
    for ii = 1:size(yData,1)
        Y = [Y yData{ii,2}]; %#ok<AGROW>
        t = [t yData{ii,1}(:).']; %#ok<AGROW>
    end
else
    Y = yData;
    t = 1:size(Y,2);
end
end

function Q = unpackReducedCoordinates(reducedCoordinates)
if iscell(reducedCoordinates)
    Q = [];
    for ii = 1:size(reducedCoordinates,1)
        Q = [Q reducedCoordinates{ii,2}]; %#ok<AGROW>
    end
else
    Q = reducedCoordinates;
end
end

function opts = localOptions(varargin)
opts = struct('style','natural','chart',[],'reducedCoordinates',[], ...
    'l_vals',1e-4,'l',[],'n_folds',0,'fold_style','default', ...
    'c1',0,'c2',0,'feature_limit',inf,'eta_center',[], ...
    'eta_scale',[],'regression','columnScaledRidge', ...
    'state_degree_penalty',0,'origin_fixed',false,'fixed_points',struct());
for ii = 1:2:numel(varargin)
    opts.(varargin{ii}) = varargin{ii+1};
end
if ~isempty(opts.l)
    opts.l_vals = opts.l;
end
end

function args = optionArgsForParaCon(opts)
names = fieldnames(opts);
drop = ["style","chart","reducedCoordinates","l","Display", ...
    "OptimalityTolerance","MaxIter","MaxFunctionEvaluations", ...
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
