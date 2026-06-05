function RDInfo = IMDynamicsFlowParaConLegendre(xpData,varargin)
% RDInfo = IMDynamicsFlowParaConLegendre(xpData,varargin)
% Identification of parameter-dependent reduced flow dynamics
%
%                        \dot{x} = R(x,p)
%
% using a scaled Legendre basis for regression and exposing the fitted
% vector field in the same monomial basis used by IMDynamicsFlowParaCon.
% The call signature and primary fields mirror IMDynamicsFlowParaCon:
% coefficients, phi and exponents refer to monomial features in state and
% parameter variables. Legendre-specific fields describe the regression
% basis and the affine scaling used before evaluating Legendre polynomials.
%
% If origin_fixed is true, the internal Legendre state factors are anchored
% so that each state-dependent feature vanishes at x=0 for all parameter
% values. This preserves the ParaCon convention R(0,p)=0 without exposing
% parameter-only monomial terms.
%
% The Legendre ridge penalty can encode different smoothness assumptions in
% state and parameter directions. A feature with state multi-index alpha_x
% and parameter multi-index alpha_p is penalized by
%
%   (1+|alpha_x|)^(2*state_degree_penalty) *
%   (1+|alpha_p|)^(2*parameter_degree_penalty).
%
% Setting either exponent to zero gives uniform ridge in that direction.

if rem(length(varargin),2) > 0 && length(varargin) > 1
    error('Error on input arguments. Missing or extra arguments.')
end

% Reshape trajectories into regression matrices.
tData = cell(1,size(xpData,1));
xData = cell(1,size(xpData,1));
dxData = cell(1,size(xpData,1));
pData = cell(1,size(xpData,1));
ind_traj = cell(1,size(xpData,1)); idx_end = 0;
for ii = 1:size(xpData,1)
    t_in = xpData{ii,1}; X_in = xpData{ii,2};
    [dXidt,Xi,ti] = finiteTimeDifference(X_in,t_in,3);
    tData{ii} = ti;
    xData{ii} = Xi;
    dxData{ii} = dXidt;
    if size(xpData,2) > 2
        Pi = xpData{ii,3};
        pData{ii} = repmat(Pi,1,length(ti));
    end
    ind_traj{ii} = idx_end+(1:length(ti)); idx_end = idx_end+length(ti);
end
t = [tData{:}];
X = [xData{:}];
dXdt = [dxData{:}];
P = [pData{:}];
options = IMdynamics_options(nargin,varargin,ind_traj,size(X,2));

k = size(X,1); l = size(P,1);
Z = [X; P];
normF = sqrt(size(Z,2));
L2 = (1+options.c1*exp(-options.c2*t)).^(-2);
options.L2 = L2;

Expmat = parametricExponents(k,l,options.Rs_PolyOrd,...
    options.Rp_PolyOrd,options.origin_fixed);
phi = @(z) monomialFeatures(z,Expmat);
Expmat_full = [zeros(1,k+l); Expmat];
phi_full = @(z) [ones(1,size(z,2)); phi(z)];

defaultCenter = isempty(options.LegendreCenter);
defaultScale = isempty(options.LegendreScale);
if defaultCenter
    options.LegendreCenter = 0.5*(max(Z,[],2)+min(Z,[],2));
else
    options.LegendreCenter = options.LegendreCenter(:);
end
if defaultScale
    if defaultCenter
        options.LegendreScale = 0.5*(max(Z,[],2)-min(Z,[],2));
    else
        options.LegendreScale = max(abs(Z-options.LegendreCenter),[],2);
    end
else
    options.LegendreScale = options.LegendreScale(:);
end
options.LegendreScale(abs(options.LegendreScale)<eps) = 1;

B_leg = scaledLegendreToMonomial(Expmat,options.LegendreCenter,...
    options.LegendreScale,k,options.origin_fixed);
legendrePhi = @(z) B_leg*phi(z);
penalty = featurePenalty(Expmat,k,options.state_degree_penalty,...
    options.parameter_degree_penalty);

[constraintFeatures,constraintTargets] = constraintsFromOptions(options,...
    k,l,legendrePhi,Expmat,options.LegendreCenter,...
    options.LegendreScale,options.origin_fixed);

if options.fig_disp_nfp ~= -1
    disp('Estimation of the parameter-dependent reduced dynamics in Legendre basis... ')
end
if strcmp(options.regression,'columnScaledRidge')
    [W_leg,l_opt,Err] = columnScaledConstrainedRidge(...
        legendrePhi(Z)/normF,dXdt/normF,L2,options.idx_folds,...
        options.l_vals,penalty,...
        constraintFeatures,constraintTargets);
elseif strcmp(options.regression,'ridgeRegression')
    [W_leg,l_opt,Err] = ridgeRegression(legendrePhi(Z)/normF,...
        dXdt/normF,L2,options.idx_folds,options.l_vals);
    W_leg = enforceLinearConstraints(W_leg,constraintFeatures,...
        constraintTargets);
else
    error('Unknown regression method "%s".',options.regression)
end

W_const = zeros(k,1);
W_r = W_leg*B_leg;

if l > 0
    R = @(x,p) W_const+W_r*phi([x; repmat(p,1,size(x,2))]);
    DxR = @(x,p) W_r*monomialDerivative([x; p],Expmat,1:k);
    DpR = @(x,p) W_r*monomialDerivative([x; p],Expmat,k+(1:l));
    R_info = assembleStruct(R,W_r,phi,Expmat,l_opt,Err);
    R_info.jacobianState = DxR; R_info.jacobianParameter = DpR;
    T_info = assembleStruct(@(x,p) x,eye(k),@(x) x,eye(k));
else
    R = @(x) W_const+W_r*phi(x);
    DxR = @(x) W_r*monomialDerivative(x,Expmat,1:k);
    R_info = assembleStruct(R,W_r,phi,Expmat,l_opt,Err);
    R_info.jacobianState = DxR;
    T_info = assembleStruct(@(x) x,eye(k),@(x) x,eye(k));
end

R_info.constant = W_const;
R_info.legendreMap = @(z) W_leg*legendrePhi(z);
R_info.legendreCoefficients = W_leg;
R_info.legendrePhi = legendrePhi;
R_info.legendreToMonomial = B_leg;
R_info.monomialPhiFull = phi_full;
R_info.monomialExponentsFull = Expmat_full;
R_info.legendreCenter = options.LegendreCenter;
R_info.legendreScale = options.LegendreScale;
R_info.regression = options.regression;
R_info.stateDegreePenalty = options.state_degree_penalty;
R_info.parameterDegreePenalty = options.parameter_degree_penalty;
R_info.originFixed = options.origin_fixed;
R_info.regressionBasis = 'legendre';
if options.fig_disp_nfp ~= -1
    fprintf('\b Done. \n')
end

iT_info = T_info; N_info = R_info; V = []; d_cont = [];
RDInfo = struct('reducedDynamics',R_info,'inverseTransformation',...
    iT_info,'conjugateDynamics',N_info,'transformation',T_info,...
    'conjugacyStyle',options.style,'dynamicsType','flow',...
    'regressionBasis','legendre-parametric',...
    'eigenvaluesLinPartFlow',d_cont,'eigenvectorsLinPart',V);
end

%---------------------------Subfunctions-----------------------------------

function str_out = assembleStruct(fun,W,phi,Emat,varargin)
PolyOrder = sum(Emat(end,:));
if isempty(varargin) == 0
    str_out = struct('map',fun,'coefficients',W,'polynomialOrder',...
        PolyOrder,'phi',phi,'exponents',Emat,'l_opt',varargin{1},...
        'CV_error',varargin{2});
else
    str_out = struct('map',fun,'coefficients',W,'polynomialOrder',...
        PolyOrder,'phi',phi,'exponents',Emat);
end
end

function Expmat = parametricExponents(k,l,Rs,Rp,origin_fixed)
ExpmatState = [];
for iOrd = 1:Rs
    ExpmatState = [ExpmatState; multivariateExponents(k,iOrd)];
end
if l > 0
    ExpmatParam = zeros(1,l);
    for iOrd = 1:Rp
        ExpmatParam = [ExpmatParam; multivariateExponents(l,iOrd)];
    end
    Expmat = [];
    for iRow = 1:size(ExpmatState,1)
        Expmat = [Expmat; repmat(ExpmatState(iRow,:),...
            size(ExpmatParam,1),1) ExpmatParam];
    end
    if origin_fixed == 0
        Expmat = [zeros(size(ExpmatParam,1)-1,k) ...
            ExpmatParam(2:end,:); Expmat];
    end
else
    Expmat = ExpmatState;
end
end

function Phi = monomialFeatures(Z,Expmat)
if isvector(Z); Z = Z(:); end
Phi = ones(size(Expmat,1),size(Z,2));
for iVar = 1:size(Expmat,2)
    Phi = Phi.*(Z(iVar,:).^Expmat(:,iVar));
end
end

function D = monomialDerivative(z,Expmat,vars)
z = z(:);
D = zeros(size(Expmat,1),numel(vars));
for j = 1:numel(vars)
    iVar = vars(j);
    active = Expmat(:,iVar) > 0;
    E = Expmat;
    E(active,iVar) = E(active,iVar)-1;
    vals = monomialFeatures(z,E);
    D(active,j) = Expmat(active,iVar).*vals(active);
end
end

function [C,T] = constraintsFromOptions(options,k,l,legendrePhi,...
    Expmat_full,center,scale,origin_fixed)
C = [];
T = [];

fixedPoints = options.fixed_points;
if isempty(fieldnames(fixedPoints)) == 0
    nFixedPoints = length(fixedPoints);
    Xo = reshape([fixedPoints(:).reducedState],k,nFixedPoints);
    if isfield(fixedPoints,'parameter') == 1
        Po = reshape([fixedPoints(:).parameter],l,nFixedPoints);
    else
        Po = zeros(l,nFixedPoints);
    end
    C = [C legendrePhi([Xo; Po])];
    T = [T zeros(k,nFixedPoints)];
end

linearParts = options.lin_part;
if isempty(fieldnames(linearParts)) == 0
    for iCon = 1:length(linearParts)
        if strcmp(options.type,'dynamics') == 1
            Wlin = linearParts(iCon).reducedDynamics;
        else
            Wlin = linearParts(iCon).parametrization;
        end
        if isfield(linearParts,'parameter') == 1
            p_i = linearParts(iCon).parameter;
        else
            p_i = zeros(l,1);
        end
        if isfield(linearParts,'reducedState') == 1
            x_i = linearParts(iCon).reducedState;
        else
            x_i = zeros(k,1);
        end
        Dleg = legendreDerivative([x_i; p_i],Expmat_full,center,scale,...
            k,origin_fixed,1:k);
        C = [C Dleg];
        T = [T Wlin];
    end
end
end

function B = scaledLegendreToMonomial(Expmat,center,scale,k,origin_fixed)
nBasis = size(Expmat,1);
B = zeros(nBasis,nBasis);
for iBasis = 1:nBasis
    alpha = Expmat(iBasis,:);
    terms = legendreProductTerms(alpha,center,scale);
    if origin_fixed && any(alpha(1:k)>0)
        paramTerms = legendreProductTerms([zeros(1,k) alpha(k+1:end)],...
            center,scale);
        anchor = legendreProductValueAtZero(alpha(1:k),center(1:k),...
            scale(1:k));
        paramTerms.coeff = -anchor*paramTerms.coeff;
        terms.exp = [terms.exp; paramTerms.exp];
        terms.coeff = [terms.coeff; paramTerms.coeff];
    elseif ~origin_fixed
        terms.exp = [terms.exp; zeros(1,size(Expmat,2))];
        terms.coeff = [terms.coeff; -legendreProductValueAtZero(alpha,...
            center,scale)];
    end
    for iTerm = 1:size(terms.exp,1)
        idx = find(ismember(Expmat,terms.exp(iTerm,:),'rows'),1);
        if ~isempty(idx)
            B(iBasis,idx) = B(iBasis,idx)+terms.coeff(iTerm);
        end
    end
end
end

function terms = legendreProductTerms(alpha,center,scale)
nVar = numel(alpha);
termExp = zeros(1,nVar);
termCoeff = 1;
for iVar = 1:nVar
    p = shiftedLegendreCoefficients(alpha(iVar),center(iVar),...
        scale(iVar));
    p = sqrt((2*alpha(iVar)+1)/2)*p;
    nextExp = [];
    nextCoeff = [];
    for iTerm = 1:size(termExp,1)
        for iDeg = 0:numel(p)-1
            if abs(p(iDeg+1)) > eps
                expNew = termExp(iTerm,:);
                expNew(iVar) = expNew(iVar)+iDeg;
                nextExp = [nextExp; expNew];
                nextCoeff = [nextCoeff; termCoeff(iTerm)*p(iDeg+1)];
            end
        end
    end
    termExp = nextExp;
    termCoeff = nextCoeff;
end
terms = struct('exp',termExp,'coeff',termCoeff);
end

function val = legendreProductValueAtZero(alpha,center,scale)
val = 1;
for iVar = 1:numel(alpha)
    p = shiftedLegendreCoefficients(alpha(iVar),center(iVar),...
        scale(iVar));
    val = val*sqrt((2*alpha(iVar)+1)/2)*p(1);
end
end

function D = legendreDerivative(z,Expmat,center,scale,k,origin_fixed,vars)
z = z(:);
nBasis = size(Expmat,1);
D = zeros(nBasis,numel(vars));
h = 1e-6*(1+abs(z));
for j = 1:numel(vars)
    zp = z; zm = z;
    zp(vars(j)) = zp(vars(j))+h(vars(j));
    zm(vars(j)) = zm(vars(j))-h(vars(j));
    Bp = scaledLegendreValues(zp,Expmat,center,scale,k,origin_fixed);
    Bm = scaledLegendreValues(zm,Expmat,center,scale,k,origin_fixed);
    D(:,j) = (Bp-Bm)/(2*h(vars(j)));
end
end

function vals = scaledLegendreValues(z,Expmat,center,scale,k,origin_fixed)
vals = zeros(size(Expmat,1),1);
for iBasis = 1:size(Expmat,1)
    alpha = Expmat(iBasis,:);
    vals(iBasis) = evalLegendreProduct(z,alpha,center,scale);
    if origin_fixed && any(alpha(1:k)>0)
        vals(iBasis) = vals(iBasis) - ...
            legendreProductValueAtZero(alpha(1:k),center(1:k),scale(1:k)) * ...
            evalLegendreProduct(z,[zeros(1,k) alpha(k+1:end)],...
            center,scale);
    elseif ~origin_fixed
        vals(iBasis) = vals(iBasis) - legendreProductValueAtZero(alpha,...
            center,scale);
    end
end
end

function val = evalLegendreProduct(z,alpha,center,scale)
val = 1;
for iVar = 1:numel(alpha)
    s = (z(iVar)-center(iVar))/scale(iVar);
    L = legendreValues(s,alpha(iVar));
    val = val*sqrt((2*alpha(iVar)+1)/2)*L(alpha(iVar)+1);
end
end

function L = legendreValues(x,order)
L = zeros(order+1,1);
L(1) = 1;
if order >= 1
    L(2) = x;
end
for n = 1:order-1
    L(n+2) = ((2*n+1)*x*L(n+1)-n*L(n))/(n+1);
end
end

function p = shiftedLegendreCoefficients(order,center,scale)
p_s = legendreCoefficients(order);
p = 0;
base = [-center/scale 1/scale];
for ii = 0:order
    p = localPolyAdd(p,p_s(ii+1)*localPolyPower(base,ii));
end
end

function p = legendreCoefficients(order)
if order == 0
    p = 1;
    return
end
p0 = 1;
p1 = [0 1];
if order == 1
    p = p1;
    return
end
for n = 1:order-1
    xpn = conv([0 1],p1);
    p2 = localPolyAdd(((2*n+1)/(n+1))*xpn,-(n/(n+1))*p0);
    p0 = p1;
    p1 = p2;
end
p = p1;
end

function p = localPolyPower(base,powerValue)
p = 1;
for ii = 1:powerValue
    p = conv(p,base);
end
end

function p = localPolyAdd(a,b)
n = max(numel(a),numel(b));
p = zeros(1,n);
p(1:numel(a)) = p(1:numel(a))+a;
p(1:numel(b)) = p(1:numel(b))+b;
end

function penalty = featurePenalty(Expmat,k,state_weight,param_weight)
state_degree = sum(Expmat(:,1:k),2);
param_degree = sum(Expmat(:,k+1:end),2);
penalty = (1+state_degree).^(2*state_weight) .* ...
    (1+param_degree).^(2*param_weight);
end

function [W,l_opt,Err] = columnScaledConstrainedRidge(Phi,Y,L,idx_folds,...
    l_vals,penalty,constraintFeatures,constraintTargets)
X = Phi.';
Y = Y.';
L = L(:);

scale = max(abs(X),[],1).';
scale(abs(scale)<1e-12) = 1;
Xs = X ./ scale.';
XL = L .* Xs;
if isempty(constraintFeatures)
    Cscaled = [];
else
    Cscaled = constraintFeatures ./ scale;
end

if isempty(idx_folds)
    l_opt = l_vals(1);
    Err = [];
else
    Err = zeros(numel(l_vals),numel(idx_folds));
    all_idx = 1:size(Xs,1);
    for il = 1:numel(l_vals)
        reg = diag(l_vals(il)*penalty(:));
        for ifold = 1:numel(idx_folds)
            test = idx_folds{ifold};
            train = all_idx;
            train(test) = [];
            if isempty(Cscaled)
                B = solveRidgeLeastSquares(Xs(train,:),Y(train,:),...
                    L(train),l_vals(il),penalty);
            else
                A = XL(train,:)'*Xs(train,:) + reg;
                rhs = XL(train,:)'*Y(train,:);
                B = solveConstrainedRidge(A,rhs,Cscaled,...
                    constraintTargets);
            end
            E = Y(test,:) - Xs(test,:)*B;
            Err(il,ifold) = mean(sqrt(sum(abs(E).^2,2)));
        end
    end
    [~,best] = min(mean(Err,2));
    l_opt = l_vals(best);
end

reg = diag(l_opt*penalty(:));
if isempty(Cscaled)
    B = solveRidgeLeastSquares(Xs,Y,L,l_opt,penalty);
else
    A = XL'*Xs + reg;
    rhs = XL'*Y;
    B = solveConstrainedRidge(A,rhs,Cscaled,constraintTargets);
end
W = (B ./ scale).';
end

function B = solveRidgeLeastSquares(X,Y,L,lambdaValue,penalty)
sqrtL = sqrt(max(L(:),0));
reg = diag(sqrt(lambdaValue*penalty(:)));
A = [sqrtL.*X; reg];
b = [sqrtL.*Y; zeros(size(X,2),size(Y,2))];
B = A\b;
end

function B = solveConstrainedRidge(A,rhs,C,target)
if isempty(C)
    B = A\rhs;
    return
end
KKT = [A C; C' zeros(size(C,2))];
sol = KKT\[rhs; target'];
B = sol(1:size(A,1),:);
end

function W = enforceLinearConstraints(W,C,target)
if isempty(C)
    return
end
for iCon = 1:size(C,2)
    c = C(:,iCon);
    residual = W*c-target(:,iCon);
    denom = c'*c;
    if denom > eps
        W = W-(residual/denom)*c';
    end
end
end

function options = IMdynamics_options(nargin_o,varargin_o,idx_traj,Ndata)
options = struct('style','default','Rs_PolyOrd',1,'Rp_PolyOrd',0,...
    'c1',0,'c2',0,'fixed_points',struct(),'origin_fixed',false,...
    'lin_part',struct(),'L2',[],'n_folds',0,'l_vals',1e-4,...
    'idx_folds',[],'fold_style',[],'type','dynamics',...
    'LegendreCenter',[],'LegendreScale',[],...
    'regression','columnScaledRidge','state_degree_penalty',0,...
    'parameter_degree_penalty',0,'fig_disp_nfp',0);
if nargin_o == 2
    options.Rs_PolyOrd = varargin_o{:};
end
if nargin_o > 2
    for ii = 1:length(varargin_o)/2
        options = setfield(options,varargin_o{2*ii-1},...
            varargin_o{2*ii});
    end
    if options.n_folds > 1
        if strcmp(options.fold_style,'traj') == 1
            options.n_folds = length(idx_traj);
            idx_folds = idx_traj;
        else
            idx_folds = cell(options.n_folds,1);
            ind_perm = randperm(Ndata);
            fold_size = floor(Ndata/options.n_folds);
            for ii = 1:options.n_folds-1
                idx_folds{ii} = ind_perm(1+(ii-1)*fold_size:ii*fold_size);
            end
            idx_folds{options.n_folds} = ind_perm(...
                1+(options.n_folds-1)*fold_size:length(ind_perm));
        end
        options.idx_folds = idx_folds;
    end
end
end
