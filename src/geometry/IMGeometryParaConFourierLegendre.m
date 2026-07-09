function [IMInfo,V] = IMGeometryParaConFourierLegendre(yData,etaData,paramData,varargin)
%IMGeometryParaConFourierLegendre Parametric Fourier-Legendre geometry fit.
%
%   [IMInfo,V] = IMGeometryParaConFourierLegendre(yData,etaData,paramData,...)
%
%   Fits the parametrization
%
%       y = V(eta,theta,delta) = W * Phi(eta,exp(i theta),delta)
%
%   where Phi is a Fourier expansion in theta and scaled Legendre products
%   in eta and delta. This is the geometry analogue of
%   IMDynamicsFlowParaConFourierLegendre: the feature library and ridge
%   penalties are the same, but the regression target is the observable
%   state y rather than a vector field.
%
%   yData{i,1}      time samples
%   yData{i,2}      observable trajectory, n x m_i
%   etaData{i,1}    time samples
%   etaData{i,2}    reduced trajectory, k x m_i
%   paramData(i).theta forcing phase at the same samples
%   paramData(i).delta parameter coordinates at the same samples
%
%   If origin_fixed is true, V(0,theta,delta)=0 is imposed as a linear
%   equality constraint on the regression coefficients for every Fourier
%   harmonic and parameter basis function represented in Phi.

if rem(length(varargin),2) > 0
    error('Missing or extra input arguments.');
end

opts = localOptions(varargin{:});

[T,Eta,Y,Theta,Delta,Weights,idx_traj] = packGeometryData(yData,etaData,paramData,opts);

k = size(Eta,1);
l = size(Delta,1);
if isempty(opts.eta_exp)
    opts.eta_exp = multiIndexRange(k,0,opts.eta_order);
end
if isempty(opts.d_exp)
    opts.d_exp = multiIndexRange(l,0,opts.delta_order);
end

[opts.eta_center,opts.eta_scale] = scalingFromData(Eta,opts.eta_center,opts.eta_scale);
[opts.delta_center,opts.delta_scale] = scalingFromData(Delta,opts.delta_center,opts.delta_scale);

Phi = parametricFourierFeatures(Eta,Theta,Delta,opts.eta_center,opts.eta_scale, ...
    opts.delta_center,opts.delta_scale,opts.eta_exp,opts.d_exp,opts.K, ...
    opts.feature_limit,opts.amplitude_gated_harmonics,opts.amplitude_gate_index, ...
    opts.amplitude_gate_scale,opts.basis);

L2 = Weights .* (1 + opts.c1*exp(-opts.c2*T)).^(-2);
idx_folds = [];
if opts.n_folds > 1
    if strcmp(opts.fold_style,'traj')
        idx_folds = idx_traj;
    else
        idx_folds = randomFolds(numel(T),opts.n_folds);
    end
end

penalty = featurePenalty(opts.eta_exp,opts.d_exp,opts.K,opts.harmonic_penalty, ...
    opts.state_degree_penalty,opts.delta_degree_penalty);
[constraintFeatures,constraintTargets] = geometryConstraints(opts,k,l,size(Y,1));

if strcmp(opts.regression,'columnScaledRidge')
    [W,l_opt,Err] = columnScaledConstrainedRidge(Phi,Y,L2,idx_folds, ...
        opts.l_vals,penalty,constraintFeatures,constraintTargets);
elseif strcmp(opts.regression,'ridgeRegression')
    [W,l_opt,Err] = ridgeRegressionWithConstraints(Phi,Y,L2,idx_folds, ...
        opts.l_vals,penalty,constraintFeatures,constraintTargets);
else
    error('Unknown regression method "%s".',opts.regression);
end

phi_basis = @(eta,theta,delta) parametricFourierFeatures(eta,theta,delta, ...
    opts.eta_center,opts.eta_scale,opts.delta_center,opts.delta_scale, ...
    opts.eta_exp,opts.d_exp,opts.K,opts.feature_limit, ...
    opts.amplitude_gated_harmonics,opts.amplitude_gate_index, ...
    opts.amplitude_gate_scale,opts.basis);

[B_to_monomial,combined_exponents] = basisToMonomialMatrix(opts,k,l);
W_public = W*B_to_monomial;
phi = @(eta,theta,delta) parametricFourierFeatures(eta,theta,delta, ...
    opts.eta_center,opts.eta_scale,opts.delta_center,opts.delta_scale, ...
    opts.eta_exp,opts.d_exp,opts.K,opts.feature_limit, ...
    opts.amplitude_gated_harmonics,opts.amplitude_gate_index, ...
    opts.amplitude_gate_scale,'monomial');
V = @(eta,theta,delta) real(W_public*phi(eta,theta,delta));

fitError = norm(real(W*Phi)-Y,'fro')/max(norm(Y,'fro'),eps);
paramInfo = struct( ...
    'map',V, ...
    'coefficients',W_public, ...
    'phi',phi, ...
    'eta_exponents',opts.eta_exp, ...
    'delta_exponents',opts.d_exp, ...
    'combined_exponents',combined_exponents, ...
    'K',opts.K, ...
    'polynomialOrderState',opts.eta_order, ...
    'polynomialOrderParam',opts.delta_order, ...
    'dimensionState',k, ...
    'dimensionParam',l, ...
    'regression',opts.regression, ...
    'harmonic_penalty',opts.harmonic_penalty, ...
    'stateDegreePenalty',opts.state_degree_penalty, ...
    'deltaDegreePenalty',opts.delta_degree_penalty, ...
    'etaCenter',opts.eta_center, ...
    'etaScale',opts.eta_scale, ...
    'deltaCenter',opts.delta_center, ...
    'deltaScale',opts.delta_scale, ...
    'originFixed',opts.origin_fixed, ...
    'basis',opts.basis, ...
    'l_opt',l_opt, ...
    'CV_error',Err, ...
    'fitError',fitError);

if strcmp(opts.basis,'legendre')
    paramInfo.legendreMap = @(eta,theta,delta) real(W*phi_basis(eta,theta,delta));
    paramInfo.legendreCoefficients = W;
    paramInfo.legendrePhi = phi_basis;
    paramInfo.legendreToMonomial = B_to_monomial;
    paramInfo.monomialCoefficientsFull = W_public;
    paramInfo.monomialPhiFull = phi;
    paramInfo.monomialExponentsFull = combined_exponents;
    paramInfo.legendreCenter = [opts.eta_center; opts.delta_center];
    paramInfo.legendreScale = [opts.eta_scale; opts.delta_scale];
    paramInfo.regressionBasis = 'legendre';
end

IMInfo = struct('chart',struct(),'parametrization',paramInfo);
end

function opts = localOptions(varargin)
opts = struct();
opts.eta_order = 3;
opts.delta_order = 2;
opts.K = 2;
opts.l_vals = 1e-4;
opts.n_folds = 0;
opts.fold_style = 'default';
opts.c1 = 0;
opts.c2 = 0;
opts.feature_limit = inf;
opts.eta_center = [];
opts.eta_scale = [];
opts.delta_center = [];
opts.delta_scale = [];
opts.eta_exp = [];
opts.d_exp = [];
opts.regression = 'columnScaledRidge';
opts.harmonic_penalty = 0;
opts.state_degree_penalty = 0;
opts.delta_degree_penalty = 0;
opts.origin_fixed = false;
opts.fixed_points = struct();
opts.amplitude_gated_harmonics = false;
opts.amplitude_gate_index = 2;
opts.amplitude_gate_scale = [];
opts.basis = 'legendre';

for ii = 1:2:numel(varargin)
    opts.(varargin{ii}) = varargin{ii+1};
end
end

function [T,Eta,Y,Theta,Delta,Weights,idx_traj] = packGeometryData(yData,etaData,paramData,~)
T = [];
Eta = [];
Y = [];
Theta = [];
Delta = [];
Weights = [];
idx_traj = cell(1,size(etaData,1));
idx_end = 0;

for ii = 1:size(etaData,1)
    t_i = etaData{ii,1}(:).';
    eta_i = etaData{ii,2};
    y_i = yData{ii,2};
    if size(eta_i,2) ~= numel(t_i) || size(y_i,2) ~= numel(t_i)
        error('Trajectory %d has inconsistent time, eta, and y dimensions.',ii);
    end

    theta_i = paramData(ii).theta(:).';
    delta_i = paramData(ii).delta;
    if isvector(delta_i)
        delta_i = delta_i(:);
    end
    if size(delta_i,2) == 1
        delta_i = repmat(delta_i,1,numel(t_i));
    end
    if numel(theta_i) ~= numel(t_i) || size(delta_i,2) ~= numel(t_i)
        error('Trajectory %d has inconsistent parameter dimensions.',ii);
    end

    if isfield(paramData,'weight') && ~isempty(paramData(ii).weight)
        weight_i = paramData(ii).weight(:).';
    elseif isfield(paramData,'traj_weight') && ~isempty(paramData(ii).traj_weight)
        weight_i = paramData(ii).traj_weight*ones(1,numel(t_i));
    else
        weight_i = ones(1,numel(t_i));
    end
    if numel(weight_i) ~= numel(t_i)
        error('Trajectory %d has inconsistent weight dimensions.',ii);
    end

    T = [T t_i]; %#ok<AGROW>
    Eta = [Eta eta_i]; %#ok<AGROW>
    Y = [Y y_i]; %#ok<AGROW>
    Theta = [Theta theta_i]; %#ok<AGROW>
    Delta = [Delta delta_i]; %#ok<AGROW>
    Weights = [Weights weight_i]; %#ok<AGROW>
    idx_traj{ii} = idx_end + (1:numel(t_i));
    idx_end = idx_end + numel(t_i);
end

if isempty(T)
    error('No samples supplied to geometry regression.');
end
end

function [center,scale] = scalingFromData(X,center,scale)
if isempty(center)
    center = 0.5*(max(X,[],2)+min(X,[],2));
else
    center = center(:);
end
if isempty(scale)
    scale = 0.5*(max(X,[],2)-min(X,[],2));
else
    scale = scale(:);
end
scale(abs(scale)<1e-12) = 1;
end

function [C,T] = geometryConstraints(opts,k,l,n_out)
C = [];
T = [];

if opts.origin_fixed
    eta0 = zeros(k,1);
    eta0_basis = stateBasisProducts(eta0,opts);
    n_eta = size(opts.eta_exp,1);
    n_delta = size(opts.d_exp,1);
    n_base = n_eta*n_delta;
    for kk = -opts.K:opts.K
        harmonic_offset = (kk+opts.K)*n_base;
        for id = 1:n_delta
            c = zeros(n_base*(2*opts.K+1),1);
            block = harmonic_offset + (id-1)*n_eta + (1:n_eta);
            c(block) = eta0_basis;
            C = [C c]; %#ok<AGROW>
            T = [T zeros(n_out,1)]; %#ok<AGROW>
        end
    end
end

fixedPoints = opts.fixed_points;
if isempty(fieldnames(fixedPoints))
    return
end
for iFix = 1:numel(fixedPoints)
    eta = fixedPoints(iFix).reducedState(:);
    y = fixedPoints(iFix).observableState(:);
    if isfield(fixedPoints,'theta')
        theta = fixedPoints(iFix).theta;
    else
        theta = 0;
    end
    if isfield(fixedPoints,'parameter')
        delta = fixedPoints(iFix).parameter(:);
    else
        delta = zeros(l,1);
    end
    c = parametricFourierFeatures(eta,theta,delta, ...
        opts.eta_center,opts.eta_scale,opts.delta_center,opts.delta_scale, ...
        opts.eta_exp,opts.d_exp,opts.K,opts.feature_limit, ...
        opts.amplitude_gated_harmonics,opts.amplitude_gate_index, ...
        opts.amplitude_gate_scale,opts.basis);
    C = [C c]; %#ok<AGROW>
    T = [T y]; %#ok<AGROW>
end
end

function [W,l_opt,Err] = columnScaledConstrainedRidge(Phi,Y,L,idx_folds,l_vals,penalty,C,T)
X = Phi.';
Yt = Y.';
L = L(:);

scale = max(abs(X),[],1).';
scale(abs(scale)<1e-12) = 1;
Xs = X ./ scale.';
XL = L .* Xs;
if isempty(C)
    Cscaled = [];
else
    Cscaled = C ./ scale;
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
                B = solveRidgeLeastSquares(Xs(train,:),Yt(train,:),L(train), ...
                    l_vals(il),penalty);
            else
                A = XL(train,:)'*Xs(train,:) + reg;
                rhs = XL(train,:)'*Yt(train,:);
                B = solveConstrainedRidge(A,rhs,Cscaled,T);
            end
            E = Yt(test,:) - Xs(test,:)*B;
            Err(il,ifold) = mean(sqrt(sum(abs(E).^2,2)));
        end
    end
    [~,best] = min(mean(Err,2));
    l_opt = l_vals(best);
end

reg = diag(l_opt*penalty(:));
if isempty(Cscaled)
    B = solveRidgeLeastSquares(Xs,Yt,L,l_opt,penalty);
else
    A = XL'*Xs + reg;
    rhs = XL'*Yt;
    B = solveConstrainedRidge(A,rhs,Cscaled,T);
end
W = (B ./ scale).';
end

function [W,l_opt,Err] = ridgeRegressionWithConstraints(Phi,Y,L,idx_folds,l_vals,penalty,C,T)
[W,l_opt,Err] = columnScaledConstrainedRidge(Phi,Y,L,idx_folds,l_vals, ...
    penalty,C,T);
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

function penalty = featurePenalty(eta_exp,d_exp,K,harmonic_weight,state_weight,delta_weight)
n_eta = size(eta_exp,1);
n_d = size(d_exp,1);
state_degree = sum(eta_exp,2);
delta_degree = sum(d_exp,2);
base_penalty = zeros(n_eta*n_d,1);
row = 1;
for id = 1:n_d
    for ip = 1:n_eta
        state_factor = (1 + state_degree(ip))^(2*state_weight);
        delta_factor = (1 + delta_degree(id))^(2*delta_weight);
        base_penalty(row) = state_factor * delta_factor;
        row = row + 1;
    end
end

penalty = zeros(n_eta*n_d*(2*K+1),1);
row = 1;
for kk = -K:K
    harmonic_factor = (1 + abs(kk))^(2*harmonic_weight);
    penalty(row:row+n_eta*n_d-1) = harmonic_factor * base_penalty;
    row = row + n_eta*n_d;
end
end

function [B_full,combined_exponents] = basisToMonomialMatrix(opts,k,l)
combined_exponents = combinedStateParameterExponents(opts.eta_exp,opts.d_exp);
n_base = size(combined_exponents,1);
if strcmp(opts.basis,'legendre')
    center = [opts.eta_center; opts.delta_center];
    scale = [opts.eta_scale; opts.delta_scale];
    B_base = scaledLegendreToMonomial(combined_exponents,center,scale);
else
    B_base = eye(n_base);
end
B_full = kron(eye(2*opts.K+1),B_base);
if k + l ~= size(combined_exponents,2)
    error('Internal exponent bookkeeping has inconsistent dimension.');
end
end

function E = combinedStateParameterExponents(eta_exp,d_exp)
E = zeros(size(eta_exp,1)*size(d_exp,1), ...
    size(eta_exp,2)+size(d_exp,2));
row = 1;
for id = 1:size(d_exp,1)
    for ie = 1:size(eta_exp,1)
        E(row,:) = [eta_exp(ie,:) d_exp(id,:)];
        row = row + 1;
    end
end
end

function B = scaledLegendreToMonomial(Expmat,center,scale)
n_basis = size(Expmat,1);
B = zeros(n_basis,n_basis);
for iBasis = 1:n_basis
    terms = legendreProductTerms(Expmat(iBasis,:),center,scale);
    for iTerm = 1:size(terms.exp,1)
        idx = find(ismember(Expmat,terms.exp(iTerm,:),'rows'),1);
        if ~isempty(idx)
            B(iBasis,idx) = B(iBasis,idx) + terms.coeff(iTerm);
        end
    end
end
end

function terms = legendreProductTerms(alpha,center,scale)
n_var = numel(alpha);
term_exp = zeros(1,n_var);
term_coeff = 1;
for iVar = 1:n_var
    p = shiftedLegendreCoefficients(alpha(iVar),center(iVar),scale(iVar));
    p = sqrt((2*alpha(iVar)+1)/2)*p;
    next_exp = [];
    next_coeff = [];
    for iTerm = 1:size(term_exp,1)
        for iDeg = 0:numel(p)-1
            if abs(p(iDeg+1)) > eps
                exp_new = term_exp(iTerm,:);
                exp_new(iVar) = exp_new(iVar) + iDeg;
                next_exp = [next_exp; exp_new]; %#ok<AGROW>
                next_coeff = [next_coeff; term_coeff(iTerm)*p(iDeg+1)]; %#ok<AGROW>
            end
        end
    end
    term_exp = next_exp;
    term_coeff = next_coeff;
end
terms = struct('exp',term_exp,'coeff',term_coeff);
end

function p = shiftedLegendreCoefficients(order,center,scale)
p_s = legendreCoefficients(order);
p = 0;
base = [-center/scale 1/scale];
for ii = 0:order
    p = localPolyAdd(p,p_s(ii+1)*localPolyPower(base,ii));
end
end

function c = legendreCoefficients(order)
if order == 0
    c = 1;
elseif order == 1
    c = [0 1];
else
    c0 = 1;
    c1 = [0 1];
    for n = 1:order-1
        c2 = ((2*n+1)*localPolyMul([0 1],c1) - n*localPad(c0,n+2))/(n+1);
        c0 = c1;
        c1 = c2;
    end
    c = c1;
end
end

function p = localPolyPower(base,powerValue)
p = 1;
for ii = 1:powerValue
    p = localPolyMul(p,base);
end
end

function c = localPolyMul(a,b)
c = conv(a,b);
end

function c = localPolyAdd(a,b)
n = max(numel(a),numel(b));
c = localPad(a,n) + localPad(b,n);
end

function a = localPad(a,n)
a = [a zeros(1,n-numel(a))];
end

function Phi = parametricFourierFeatures(Eta,theta,delta,eta_center,eta_scale, ...
    delta_center,delta_scale,eta_exp,d_exp,K,feature_limit, ...
    amplitude_gated_harmonics,amplitude_gate_index,amplitude_gate_scale,basis)
if isvector(Eta)
    Eta = Eta(:);
end
if isvector(delta)
    delta = delta(:);
end
if isscalar(theta)
    theta = theta(:).';
end
if size(delta,2) == 1 && size(Eta,2) > 1
    delta = repmat(delta,1,size(Eta,2));
end
if isscalar(theta) && size(Eta,2) > 1
    theta = repmat(theta,1,size(Eta,2));
end

n = size(Eta,2);
n_features = size(eta_exp,1)*size(d_exp,1)*(2*K+1);
Phi = zeros(n_features,n);

for j = 1:n
    EtaBasis = basisProducts(Eta(:,j),eta_exp,eta_center,eta_scale, ...
        feature_limit,basis);
    Pd = basisProducts(delta(:,j),d_exp,delta_center,delta_scale, ...
        feature_limit,basis);
    base = kron(Pd,EtaBasis);
    alpha = exp(1i*theta(j));
    gate = 1;
    if amplitude_gated_harmonics
        if size(delta,1) < amplitude_gate_index
            error('amplitude_gate_index exceeds number of parameter coordinates.');
        end
        gate = delta(amplitude_gate_index,j);
        if ~isempty(amplitude_gate_scale)
            gate = gate/amplitude_gate_scale;
        end
    end
    row = 1;
    for kk = -K:K
        block_gate = 1;
        if amplitude_gated_harmonics && kk ~= 0
            block_gate = gate;
        end
        Phi(row:row+numel(base)-1,j) = block_gate*base*alpha^kk;
        row = row + numel(base);
    end
end
end

function vals = stateBasisProducts(eta,opts)
vals = basisProducts(eta,opts.eta_exp,opts.eta_center,opts.eta_scale, ...
    opts.feature_limit,opts.basis);
end

function vals = basisProducts(x,E,center,scale,feature_limit,basis)
switch string(basis)
    case "legendre"
        x_scaled = (x-center)./scale;
        if isfinite(feature_limit)
            x_scaled = min(max(x_scaled,-feature_limit),feature_limit);
        end
        vals = evalLegendreProducts(x_scaled,E);
    case {"monomial","taylor"}
        vals = evalMonomialProducts(x,E);
    otherwise
        error('Unknown geometry basis "%s".',basis);
end
end

function vals = evalMonomialProducts(x,E)
vals = ones(size(E,1),1);
for i = 1:size(E,1)
    for j = 1:numel(x)
        vals(i) = vals(i) * x(j)^E(i,j);
    end
end
end

function vals = evalLegendreProducts(x,E)
vals = zeros(size(E,1),1);
for i = 1:size(E,1)
    v = 1;
    for j = 1:numel(x)
        deg = E(i,j);
        L = legendreValues(x(j),deg);
        v = v*sqrt((2*deg+1)/2)*L(deg+1);
    end
    vals(i) = v;
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

function E = multiIndexRange(n,min_order,max_order)
Eall = multiIndex(n,max_order);
deg = sum(Eall,2);
E = Eall(deg >= min_order & deg <= max_order,:);
end

function E = multiIndex(n,order)
if n == 0
    E = zeros(1,0);
    return
end
if n == 1
    E = (0:order)';
    return
end
E = [];
for k = 0:order
    sub = multiIndex(n-1,order-k);
    E = [E; [k*ones(size(sub,1),1), sub]]; %#ok<AGROW>
end
end

function folds = randomFolds(N,n_folds)
perm = randperm(N);
folds = cell(n_folds,1);
fold_size = floor(N/n_folds);
for ii = 1:n_folds-1
    folds{ii} = perm(1+(ii-1)*fold_size:ii*fold_size);
end
folds{n_folds} = perm(1+(n_folds-1)*fold_size:end);
end
