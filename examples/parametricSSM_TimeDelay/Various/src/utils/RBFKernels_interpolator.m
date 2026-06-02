function RBF_object = RBFKernels_interpolator(X, y, kernel, params, lambda)
% Interpolatore RBF con kernel selezionabile (stessa metodologia con pdist2).
% X: (d x n)   y: (m x n) oppure (n x m) -> usiamo y'
% kernel: stringa o function handle @(D,params)->K, dove D sono distanze euclidee
% params: struct con iperparametri (es. sigma, c, ell)
% lambda: regolarizzazione (default 0)

    if nargin < 3 || isempty(kernel), kernel = 'gaussian'; end
    if nargin < 4 || isempty(params), params = struct(); end
    if nargin < 5 || isempty(lambda), lambda = 0; end

    % y come (n x m)
    if size(y,1) ~= size(X,2), y = y.'; end

    % Distanze euclidee (metodologia originale)
    D = pdist2(X', X');  % (n x n)

    % Kernel
    phi = getKernel(kernel, params);          % funzione K = phi(D, params)
    K = phi(D, params);

    % Regolarizzazione
    if lambda > 0
        K = K + lambda*eye(size(K,1), 'like', K);
    end

    % Coefficienti
    coeffs = K \ y;

    % Oggetto
    RBF_object.coeffs  = coeffs;
    RBF_object.centers = X;
    RBF_object.kernel  = kernel;
    RBF_object.params  = params;
    RBF_object.kernel_querry = @(x_query) phi(pdist2(x_query', RBF_object.centers'), RBF_object.params);
    RBF_object.evaluate = @(x_query) transpose(RBF_object.kernel_querry(x_query) * RBF_object.coeffs);
end

% ---------------- helpers ----------------
function phi = getKernel(kernel, params)
    if isa(kernel,'function_handle')
        phi = @(D,p) kernel(D,p); return;
    end
    % default params
    if ~isfield(params,'sigma'), params.sigma = []; end
    if ~isfield(params,'c'),     params.c     = 1;  end
    if ~isfield(params,'ell'),   params.ell   = []; end
    if ~isfield(params,'nu'),    params.nu    = []; end

    switch lower(kernel)
        case {'gauss','gaussian'}
            % K = exp(-(D.^2)/(2*sigma^2))
            phi = @(D,p) exp(- (D.^2) / (2*autoSigma(D,p)^2));

        case {'mq','multiquadric'}
            % K = sqrt(D.^2 + c^2)
            phi = @(D,p) sqrt(D.^2 + p.c^2);

        case {'imq','inverse_mq','inverse-mq','inverse_multiquadric'}
            % K = 1 ./ sqrt(D.^2 + c^2)
            phi = @(D,p) 1 ./ sqrt(D.^2 + p.c^2);

        case {'tps','thin-plate','thin_plate_spline'}
            % K = r^2 log r, con K(0)=0
            phi = @(D,p) tpsKernel(D);

        case {'linear','r'}
            % K = r
            phi = @(D,p) D;

        case {'cubic'}
            % K = r^3
            phi = @(D,p) D.^3;

        case {'matern12','matern-1/2','exponential'}
            % K = exp(-r/ell)
            phi = @(D,p) exp(- D ./ autoEll(D,p));

        case {'matern32','matern-3/2'}
            % K = (1 + sqrt(3) r/ell) exp(-sqrt(3) r/ell)
            phi = @(D,p) maternKernel(D, autoEll(D,p), 3/2);

        case {'matern52','matern-5/2'}
            % K = (1 + sqrt(5) r/ell + 5 r^2/(3 ell^2)) exp(-sqrt(5) r/ell)
            phi = @(D,p) maternKernel(D, autoEll(D,p), 5/2);

        otherwise
            error('Kernel "%s" non riconosciuto.', kernel);
    end
end

function s = autoSigma(D,p)
    if ~isempty(p.sigma), s = p.sigma; return; end
    v = D(D>0);
    if isempty(v), s = 1; else, s = median(v); end
end

function e = autoEll(D,p)
    if ~isempty(p.ell), e = p.ell; return; end
    v = D(D>0);
    if isempty(v), e = 1; else, e = median(v); end
end

function K = tpsKernel(D)
    K = D.^2;
    m = D>0;
    K(m) = K(m) .* log(D(m));
    K(~m) = 0;
end

function K = maternKernel(D, ell, nu)
    r = D ./ ell;
    switch nu
        case 3/2
            K = (1 + sqrt(3)*r) .* exp(-sqrt(3)*r);
        case 5/2
            K = (1 + sqrt(5)*r + 5*(r.^2)/3) .* exp(-sqrt(5)*r);
        otherwise % fallback a exp(-r) se nu=1/2
            K = exp(-r);
    end
end
