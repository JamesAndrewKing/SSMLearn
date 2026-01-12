function InterpDynamics(mu_query, config, varargin)
%INTERPDYNAMICS Interpolate and reconstruct reduced dynamics at a query parameter.
%   Interpolates RDInfo.reducedDynamics.coefficients from training data at
%   config.mu_values using either the default method (config.interpMethodDynamics)
%   or, if 'hybrid' is enabled, methodHybrid1/2 depending on which interval
%   (mu_range1 / mu_range2) contains mu_query.
%
%   Methods:
%       linear   → use two nearest points
%       constant → nearest-left point
%       others   → use all training points
%
%   Optional inputs:
%       'C', 'hybrid', 'mu_range1', 'mu_range2', 'methodHybrid1', 'methodHybrid2'
%
%   Output: saves the reconstructed RDInfo (map, coeffs, eigenvalues, etc.) into:
%       IMInfoRDInfoInterpolated/<folderPattern(mu_query)>/RDInfo.mat   
%
%   Requires: Interpolation Utilities by Joe Henning
%        Joe Henning (2025). Interpolation Utilities 
%        (https://ch.mathworks.com/matlabcentral/fileexchange/36800-interpolation-utilities), 
%        MATLAB Central File Exchange. Retrieved November 19, 2025.
%---------- defaults ----------
    opts = struct( ...
        'C',              1, ...
        'hybrid',         false, ...
        'mu_range1',      [], ...
        'mu_range2',      [], ...
        'methodHybrid1',  '', ...
        'methodHybrid2',  '' );

%---------- read name–value pairs ----------
    for k = 1:2:length(varargin)
        name  = varargin{k};
        value = varargin{k+1};

        switch lower(name)
            case {'c_floaterhormann'}
                opts.C = value;
            otherwise
                if isfield(opts, name)
                    opts.(name) = value;
                else
                    error('Unknown option: %s', name);
                end
        end
    end

    baseTrainDir = 'IMInfoRDInfoForTraining';
    mu_values    = config.mu_values(:).';
  
%---------- check ----------
    if mu_query < mu_values(1)
        error('Query parameter value outside training range (too small)')
    elseif mu_query > mu_values(end)
        error('Query parameter value outside training range (too big)')
    end 

%---------- 1) Choose the interpolation method to be used at mu_query ----------
    if strcmp(opts.hybrid, 'true')
        method = config.interpMethodDynamics;
    else
       %Hybrid case: method depends on the parameter interval in which
       %mu_query falls
       if ~isempty(opts.mu_range1) && mu_query >= opts.mu_range1(1) && mu_query <= opts.mu_range1(2)
           method = opts.methodHybrid1;
       elseif ~isempty(opts.mu_range2) && mu_query >= opts.mu_range2(1) && mu_query <= opts.mu_range2(2)
           method = opts.methodHybrid2;
       else
           % outside hybrid ranges → fallback to non-hybrid case
            method = config.interpMethodDynamics;
       end
        fprintf('Dynamics:\n Interpolation method to be used to evaluate the model at mu_query = %.2f: %s\n', mu_query, method);
    end

%---------- 2) interpolation methods ----------
    if any(strcmp(method, {'linear','constant'}))
        k  = find(mu_values <= mu_query, 1, 'last');
        if k == numel(mu_values) 
            k = k-1; 
        end
        mu_used = mu_values([k, k+1]);   % only the two points of the training parameter bin 
    elseif any(strcmp(method, {'spline','bulirschstoer','floaterhormann','ratint','barylag'}))
        mu_used = mu_values;
    else
        error('Unknown interpolation method: %s', method);
    end
    fprintf('Employed training parameter values = [%s]\n \n', num2str(mu_used,'%.2f '));
     
%---------- 3) interpolation ----------
    % list of fields to interpolate
    fields = {
        'reducedDynamics.coefficients'
    };
    RDInfo_query = [];
    % interpolate each field
    for i = 1:numel(fields)
       parts = strsplit(fields{i}, '.');
       % gather data
       dataArr = [];
       for t = mu_used
           R = load(fullfile(baseTrainDir, sprintf(config.folderPattern, t), 'RDInfo.mat'), 'RDInfo').RDInfo;
           A = getfield(R, parts{:});
           dataArr = cat(ndims(A)+1, dataArr, A);
       end
       % reshape & interp
       sz      = size(dataArr);
       flat    = reshape(dataArr, [], numel(mu_used));
       if isempty(flat)
            coefficients_query = [];
       else
           if any(strcmp(method, {'linear','spline'}))
                interpF = interp1(mu_used, flat.', mu_query, method).';
           elseif strcmp(method,'constant')
               % nearest-left interpolation
               k = find(mu_used <= mu_query, 1, 'last');
                   if isempty(k)
                       error('Constant interpolation: mu_query is smaller than all mu values.');
                   end
                interpF = flat(:, k);
            elseif strcmp(method, 'bulirschstoer')
                interpF = zeros(size(flat,1),1);
                for kk = 1:size(flat,1)
                    interpF(kk) = bulirschstoer(mu_used, flat(kk,:), mu_query).';
                end
            elseif strcmp(method, 'floaterhormann')
                interpF = zeros(size(flat,1),1);
                for kk = 1:size(flat,1)
                    interpF(kk) = floaterhormann(mu_used, flat(kk,:), mu_query, opts.C).';
                end
            elseif strcmp(method, 'ratint')
                interpF = zeros(size(flat,1),1);
                for kk = 1:size(flat,1)
                    interpF(kk) = ratint(mu_used, flat(kk,:), mu_query).';
                end
            elseif strcmp(method, 'barylag')
                interpF = zeros(size(flat,1),1);
                for kk = 1:size(flat,1)
                    interpF(kk) = barylag(mu_used, flat(kk,:), mu_query).';
                end
            end 
            coefficients_query = reshape(interpF, sz(1:end-1));
       end
        % assign back
        RDInfo_query = setfield(RDInfo_query, parts{:}, coefficients_query);
    end
    
%---------- 4) Assign to RDInfo query ----------
    coeff = RDInfo_query.reducedDynamics.coefficients;
    polyOrder = R.reducedDynamics.polynomialOrder;
    [phi,Emat] = multivariatePolynomial(size(coeff,1),1,polyOrder);
    map =  @(x)RDInfo_query.reducedDynamics.coefficients * phi(x);
    J = coeff(:,1:size(coeff,1));
    [V,~,d] = eigSorted(J);
        
    RDInfo_query.reducedDynamics        = assembleStruct(map, coeff,phi, Emat);
    RDInfo_query.eigenvaluesLinPartFlow = d;
    RDInfo_query.eigenvectorsLinPart    = V;
    RDInfo_query.dynamicsType           = R.dynamicsType;

    % save RDInfo.mat
    outDir = fullfile('IMInfoRDInfoInterpolated', sprintf(config.folderPattern, mu_query));
    if ~exist(outDir,'dir'), mkdir(outDir); end
    RDInfo = RDInfo_query;
    save(fullfile(outDir, 'RDInfo.mat'), 'RDInfo');
end
        
%%%%%%%%%%%%%%%%%%%% subfunctions %%%%%%%%%%%%%%%%%%%%

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
