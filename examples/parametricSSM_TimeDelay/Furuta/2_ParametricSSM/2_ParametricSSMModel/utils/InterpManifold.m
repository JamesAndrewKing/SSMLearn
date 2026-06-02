function InterpManifold(mu_query, config, varargin)
%INTERPMANIFOLD Interpolate and reconstruct manifold at a query parameter.
%   Interpolates IMInfo.parametrization.tangentSpaceAtOrigin and
%   IMInfo.parametrization.nonlinearCoefficients from training data at
%   config.mu_values using either the default method (config.interpMethodManifold)
%   or, if 'hybrid' is enabled, methodHybrid1/2 depending on which interval
%   (mu_range1 / mu_range2) contains mu_query.
%
%   Training data are read from:
%       IMInfoRDInfoForTraining/<folderPattern(mu)>/IMInfo.mat
%
%   Methods:
%       linear, constant → use only the two nearest training points
%       spline, bulirschstoer, floaterhormann, ratint, barylag → use all
%           available training points in config.mu_values
%       constant → interpreted as nearest-left interpolation in mu
%
%   Optional name–value inputs:
%       'C'              → shape parameter for 'floaterhormann' / 'ratint'
%       'hybrid'         → logical flag to enable hybrid interpolation
%       'mu_range1'      → [mu_min mu_max] for first hybrid interval
%       'mu_range2'      → [mu_min mu_max] for second hybrid interval
%       'methodHybrid1'  → method used in mu_range1
%       'methodHybrid2'  → method used in mu_range2
%
%   Output: constructs the interpolated manifold (chart map, parametrization
%   map, polynomial order, coefficients, and exponents) and saves IMInfo into:
%       IMInfoRDInfoInterpolated/<folderPattern(mu_query)>/IMInfo.mat
%
%   Requires: Interpolation Utilities by Joe Henning
%        Joe Henning (2025). Interpolation Utilities 
%        (https://ch.mathworks.com/matlabcentral/fileexchange/36800-interpolation-utilities), 
%        MATLAB Central File Exchange. Retrieved November 19, 2025.

%---------- defaults ----------
    opts = struct( ...
        'C',              1, ...
        'hybrid',         'false', ...
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
        method = config.interpMethodManifold;
    else
       %Hybrid case: method depends on the parameter interval in which
       %mu_query falls
       if ~isempty(opts.mu_range1) && mu_query >= opts.mu_range1(1) && mu_query <= opts.mu_range1(2)
           method = opts.methodHybrid1;
       elseif ~isempty(opts.mu_range2) && mu_query >= opts.mu_range2(1) && mu_query <= opts.mu_range2(2)
           method = opts.methodHybrid2;
       else
           % outside hybrid ranges → fallback to non-hybrid case
            method = config.interpMethodManifold;
       end
        fprintf('Manifold: \n Interpolation method to be used to evaluate the model at mu_query = %.2f: %s\n', mu_query, method);
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
        'parametrization.tangentSpaceAtOrigin'
        'parametrization.nonlinearCoefficients'
    };
   IMInfo_query = [];
   % interpolate each field
   for i = 1:numel(fields)
      parts = strsplit(fields{i}, '.');
      % gather data
      dataArr = [];
      for t = mu_used
          R = load(fullfile(baseTrainDir, sprintf(config.folderPattern, t), 'IMInfo.mat'), 'IMInfo').IMInfo;
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
       IMInfo_query = setfield(IMInfo_query, parts{:}, coefficients_query);
   end

%---------- 4) Assign to IMInfo query ----------
    coeff_tanSpace = IMInfo_query.parametrization.tangentSpaceAtOrigin;
    coeff_nonLinear = IMInfo_query.parametrization.nonlinearCoefficients;
    polyOrder = R.parametrization.polynomialOrder;
 
    if polyOrder == 1 
        paramMap =  @(q)coeff_tanSpace*q; 
        phi = @(q)[];
        Emat = [];
    else
        [phi,Emat] = multivariatePolynomial(size(coeff_tanSpace,2),2,polyOrder);
        paramMap =  @(q)coeff_tanSpace*q + coeff_nonLinear*phi(q);
    end    
    paramMapOut = @(q)q;
    chartMap =  @(x)transpose(coeff_tanSpace)*x;
       
    IMInfo_query.chart.map                             = chartMap;
    IMInfo_query.chart.polynomialOrder                 = 1;
    IMInfo_query.parametrization.map                   = paramMap;
    IMInfo_query.parametrization.polynomialOrder       = polyOrder;
    IMInfo_query.parametrization.dimension             = size(coeff_tanSpace,2);
    IMInfo_query.parametrization.tangentSpaceAtOrigin  = coeff_tanSpace;
    IMInfo_query.parametrization.nonlinearCoefficients = coeff_nonLinear;
    IMInfo_query.parametrization.phi                   = phi;
    IMInfo_query.parametrization.exponents             = Emat;
    IMInfo_query.parametrization.exponents             = paramMapOut;
    
    % save IMInfo.mat
    outDir = fullfile('IMInfoRDInfoInterpolated', sprintf(config.folderPattern, mu_query));
    if ~exist(outDir,'dir'), mkdir(outDir); end
    IMInfo = IMInfo_query;
    save(fullfile(outDir, 'IMInfo.mat'), 'IMInfo');
end