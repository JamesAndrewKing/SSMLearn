function [RDInfo,R,iT,N,T] = IMDynamicsFlow(etaData,varargin)
% [RDInfo,R,iT,N,T] = IMDynamicsFlow(etaData)
% Identification of the reduced dynamics in k coordinates, i.e. the vector
% field
%
%                        \dot{x} = R(x)
%
% via a weighted ridge regression. R(x) = W_r * phi(x) where phi is a
% k-variate polynomial from order 1 to order M. Cross-validation can be
% performed on random folds or on the trajectories for the map R.
% Upon request, the dynamics is returned via a coordinate change, i.e.
%
%                         R = D_T o N o iT
%
% where iT, T and N depend on the selected style.
% If the style is selected as modal, then the coordinate change is a linear
% map that transforms the linear part of R into the diagonal matrix of its
% eigenvalues.
% If the style is selected as normalform, then the functions seeks from
% data the maps iT, N and T such that the dynamics N is in normal form.
% This option is only available for purely oscillatory dynamics (i.e., the
% eigenvalues of the linear part of R are complex conjugated only). The
% normal form is detected by evaluating the small denominators that would
% occur when computing analytically the normal form from the knowledge of
% the vector field. These small denominators occur when the real part of
% the eigenvalues of the linear part of R is small or resonant. With small,
% it is intended to be smaller then a user-defined tolerance. To overcome
% tolerance-based issues, the user can enforce the detection of the
% coefficients of the normal form of the equivalent center manifold reduced
% order model, specifying eventual resonances among the frequencies.
%
% OUTPUTS
% R  - vector field in the given coordinates of etaData
% iT - transformation from the coordinates of etaData to normal form ones
% N  - vector field in the normal form coordinates
% T  - transformation from the normal form coordinates to those of etaData
% Maps_info - struct containing the information of all these mapings
%
% INPUTS
% etaData - cell array of dimension (N_traj,2) where the first column
%          contains time instances (1 x mi each) and the second column the
%          trajectories (k x mi each). Sampling time is assumed to be
%          constant
%  varargin = polynomial order of R
%     or
%  varargin = options list: 'field1', value1, 'field2', value2, ... . The
%             options fields and their default values are:
%     'c1' - error coefficient for slow manifolds weighting
%           (1+c1*exp(-c2*t)).^(-1), default 0
%     'c2' - error coefficient for slow manifolds weighting
%           (1+c1*exp(-c2*t)).^(-1), default 0
% 'l_vals' - regularizer values for the ridge regression, default 0
% 'n_folds'- number of folds for the cross validation, default 0
% 'fold_style' - either 'default' or 'traj'. The former set random folds
%               while the latter exclude 1 trajectory at time for the cross
%               validation
% 'style' - none, modal or normalform
% 'nf_style' - 'center_mfld' or 'actual eigs'
% 'tol_nf' - parameter for the tolerance in the detection of small
%            denominators in the normal form. The tolreance is set to be
%            tol_nf*max(abs(real(eig(R)))). Default value for tol_nf is 10
% 'frequencies_norm' - expected frequencies ratios (e.g. [1 2]) for 1:2
%                      resonance with the first and the second frequencies,
%                      by the default the code uses the actual values of
%                      of the frequencies. This option only works with the
%                      normal form style center manifold
% 'IC_nf' - initial condition for the optimization in the normal form.
%           0 (default): zero initial condition;
%           1: initial estimate based on the coefficients of R
%           2: normally distributed with the variance of case 1
% 'rescale' - rescale for the modal coordinates.
%           0: no rescale
%           1 (default): the maximum amplitude is 0.5 (ratios kept)
%           2: the maximum amplitude of all coordinates is 0.5
% 'fig_disp_nf' - display of the normal form.
%               0:  command line only
%               r:  LaTex-style figure with r terms per row (default 1).
%                   Command line also appears if the LaTex string is too
%                   long
%               -r: both command line and LaTex-style figure with r terms
%                   per row.
% 'fig_disp_nfp' - display of the normal form in a figure.
%               0: polar normal form display (default)
%               1: complex normal form display
%               2: both polar and complex normal form display
%              -1: no displays
% 'Display' - default 'iter'
% 'OptimalityTolerance' - default 1e-4 times the number of datapoints
% 'MaxIter' - default 1e3
% 'MaxFunctionEvaluations' - default 1e4
% 'SpecifyObjectiveGradient' - default true
%     the last five options are for Matlab function fminunc.
%     For more information, check out its documentation.

if rem(length(varargin),2) > 0 && length(varargin) > 1
    error('Error on input arguments. Missing or extra arguments.')
end

% Reshape of trajectories into matrices
t = []; % time values
X = []; % coordinates at time k
dXdt = []; % time derivatives at time k
ind_traj = cell(1,size(etaData,1)); idx_end = 0;
for ii = 1:size(etaData,1)
    t_in = etaData{ii,1}; X_in = etaData{ii,2};
    [dXidt,Xi,ti] = finiteTimeDifference(X_in,t_in,3);
    t = [t ti]; X = [X Xi]; dXdt = [dXdt dXidt];
    ind_traj{ii} = idx_end+[1:length(ti)]; idx_end = length(t);
end
options = IMdynamics_options(nargin,varargin,ind_traj,size(X,2));
% Phase space dimension & Error Weghting
k = size(Xi,1); L2 = (1+options.c1*exp(-options.c2*t)).^(-2);
options = setfield(options,'L2',L2);

% Construct phi and ridge regression
[phi,Expmat] = multivariatePolynomial(k,1,options.R_PolyOrd);

if strcmp(options.regression_type, 'rational')
    if options.fig_disp_nfp ~= -1
        disp('Estimation of the reduced dynamics using rational regression... ')
    end

    % Step 1: Linear unconstrained optimization
    if options.fig_disp_nfp ~= -1
        disp('Step 1: Linear unconstrained optimization...');
    end
    % Transpose dXdt for each component
    for i = 1:size(dXdt,1)
        coeffs_unconstrained{i} = rational_approximant(X, dXdt(i,:), options.R_PolyOrd, options.R_DenomOrd, ...
            'loss_type', 'linear', 'constrained', false);

        coeffs_constrained{i} = rational_approximant(X, dXdt(i,:), options.R_PolyOrd, options.R_DenomOrd, ...
            'loss_type', 'linear', 'constrained', true, ...
            'init_coeffs', coeffs_unconstrained{i}, ...
            'delta', options.R_DenomDelta);

        coeffs_rational{i} = rational_approximant(X, dXdt(i,:), options.R_PolyOrd, options.R_DenomOrd, ...
            'loss_type', 'nonlinear', 'constrained', true, ...
            'init_coeffs', coeffs_constrained{i}, ...
            'delta', options.R_DenomDelta);
    end

    % Get dimensions and unpack coefficients
    [XX_p, XX_q] = generate_features(X, options.R_PolyOrd, options.R_DenomOrd, true);
    num_unknowns_num = size(XX_p,2);
    num_unknowns_den = size(XX_q,2);
    
    % Handle each component separately
    for i = 1:size(dXdt,1)
        [a_rational{i}, b_rational{i}] = unpack_coeffs(coeffs_rational{i}, ...
            num_unknowns_num, num_unknowns_den, 1);
    end

    % Create function handle that handles all components
    R = @(x) reshape(cell2mat(cellfun(@(a,b) evaluate_rational_model(x(:), a, b, ...
        options.R_PolyOrd, options.R_DenomOrd), ...
        a_rational, b_rational, 'UniformOutput', false)), [], 1);

    % Store info
    rational_coeffs = struct('numerator', {a_rational}, 'denominator', {b_rational});
    R_info = assembleStruct(R, rational_coeffs, phi, Expmat);

    % Identity maps for coordinate changes
    T = @(x) x; iT=@(y) y; N =@(y) R(y);
    T_info = assembleStruct(@(x) x,eye(k),@(x) x,eye(k));
    iT_info = T_info; N_info = R_info;
    d = []; V = [];

    if options.fig_disp_nfp ~= -1
        fprintf('\b Done. \n')
    end

elseif strcmp(options.regression_type, 'polynomial')

    if isempty(options.R_coeff) == 1
        if options.fig_disp_nfp ~= -1
            disp('Estimation of the reduced dynamics... ')
        end
        [W_r,l_opt,Err] = ridgeRegression(phi(X),dXdt,options.L2,...
            options.idx_folds,options.l_vals);
    else
        W_r_known = options.R_coeff;
        nCoefs = size(W_r_known,2);
        if nCoefs == size(Expmat,1)
            W_r =  options.R_coeff; l_opt = 0; Err = 0;
        else
            Xtransformed = phi(X);
            Xreg = Xtransformed(nCoefs+1:end,:);
            Yreg = dXdt-W_r_known*Xtransformed(1:nCoefs,:);
            [W_r_unknown,l_opt,Err] = ridgeRegression(Xreg,Yreg,...
                options.L2,options.idx_folds,options.l_vals);
            W_r = [W_r_known W_r_unknown];
        end
    end
    R = @(x) W_r*phi(x);
    R_info = assembleStruct(@(x) W_r*phi(x),W_r,phi,Expmat,l_opt,Err);
    options.l = l_opt;
    if options.fig_disp_nfp ~= -1
        fprintf('\b Done. \n')
    end
    [V,D,d] = eigSorted(W_r(:,1:k));
    % Find the change of coordinates desired
    switch options.style
        
        case 'modal'
            % Linear transformation
            iT = @(x) V\x; T = @(y) V*y;
            T_info = assembleStruct(T,V,@(x) x,eye(k));
            iT_info = assembleStruct(iT,inv(V),@(y) y,eye(k));
            % Nonlinear modal dynamics coefficients
            V_M = multivariatePolynomialLinTransf(V,k,options.R_PolyOrd);
            W_n = V\W_r*V_M; N = @(y) V\(W_r*phi(V*y));
            N_info = assembleStruct(N,W_n,phi,Expmat);
            iT_info.lintransf = inv(V); T_info.lintransf = V;
        case 'normalform'
            if options.fig_disp_nfp ~= -1
                disp('Estimation of the reduced dynamics in normal form...')
            end
            n_real_eig = sum(imag(d)==0);
            if n_real_eig>0
                disp('Normal form not available. Returning modal style.')
                % Linear transformation
                iT = @(x) V\x; T = @(y) V*y;
                T_info = assembleStruct(T,V,@(x) x,eye(k));
                iT_info = assembleStruct(iT,inv(V),@(y) y,eye(k));
                % Nonlinear modal dynamics coefficients
                V_M = multivariatePolynomialLinTransf(V,k,options.R_PolyOrd);
                W_n = V\W_r*V_M; N = @(y) V\(W_r*phi(V*y));
                N_info = assembleStruct(N,W_n,phi,Expmat);
            else
                if options.rescale == 1
                    v_rescale = max(abs(V\X),[],2);
                    V = 2*V*diag(max(v_rescale(1:k/2))*ones(1,k));
                end
                if options.rescale == 2
                    v_rescale = max(abs(V\X),[],2);
                    V = 2*V*diag(v_rescale);
                end
                switch options.N_PolyOrd
                    case 1
                        phi_lin = @(x) x(1);
                        eye_red = eye(k); k_red = k/2; eye_red = eye_red(1:k_red,:); 
                        dr = diag(D); 
                        iT_info = struct('Map',@(x) V\x,'coeff',inv(V),'phi',phi_lin,'Exponents',...
                            eye_red);
                        T_info = struct('Map',@(z) real(V*z),'coeff',V,'phi',phi_lin,'Exponents',...
                            eye_red);
                        N_info = struct('Map',@(z) transformationComplexConj(D(1:k_red,:)*z),'coeff',dr(1:k_red),'phi',phi_lin,'Exponents',...
                            eye_red);
                        Maps = struct('iT',iT_info,'N',N_info,'T',T_info,'V',V);
                    case 2
                        error('Normal form available with order > 2.')
                    otherwise
                        % Initialize the normal form
                        Maps_info_opt=initialize_nf_flow(V,D,d,W_r,etaData,options);
                        % Get normal form mappings T, N and T^{-1}
                        Maps = dynamicsCoordChangeNF(Maps_info_opt,options);
                end
                % Final output
                T_info_opt = Maps.T; N_info_opt = Maps.N;
                iT_info_opt = Maps.iT;
                iT  = iT_info_opt.Map; N = N_info_opt.Map; T = T_info_opt.Map;
                iT_info = assembleStruct(iT,iT_info_opt.coeff,...
                    iT_info_opt.phi,iT_info_opt.Exponents);
                N_info = assembleStruct(N,N_info_opt.coeff,N_info_opt.phi,...
                    N_info_opt.Exponents);
                T_info = assembleStruct(T,T_info_opt.coeff,T_info_opt.phi,...
                    T_info_opt.Exponents);
                iT_info.lintransf = inv(V); T_info.lintransf = V;
                
                % Display the obtained normal form
                flag_long = 0;
                if abs(options.fig_disp_nf)>0
                    [str_eqn,str_eqn_plot,flag_long] = dispNormalFormFigure(N_info_opt.coeff,...
                        N_info_opt.Exponents,abs(options.fig_disp_nf));
                    N_info.LaTeXComplex = str_eqn;
                    if length(str_eqn_plot)>1 && options.fig_disp_nfp ~= 0
                        figure;
                        h = plot(0,0);
                        set(gcf,'color','w');
                        str_above = ['Using the notation $\bar{\,}$ for the complex conjugates, the identified normal form is'];
                        annotation('textbox','FontSize',18,'Interpreter','latex','FaceAlpha','1','EdgeColor','w','Position',[0.01 0.1 0.99 0.9], 'String',str_above);
                        annotation('textbox','FontSize',18,'Interpreter','latex','FaceAlpha','1','EdgeColor','w','Position',[0.02 0.12 0.98 0.76],'String',['$' str_eqn_plot '$']);
                        delete(h);
                        set(gca,'Visible','off')
                    end
                end
                if options.fig_disp_nfp > 0
                    if (options.fig_disp_nf<=0) || (flag_long==1)
                        fprintf('\n')
                        disp(['The data-driven normal form dynamics reads:'])
                        fprintf('\n')
                        table_nf = dispNormalForm(N_info_opt.coeff,...
                            N_info_opt.Exponents);
                        disp(table_nf)
                        if k == 2
                            disp(['Notation: z is a complex number; z` is the ' ...
                                'complex conjugated of z; z^k is the k-th power of z.'])
                        else
                            disp(['Notation: each z_j is a complex number; z`_j is the '...
                                'complex conjugated of z_j; z^k_j is the k-th power of z_j.'])
                        end
                    end
                end
                % Polar Normal Form
                if abs(options.fig_disp_nfp) ~= 1
                    N_info = polarNormalForm(N_info,1);
                else
                    N_info = polarNormalForm(N_info,0);
                end
            end
            
        otherwise
            T = @(x) x; iT=@(y) y; N =@(y) R(y);
            T_info = assembleStruct(@(x) x,eye(k),@(x) x,eye(k));
            iT_info = T_info; N_info = R_info;
    end
else
    disp('Regression type must be polynomial or rational.')
end
RDInfo = struct('reducedDynamics',R_info,'inverseTransformation',...
    iT_info,'conjugateDynamics',N_info,'transformation',T_info,...
    'conjugacyStyle',options.style,'dynamicsType','flow',...
    'eigenvaluesLinPartFlow',d,'eigenvectorsLinPart',V);
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

%- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

function [Maps_info_opt]=initialize_nf_flow(V,D,d,W_r,etaData,options)
% Preparation function for the estimate of the normal form maps. Based on
% the optimization properties, the functions seeks the coefficients of the
% normal form dynamics and sets to zero those coefficients for the
% transformation T^{-1}. Their indexes are stored in the output struct.
% The error at time instant k for the successive optimization is
%
% Err_k = dYdt-D*Y+W_it_nl*Dphi_it(Y)*dYdt-W_n*phi_n(Y+W_it_nl*phi_it(Y))
%
% and this function also precomputes the difference dYdt-D*Y and the
% transformations Dphi_it(Y)*dYdt and phi_it(Y) for a more efficient
% optimization. The overall process consider complex numbers, and the
% conjugated are ignored.

% Modal transformation
k = size(W_r,1); ndof = k/2;
% Transformation of coordinates & nonlinear maps
[phi_it,Expmat_it] = multivariatePolynomial(k,2,options.iT_PolyOrd);
[~,Expmat_n] = multivariatePolynomial(k,2,options.N_PolyOrd);
% Reshape of trajectories into matrices
Y = []; % modal coordinates at time k
dYdt = []; % time derivatives at time k
Phi_iT_Y = []; % modal transformation at time k
DPhi_iT_dYdt = []; % time derivatives at time k
for ii = 1:size(etaData,1)
    t_in = etaData{ii,1}; Y_in = V\(etaData{ii,2});
    Phi_iT_Y_in = phi_it(Y_in);
    [dYDphi_i_dt,YDphi_i,~] = finiteTimeDifference([Y_in; Phi_iT_Y_in],...
        t_in,3);
    Y = [Y YDphi_i(1:k,:)]; dYdt = [dYdt dYDphi_i_dt(1:k,:)];
    Phi_iT_Y = [Phi_iT_Y YDphi_i(k+1:end,:)];
    DPhi_iT_dYdt = [DPhi_iT_dYdt dYDphi_i_dt(k+1:end,:)];
end
Y_red = Y(1:ndof,:); dYdt_red = dYdt(1:ndof,:);
d_red = diag(D); d_red=d_red(1:ndof);
V_M = multivariatePolynomialLinTransf(V,k,options.R_PolyOrd);
W_modal = V\W_r*V_M;
% Get the terms of the normal form
if strcmp(options.nf_style,'center_mfld')
    tol_nf = 1e-8;
    if isempty(options.frequencies_norm) == 1
        d_nf = 1i*imag(d);
    else
        d_nf = transpose([+1i*options.frequencies_norm ...
            -1i*options.frequencies_norm]*imag(d(1)));
    end
else
    d_nf = d; tol_nf = options.tol_nf*max(abs(real(d_nf)));
end

lidx_n = find(abs(repmat(d_nf,1,size(Expmat_n,1))-...
    repmat(transpose(Expmat_n*d_nf),k,1))<tol_nf);
if options.R_PolyOrd>options.N_PolyOrd
    W_n_0 = W_modal(:,k+[1:size(Expmat_n,1)]);
else
    W_n_0 = [W_modal(:,k+1:end) zeros(k,size(Expmat_n,1)+k-size(W_r,2))];
end
if options.R_PolyOrd>options.iT_PolyOrd
    W_it_0 = -W_modal(:,k+[1:size(Expmat_it,1)]);
else
    W_it_0 =-[W_modal(:,k+1:end) zeros(k,size(Expmat_it,1)+k-size(W_r,2))];
end
W_it_0=W_it_0./(repmat(d,1,size(Expmat_it,1))-...
    repmat(transpose(Expmat_it*d),k,1));
lidx_elim_it = lidx_n;
if options.iT_PolyOrd<options.N_PolyOrd
    lidx_elim_it(lidx_elim_it>numel(W_it_0)) = [];
end
lidx_it  = transpose(1:numel(W_it_0));
lidx_it(lidx_elim_it)  = [];
% Set the indexes for the coefficients of T^{-1} and N
[idx_it(:,1),idx_it(:,2)] = ind2sub(size(W_it_0),lidx_it);
idx_it(idx_it(:,1)>ndof,:) = []; % Eliminate cc rows
W_it_0_up = W_it_0(1:ndof,:);
lidx_it_up = sub2ind(size(W_it_0_up),idx_it(:,1),idx_it(:,2));
% Eliminate uncessary exponents for N
[idx_n(:,1),  idx_n(:,2)] = ind2sub(size(W_n_0),lidx_n);
idx_n(idx_n(:,1)>ndof,:) = []; % Eliminate cc rows
W_n_0_up = W_n_0(1:ndof,:);
lidx_n_up = sub2ind(size(W_n_0_up),idx_n(:,1),idx_n(:,2));
W_n_0_up(setdiff(1:numel(W_n_0_up),lidx_n_up)) = 0;
IDX_expnts = ones(size(W_n_0_up));
IDX_expnts(setdiff(1:numel(W_n_0_up),lidx_n_up)) = 0;
idx_expnts = find(sum(IDX_expnts,1));
[phi_n,Expmat_n,D_phi_n_info] = multivariatePolynomialSelection(k,2,...
    options.N_PolyOrd,idx_expnts);
W_n_0_up = W_n_0_up(:,idx_expnts); IDX_expnts = IDX_expnts(:,idx_expnts);
lidx_n_up = find(IDX_expnts);
[idx_n(:,1),  idx_n(:,2)] = ind2sub(size(W_n_0_up),lidx_n_up);
% Initial condition for the optimization
if ndof == 1
    IC_opt_complex =transpose([W_it_0_up(lidx_it_up) W_n_0_up(lidx_n_up)]);
else
    IC_opt_complex = [W_it_0_up(lidx_it_up); W_n_0_up(lidx_n_up)];
end
IC_opt = [real(IC_opt_complex); imag(IC_opt_complex)];
% Maps info
N_info_opt  = struct('phi',phi_n,'Exponents',Expmat_n,'idx',idx_n,...
    'lidx',lidx_n_up);
iT_info_opt = struct('phi',phi_it,'Exponents',Expmat_it,'idx',idx_it,...
    'lidx',lidx_it_up);
D_phi_n = cell2struct(D_phi_n_info,{'Derivative','Indexes'},2);

% Final Output
Y_1_DY_red = dYdt_red - d_red.*Y_red;
Maps_info_opt = struct('IC_opt',IC_opt,'V',V,'d_r',d_red,'Yk_r',Y_red,...
    'Yk_1_r',dYdt_red,'Yk_1_DYk_r',Y_1_DY_red,...
    'Phi_iT_Yk',Phi_iT_Y,'Phi_iT_Yk_1',DPhi_iT_dYdt,...
    'iT',iT_info_opt,'N',N_info_opt,...
    'D_phi_n_info',D_phi_n);
end

%- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

% Default options

function options = IMdynamics_options(nargin_o,varargin_o,idx_traj,Ndata)
options = struct('Style','default','R_PolyOrd', 1,'iT_PolyOrd',1,...
    'N_PolyOrd',1,'T_PolyOrd',1,'c1',0,'c2',0,...
    'L2',[],'n_folds',0,'l_vals',0,'idx_folds',[],'fold_style',[],...
    'style','none','nf_style','center_mfld','frequencies_norm',[],...
    'tol_nf',1e1,'IC_nf',1,'R_coeff',[],'rescale',1,'fig_disp_nf',1,...
    'fig_disp_nfp',0,'Display','iter',...
    'OptimalityTolerance',10^(-8-floor(log10(Ndata))),...
    'MaxIter',1e3,...
    'MaxFunctionEvaluations',1e4,...
    'SpecifyObjectiveGradient',true,...
    'regression_type', 'polynomial',...  % 'polynomial' or 'rational'
    'R_DenomOrd', 0,...                 % Order of denominator for rational fit
    'R_DenomDelta', 0.1);              % Minimum denominator value
% Default case
if nargin_o == 2; options.R_PolyOrd = varargin_o{:};
    options.N_PolyOrd = varargin_o{:}; end
% Custom options
if nargin_o > 2
    for ii = 1:length(varargin_o)/2
        options = setfield(options,varargin_o{2*ii-1},...
            varargin_o{2*ii});
    end
    % Some default options for polynomial degree
    if strcmp(options.style,'normalform')==1 && ...
            options.iT_PolyOrd*options.N_PolyOrd*options.T_PolyOrd == 1
        options.N_PolyOrd = options.R_PolyOrd;
    end
    if strcmp(options.style,'normalform')==1 && options.N_PolyOrd > 1
        if options.T_PolyOrd*options.T_PolyOrd == 1
            options.T_PolyOrd = options.N_PolyOrd;
            options.iT_PolyOrd = options.N_PolyOrd;
        else
            PolyOrdM = max([options.T_PolyOrd options.iT_PolyOrd]);
            options.T_PolyOrd = PolyOrdM;
            options.iT_PolyOrd = PolyOrdM;
        end
    end
    % Fold indexes
    if options.n_folds > 1
        if strcmp(options.fold_style,'traj') == 1
            options = setfield(options,'n_folds',length(idx_traj));
            idx_folds = idx_traj;
        else
            idx_folds = cell(options.n_folds,1);
            ind_perm = randperm(Ndata);
            fold_size = floor(Ndata/options.n_folds);
            for ii = 1:options.n_folds-1
                idx_folds{ii} = ind_perm(1+(ii-1)*fold_size:ii*fold_size);
            end
            ii = ii+1;
            idx_folds{ii} = ind_perm(1+(ii-1)*fold_size:length(ind_perm));
        end
        options = setfield(options,'idx_folds',idx_folds);
    end
end
end

function [XX_p, XX_q] = generate_features(X, order_num, order_denom, bias)
    % Generate polynomial features for numerator and denominator
    %
    % Parameters:
    % X : array, size (n_features, n_samples)
    % order_num : integer, numerator polynomial order
    % order_denom : integer, denominator polynomial order
    % bias : logical, include bias term (default true)
    
    if nargin < 4
        bias = true;
    end
    
    % Generate features separately for numerator and denominator
    if order_num > 0
        XX_p = generate_polynomial_features(X', order_num, false);
    else
        XX_p = [];
    end
    
    if order_denom > 0
        XX_q = generate_polynomial_features(X', order_denom, bias);
    else
        XX_q = [];
    end
end

function b = polynomial_approximant(X, y, order, bias)
    % Fit a polynomial approximant to the data using least squares.
    %
    % Parameters:
    % X : array, size (n_features, n_samples)
    %     Input data
    % y : array, size (n_samples,)
    %     Target values
    % order : integer
    %     Order of the polynomial
    % bias : logical (optional)
    %     If true, include bias term. Default is true
    %
    % Returns:
    % b : array, size (n_features,)
    %     Coefficients of polynomial approximant
    
    if nargin < 4
        bias = true;
    end
    
    % Use generate_polynomial_features directly
    XX_p = generate_polynomial_features(X', order, false);
    b = pinv(XX_p) * y(:);  % Ensure y is a column vector
end

function err = linear_error_function_scalar(y, coeffs, XX_p, XX_q)
    % Compute the linearized error function for a scalar output.
    %
    % Parameters:
    % y : array, size (n_samples, n_outputs)
    %     Target values
    % coeffs : array
    %     Coefficients vector [denominator; numerator]
    % XX_p : array, size (n_samples, num_unknowns_numerator)
    %     Polynomial features for numerator
    % XX_q : array, size (n_samples, num_unknowns_denominator)
    %     Polynomial features for denominator
    %
    % Returns:
    % err : scalar
    %     Error value |y*(b*x^n) - a*x^n|^2
    
    num_unknowns_numerator = size(XX_p, 2);
    num_unknowns_denominator = size(XX_q, 2);
    
    b = coeffs(1:num_unknowns_denominator);
    A = reshape(coeffs(num_unknowns_denominator+1:end), num_unknowns_numerator, []);
    error = XX_p * A - reshape(XX_q * b, [], 1) .* y;
    err = norm(error(:))^2;
end

function err = nonlinear_error_function_scalar(y, coeffs, XX_p, XX_q)
    % Compute the full, nonlinear error function for a scalar output.
    %
    % Parameters:
    % y : array, size (n_samples, n_outputs)
    %     Target values
    % coeffs : array
    %     Coefficients vector [denominator; numerator]
    % XX_p : array, size (n_samples, num_unknowns_numerator)
    %     Polynomial features for numerator
    % XX_q : array, size (n_samples, num_unknowns_denominator)
    %     Polynomial features for denominator
    %
    % Returns:
    % err : scalar
    %     Error value |y - (a*x^n)/(b*x^n)|^2
    
    num_unknowns_numerator = size(XX_p, 2);
    num_unknowns_denominator = size(XX_q, 2);
    
    % Extract coefficients
    b = coeffs(1:num_unknowns_denominator);
    A = reshape(coeffs(num_unknowns_denominator+1:end), num_unknowns_numerator, []);
    error = (XX_p * A) ./ reshape(XX_q * b, [], 1) - y;
    err = norm(error(:))^2;
end

function [A, b] = unpack_coeffs(coeffs, num_unknowns_numerator, num_unknowns_denominator, n_features)
    % Unpack coefficients vector into numerator and denominator matrices
    %
    % Parameters:
    % coeffs : array, size (num_unknowns_denominator + num_unknowns_numerator, 1)
    %     Combined coefficient vector [denominator; numerator]
    % num_unknowns_numerator : integer
    %     Number of unknowns in numerator polynomial
    % num_unknowns_denominator : integer
    %     Number of unknowns in denominator polynomial
    % n_features : integer
    %     Number of output features
    
    % Extract denominator coefficients
    b = coeffs(1:num_unknowns_denominator);
    
    % Extract and reshape numerator coefficients
    num_coeffs = coeffs(num_unknowns_denominator+1:end);
    A = reshape(num_coeffs, [], n_features);  % Let MATLAB calculate first dimension
end

function coeffs = rational_approximant(X, y, order_num, order_denom, varargin)
    % Computes a rational approximant using least squares minimization.
    %
    % Parameters:
    % X : array, size (n_features, n_samples)
    %     Input data
    % y : array, size (n_samples,)
    %     Target values
    % order_num : integer
    %     Order of numerator polynomial
    % order_denom : integer
    %     Order of denominator polynomial
    %
    % Optional Parameters (Name-Value Pairs):
    % 'loss_type' : string, 'linear' (default) or 'nonlinear'
    % 'init_coeffs' : initial coefficient vector
    % 'constrained' : logical, default true
    % 'delta' : positive scalar, default 0.5
    %
    % Returns:
    % coeffs : vector containing [denominator; numerator] coefficients
    
    % Parse inputs
    p = inputParser;
    addParameter(p, 'loss_type', 'linear');
    addParameter(p, 'init_coeffs', []);
    addParameter(p, 'constrained', true);
    addParameter(p, 'delta', 0.5);
    parse(p, varargin{:});
    
    % Generate features with correct orders
    [XX_p, XX_q] = generate_features(X, order_num, order_denom, true);
    num_unknowns_num = size(XX_p,2);
    num_unknowns_den = size(XX_q,2);
    num_features = size(X,1);  % Changed: use input features dimension
    
    % Initialize coefficients matching Python version exactly
    if isempty(p.Results.init_coeffs)
        [XX_poly, ~] = generate_features(X, order_num, 0, false);
        a_poly = pinv(XX_poly) * y(:);  % Changed: ensure y is column vector
        b0 = zeros(num_unknowns_den,1);  % Changed: initialize with ones
        % b0 = b0 / norm(b0);  % Changed: normalize like Python
        init_coeffs = [b0; a_poly(:)];
    else
        init_coeffs = p.Results.init_coeffs;
    end
    
    % Set up objective function with correct dimensions
    if strcmp(p.Results.loss_type, 'nonlinear')
        obj_fun = @(x) nonlinear_error_function_scalar(y(:), x, XX_p, XX_q);  % Changed: ensure y is column
    else
        obj_fun = @(x) linear_error_function_scalar(y(:), x, XX_p, XX_q);  % Changed: ensure y is column
    end
    
    % Set up optimization
    options = optimoptions('fmincon', ...
        'Display', 'iter', ...
        'MaxIterations', 1000, ...
        'Algorithm', 'sqp',  ...
        'OptimalityTolerance', 1e-6);
    
    if p.Results.constrained
        % Match Python's constraint formulation
        num_constr_points = size(XX_q,1);
        Aineq = -XX_q;  % Negative because MATLAB uses ≤ while Python uses ≥
        bineq = -p.Results.delta * ones(size(XX_q,1),1);
        
        % Add zeros for numerator coefficients to match dimensions
        Aineq = [Aineq, zeros(num_constr_points, num_unknowns_num)];
        
        coeffs = fmincon(obj_fun, init_coeffs, Aineq, bineq, [], [], [], [], [], options);
    else
        coeffs = fminunc(obj_fun, init_coeffs, options);
    end
end

function y = evaluate_rational_model(X, a, b, order_num, order_denom)
    % Evaluate the rational model at the given input data.
    %
    % Parameters:
    % X : array, size (n_features, n_samples)
    %     Input data
    % a : array, size (num_unknowns_numerator, n_features)
    %     Coefficients of numerator polynomial
    % b : array, size (num_unknowns_denominator, 1)
    %     Coefficients of denominator polynomial
    % order_num : integer
    %     Order of numerator polynomial
    % order_denom : integer
    %     Order of denominator polynomial
    %
    % Returns:
    % y : array, size (n_samples, n_features)
    %     Evaluated rational model at input data
    
    [XX_p, XX_q] = generate_features(X, order_num, order_denom, true);
    y = (XX_p * a) ./ reshape(XX_q * b, [], 1);  % Match Python reshaping
end

function XX = generate_polynomial_features(X, order, include_bias)
    % Generate polynomial features up to specified order.
    % Matches scikit-learn's PolynomialFeatures behavior
    %
    % Parameters:
    % X : array, size (n_samples, n_features)
    %     Input samples
    % order : integer
    %     Maximum polynomial order
    % include_bias : logical
    %     Whether to include a bias column (all ones)
    %
    % Returns:
    % XX : array, size (n_samples, n_features_out)
    %     Transformed input with polynomial features
    
    [n_samples, n_features] = size(X);
    
    % Get all combinations of features up to given order
    combinations = {};
    for deg = 0:order
        combs = nchoosek(repmat(1:n_features, 1, deg), deg);
        combinations{deg + 1} = unique(combs, 'rows');
    end
    
    % If not including bias, remove the zero-degree term
    if ~include_bias
        combinations = combinations(2:end);
    end
    
    % Calculate features
    feature_list = [];
    for i = 1:length(combinations)
        combs = combinations{i};
        for j = 1:size(combs, 1)
            % Count occurrences of each feature
            degrees = histcounts(combs(j,:), 0.5:n_features+0.5);
            % Calculate feature
            feat = ones(n_samples, 1);
            for k = 1:n_features
                if degrees(k) > 0
                    feat = feat .* X(:,k).^degrees(k);
                end
            end
            feature_list = [feature_list, feat];
        end
    end
    
    XX = feature_list;
end