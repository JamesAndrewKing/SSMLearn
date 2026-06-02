function [nu] = CorrelationDimensionGP(xData, slicexData, d, tau,  nTrajToUse, lmin, lmax, expandPointsLinearFit)
% - xData - cell array nTrajx2 containing trajectories 
%       (first column: time vector, second column: scalar observable) 
% - d - delay embedding space dimension
% - tau - time lag for delay embedding
% - nTrajToUse - number of first trajectories in xData to be used
% - lmin, lmax - min and max distance to fit correlation dimension
%       (default: min and max l found in xData)
% - expandPointsLinearFit - expand the estimated linear regime on right and
%       left with expandPointsLinearFit points (default = 3)
% Reference: P. Grassberger, I. Procaccia, 
%   Measuring the strangeness of strange attractors, Physica 9D [1983]
    if nargin < 5
        nTraj = size(xData,1);
    else 
        nTraj = nTrajToUse;
    end

    x = [];
    for iii = 1:nTraj
        x = [x, xData{iii,2}(1,(end-slicexData+1):end)];
    end
    
    N = numel(x);
    xi = []; %d * Mpoints_in_delay_emb_space
    for jj = 1:d
        xi(jj,:) = x(1+(jj-1)*tau:tau:(numel(x)-((d-jj)*tau)));
    end
    X = xi.';   % X rows: points in R^d
    D = squareform( pdist(X,'euclidean') );   % N x N, matrix of euclidean distance of each pair of points in R^d
    
    % Use only the upper–triangular part of D (i < j):
    %   – avoids counting each point with itself (distance = 0)
    %   – avoids counting each pair twice (if i is close to j, j is also close to i,
    %     but this pair must be counted only once)
    mask = triu(true(size(D)),1);   
    Dvals = D(mask);
    if nargin < 7
        l_min = min(D(D>0));        % min non-zero distance found
        l_max = max(D(:));          % max distance found
    else
        l_min = lmin;
        l_max = lmax;
    end

    n_l   = 40;                
    l_vals = logspace(log10(l_min), log10(l_max), n_l);  
    C_l    = zeros(size(l_vals));
    for k = 1:length(l_vals)
        ell = l_vals(k);    
        C_l(k) = sum(Dvals <= ell) / (N^2); % GP formula    
    end
    

    %%%%%%%%%%%%%%%%% Plots and linear fit %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    x = log10(l_vals(:));          % log(l)
    y = log10(C_l(:));          % log(C(l))

    % Find most linear region in log-log space (max R^2 over sliding windows)    
    minWin  = max(5, floor(numel(C_l)/4));   % minimum window length
    bestR2  = -Inf;
    bestIdx = [1 numel(C_l)];
    
    for i1 = 1:(numel(C_l) - minWin + 1)
        for i2 = (i1 + minWin - 1):numel(C_l)
            Xw = [ones(i2-i1+1,1), x(i1:i2)];
            [b_tmp, ~, ~, ~, st] = regress(y(i1:i2), Xw);
            if st(1) > bestR2           % st(1) = R^2
                bestR2  = st(1);
                bestIdx = [i1 i2];
                beta    = b_tmp;
            end
        end
    end
    
    i1 = bestIdx(1);
    i2 = bestIdx(2);
    if nargin < 8
        expandPoints = 3; % Expand region by a user-chosen number of points and refit  
    else
        expandPoints = expandPointsLinearFit;
    end
    i1 = max(1,   i1 - expandPoints);
    i2 = min(numel(C_l),   i2 + expandPoints);    
    x_fit = x(i1:i2);   y_fit = y(i1:i2);
    l_fit = l_vals(i1:i2);   C_fit = C_l(i1:i2);    
    X_fit = [ones(numel(x_fit),1), x_fit];
    [beta, bint, ~, ~, stats] = regress(y_fit, X_fit);
    
    nu    = beta(2);
    nu_CI = bint(2,:);      % [low, high]
    nu_low  = nu_CI(1);
    nu_high = nu_CI(2);
    nu_err  = 0.5 * (nu_high - nu_low);   % metà ampiezza CI

    l_dense = logspace(log10(min(l_fit)), log10(max(l_fit)), 200);
    x_dense = log10(l_dense);
    y_dense = beta(1) + beta(2)*x_dense;
    C_dense = 10.^y_dense;
    
    figure('Color','w', 'Position', [100 100 800 600]);
    loglog(l_vals,C_l,'o','MarkerFaceColor',[76 114 176]/255,'MarkerEdgeColor',[76 114 176]/255,'MarkerSize',10,'DisplayName','data'); hold on
    loglog(l_fit,C_fit,'s','MarkerFaceColor',[221 132 82]/255,'MarkerEdgeColor',[221 132 82]/255,'MarkerSize',10,'DisplayName','used for fit')
    loglog(l_dense,C_dense,'-','LineWidth',4,'DisplayName',sprintf('$C(\\ell)\\propto\\ell^{%.2f}$',nu))
    
    xlabel('$\ell$','Interpreter','latex'); ylabel('$C(\ell)$','Interpreter','latex')
    set(gca,'FontSize',25,'TickLabelInterpreter','latex')
    legend('show','Location','best','Interpreter','latex','FontSize',24)
    title('Correlation integral: best power-law fit','Interpreter','latex')
    grid on
end