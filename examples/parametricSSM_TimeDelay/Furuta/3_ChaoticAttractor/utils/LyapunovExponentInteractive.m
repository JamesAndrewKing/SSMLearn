function [LyExp, LyTime] = LyapunovExponentInteractive(yData, tEvalLyExp)
%%% LyapunovExponent
% Computes the largest Lyapunov exponent from a set of trajectories

    nTraj = size(yData,1);
    if nTraj < 2
        error('LyapunovExponent:AtLeastTwoTrajectories', ...
              'At least two trajectories are required.');
    end

    figure('Color','w', 'Position', [200 200 1200 900]);
    hold on
    box on

    idx = 1;
    % Loop over all pairs of trajectories
    for ii = 1:nTraj-1
        for jj = ii+1:nTraj
            tPoints = min( cellfun(@numel, yData(1:nTraj,1)) );
            t = yData{ii,1}(1:tPoints);
            phi = yData{ii,2}(:,1:tPoints).';     % rows: points in R^d
            phi_eps = yData{jj,2}(:,1:tPoints).'; % rows: points in R^d
            delta = sqrt( sum( (phi_eps - phi).^2, 2 ) );
            delta_ratio = delta./delta(1);
    
            semilogy(t, delta_ratio, 'Color', [0.55, 0.55, 0.55], ...
                     'LineWidth', 1, 'HandleVisibility','off');
            
            if numel(t) < 6
                continue;
            end
            x(:,idx) = t.';   
            y(:, idx) = delta_ratio;
            idx = idx+1;
        end
    end
    
    semilogy(NaN, NaN, '-', 'Color', [0.55, 0.55, 0.55], 'LineWidth', 1.5, ...
        'DisplayName', 'Separation of pairs of trajectories');

    % Ensemble average of separations
    mean_separation = mean(y,2);
    semilogy(x(:,1), mean_separation, '-', 'Color', [1.0, 0.65, 0.3], ...
             'LineWidth', 2.5, 'DisplayName', 'Ensemble average of separation');

    t_all = x(:,1);
    mask0 = t_all <= tEvalLyExp & mean_separation > 0;
    X_ensemble = t_all(mask0);
    Y_ensemble = log(mean_separation(mask0)); 

    [p_ensemble, ~] = polyfit(X_ensemble, Y_ensemble, 1);
    LyExp_ensemble  = p_ensemble(1);
    LyTime_ensemble = 1 / LyExp_ensemble;

    t_fit = linspace(0, tEvalLyExp, 300);
    log_delta_fit = polyval(p_ensemble, t_fit);
    delta_ratio_fit_ensemble = exp(log_delta_fit);

    fitLabel = sprintf('$\\exp(%.3g\\, t %+ .3g)$', p_ensemble(1), p_ensemble(2));
    
    hFit = semilogy(t_fit, delta_ratio_fit_ensemble, 'k-', ...
        'LineWidth', 2, ...
        'DisplayName', fitLabel);

    ax = gca;
    hLine = xline(tEvalLyExp, 'k--', ...
        'DisplayName', sprintf('t = %.3g', tEvalLyExp), ...
        'LineWidth', 1.5);

    hText = text(ax, 0.05, 0.9, ...
        sprintf('$\\lambda = %.3g,\\; T_L = %.3g$', ...
                LyExp_ensemble, LyTime_ensemble), ...
        'Units', 'normalized', ...
        'Interpreter', 'latex', ...
        'FontSize', 16);

    hold off;    
    
    xlabel('t', 'Interpreter', 'latex', 'FontSize', 16);
    ylabel('$\delta(t) / \delta(0)$', 'Interpreter', 'latex', 'FontSize', 16);
    title('Growth of Separation $\delta(t)$', 'Interpreter', 'latex', 'FontSize', 18);
    
    lgd = legend('show', 'Location', 'best');
    lgd.Interpreter = 'latex';
    
    ax.FontSize = 20;
    ax.LineWidth = 1.5;
    ax.Box = 'on';
    ax.XMinorGrid = 'on';
    ax.YMinorGrid = 'on';
    ax.GridAlpha = 0.3;
    ax.MinorGridAlpha = 0.15;
    ax.YScale = 'log';
    ax.YTick = 10.^(0:5);
    ax.YTickLabel = arrayfun(@(v) sprintf('$10^{%d}$', v), 0:5, ...
                         'UniformOutput', false);
    ax.TickLabelInterpreter = 'latex';

    LyExp  = LyExp_ensemble;
    LyTime = LyTime_ensemble;

    fig = gcf;
    fig.UserData = struct( ...
        't',               t_all, ...
        'mean_separation', mean_separation, ...
        'hFit',            hFit, ...
        'hLine',           hLine, ...
        'hText',           hText);

    set(hLine, 'ButtonDownFcn', @(src, evt) startDragLine(src));

end

% ====================== Helper callbacks ====================== %

function startDragLine(hLine, ~)
    fig = ancestor(hLine,'figure');
    fig.WindowButtonMotionFcn = @(fig, evt) dragLine(fig);
    fig.WindowButtonUpFcn     = @(fig, evt) stopDragLine(fig);
end

function dragLine(fig, ~)
    ax = fig.CurrentAxes;
    cp = ax.CurrentPoint;
    newT = cp(1,1); 

    xl = ax.XLim;
    newT = max(min(newT, xl(2)), xl(1));

    updateLyapunov(ax, newT);
end

function stopDragLine(fig, ~)
    fig.WindowButtonMotionFcn = '';
    fig.WindowButtonUpFcn     = '';
end

function updateLyapunov(ax, newT)
    fig = ax.Parent;
    ud  = fig.UserData;

    t   = ud.t(:);
    ms  = ud.mean_separation(:);

    mask = t <= newT & ms > 0;
    if sum(mask) < 2
        return; % not enough points
    end

    X = t(mask);
    Y = log(ms(mask));

    [p, ~] = polyfit(X, Y, 1);
    LyExp  = p(1);
    LyTime = 1 / LyExp;

    t_fit = linspace(min(X), max(X), 300);
    log_delta_fit = polyval(p, t_fit);
    delta_fit = exp(log_delta_fit);

    set(ud.hFit, 'XData', t_fit, 'YData', delta_fit);

    set(ud.hLine, 'Value', newT, 'Label', sprintf('t = %.3g', newT));

    set(ud.hText, 'String', ...
        sprintf('$\\lambda = %.3g,\\; T_L = %.3g$', LyExp, LyTime));

    ud.LyExp  = LyExp;
    ud.LyTime = LyTime;
    fig.UserData = ud;

    fprintf('Updated LyExp = %.6g, LyTime = %.6g (tEvalLyExp = %.6g)\n', ...
            LyExp, LyTime, newT);
end
