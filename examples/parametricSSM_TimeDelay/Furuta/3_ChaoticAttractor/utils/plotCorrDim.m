function plotCorrDim(xData, slicexData, dSweep, tau, nTraj_nu)
    nu = zeros(size(dSweep));
    parfor k = 1:numel(dSweep)
        nu(k) = CorrelationDimensionGP(xData, slicexData, dSweep(k), tau, nTraj_nu);
    end
    
    figure('Color','w'); 
    hold on;
    set(gca,'TickLabelInterpreter','latex','FontSize',15);
    
    plot(dSweep, nu, '-o', 'Color','k','LineWidth', 1.2);
    
    idx     = numel(nu)-4:numel(nu);              % last 5 points
    [p,S]   = polyfit(dSweep(idx), nu(idx), 0);   % constant fit
    m       = p(1);                               % fitted value
    C       = inv(S.R)*inv(S.R')*(S.normr^2/S.df);
    sigma_m = sqrt(C(1,1));
    IC      = tinv(0.975, S.df)*sigma_m;          % 95% CI
    
    fill([dSweep(1) dSweep(end) dSweep(end) dSweep(1)], ...
         [m-IC m-IC m+IC m+IC], [0.8 0.8 1], ...
         'EdgeColor','none','FaceAlpha',0.3);
    
    plot(dSweep([1 end]), [m m], 'b--', 'LineWidth', 2);
    plot(dSweep(idx), nu(idx), 'bo', 'MarkerFaceColor', 'b');
    
    legend({'Data', '95\% CI', ...
           sprintf('$\\nu = %.3f \\pm %.3f$', m, IC), ...
           'Fit points'}, ...
           'Interpreter', 'latex', 'Location', 'best', 'FontSize',15);    
    
    xlabel('$d$', 'Interpreter','latex', 'FontSize',15);
    ylabel('$\nu$', 'Interpreter','latex', 'FontSize',15);
    title('Correlation dimension $\nu$ vs delay embedding space dimension $d$', ...
          'Interpreter','latex','FontSize',15);
    
    grid on; 
    hold off;
end
