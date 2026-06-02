function plotICandFFT(xData, xDataIC, yDataDE)
% plotICandFFT
% Plots the first 3 initial-condition trajectories (IC1–IC3)
% and the FFT of IC2 over the first Tfft seconds.
%
% INPUTS:
%   xData     : cell array {n,2} with trajectories {t, x(t)}
%   xDataIC   : truncated IC trajectories for FFT extraction
%   Tfft      : number of seconds for the FFT window

    outDir = fullfile(pwd, 'BoA');
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end

    % Colors
    colors = [0.0000 0.4470 0.7410;   % IC1 blue
              [0.40, 0.22, 0.08];   % IC2 brown
              0.9290 0.6940 0.1250];  % IC3 dark yellow

    for ii = 1:3
        t = xData{ii,1}(:);
        x = xData{ii,2}(1,:);

        hFig = figure('Color','w', ...
                      'Units','normalized', ...
                      'Position',[0.15 0.15 0.55 0.40]);

        plot(t, x, 'LineWidth', 1.4, 'Color', colors(ii,:));
        hold on;

        grid on;
        set(gca, 'TickLabelInterpreter','latex', ...
                 'FontSize', 25, ...
                 'LineWidth', 0.9);

        xlabel('$\mathrm{Time\ [s]}$', 'Interpreter','latex', 'FontSize', 25);
        ylabel('$\theta\ [\mathrm{deg}]$', 'Interpreter','latex', 'FontSize', 25);

        title(sprintf('IC%d', ii), 'Interpreter','latex', 'FontSize', 25);

        xlim([0, t(end)*1.1]);

        box on;

        saveas(hFig, fullfile(outDir, sprintf('IC%d.png', ii)));
    end


    t2 = xDataIC{2,1}(:);
    x2 = xDataIC{2,2}(:);
    Tfft = 600;
    idx = t2 <= Tfft;
    t2_cut = t2(idx);
    x2_cut = x2(idx);

    dt = mean(diff(t2_cut));
    Fs = 1/dt;
    L  = numel(t2_cut);

    Y  = fft(x2_cut);
    P2 = abs(Y/L);
    P1 = P2(1:floor(L/2)+1);
    P1(2:end-1) = 2*P1(2:end-1);

    f = Fs * (0:floor(L/2)) / L;

    hFigFFT = figure('Color','w', ...
                     'Units','normalized', ...
                     'Position',[0.15 0.15 0.55 0.40]);

    semilogy(f, P1, 'LineWidth', 1.5, 'Color', colors(2,:));
    hold on;

    grid on;
    set(gca, 'TickLabelInterpreter','latex', ...
             'FontSize', 25, ...
             'LineWidth', 1);

    xlabel('$f\ (\mathrm{Hz})$', 'Interpreter','latex', 'FontSize', 25);
    ylabel('Amplitude spectrum', 'Interpreter','latex', 'FontSize', 25);

    title(sprintf('IC2 (first %.0f s)', Tfft), ...
          'Interpreter','latex', 'FontSize', 25);

    xlim([0 30]);

    legend(sprintf('IC2 (first %.0f s)', Tfft), ...
           'Interpreter','latex', ...
           'FontSize', 20, ...
           'Location','northeast');

    saveas(hFigFFT, fullfile(outDir, sprintf('IC2_FFT_%ds.png', Tfft)));

    k1 = 60;        % first delay index
    k2 = 120;        % second delay index
    
    dt = yDataDE{1,1}(2) - yDataDE{1,1}(1);   % time step from first column
    
    hFigY = figure('Color','w', ...
                   'Units','normalized', ...
                   'Position',[0.15 0.15 0.55 0.40]);
    
    hold on;
    grid on;
    box on;

    Xall = []; Yall = []; Zall = [];
    
    for ii = 1:3
        y = yDataDE{ii,2};   
    
        Xall = [Xall, y(1,:)];
        Yall = [Yall, y(k1,:)];
        Zall = [Zall, y(k2,:)];
    end
    
    tStart = [294588, 21452, 19100];  
    hTraj = gobjects(3,1);
    
    for ii = 1:3
        y = yDataDE{ii,2};
    
        hTraj(ii) = plot3(y(1,tStart(ii):end), ...
                          y(k1,tStart(ii):end), ...
                          y(k2,tStart(ii):end), ...
                          'LineWidth', 1.4, ...
                          'Color', colors(ii,:));
    end
    
    a = 14;
    xlim([-a a]);
    ylim([-a a]);
    zlim([-18 a]);
    
    set(gca, 'TickLabelInterpreter','latex', ...
             'FontSize', 25, ...
             'LineWidth', 0.9);

     xlabel('$y(t)$', 'Interpreter','latex', 'FontSize', 25); 
    ylabel(sprintf('$y(t+%.2f\\,\\mathrm{s})$', k1*dt), 'Interpreter','latex', 'FontSize', 25); 
    zlabel(sprintf('$y(t+%.2f\\,\\mathrm{s})$', k2*dt), 'Interpreter','latex', 'FontSize', 25);

    legend(hTraj, ...
           {sprintf('IC1 (last $%.f\\,\\mathrm{s}$)', ...
                    (size(yDataDE{1,2},2)-tStart(1)+1)*dt), ...
            sprintf('IC2 (last $%.f\\,\\mathrm{s}$)', ...
                    (size(yDataDE{2,2},2)-tStart(2)+1)*dt), ...
            sprintf('IC3 (last $%.f\\,\\mathrm{s}$)', ...
                    (size(yDataDE{3,2},2)-tStart(3)+1)*dt)}, ...
           'Interpreter','latex', ...
           'FontSize', 20, ...
           'Location','northeast');

    
    view([200 50]);
    saveas(hFigY, fullfile(outDir, 'yIC_3D.png'));

end