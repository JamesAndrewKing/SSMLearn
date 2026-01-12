function [yTrunc, yRec] = predict(k_new, config, lag, overEmbed, indTest, sliceInt, NMTEorDTW, figOpts)
%RECONSTRUCTANDCOMPARE  Load interpolated IM/RD info, reconstruct trajectories,
%and compare to full‐system data at Ts_new
%
%   reconstructAndCompare(Ts_new, config, lag, overEmbed, indTest, sliceInt)
%
%   Inputs:
%     Ts_new     — desired sampling time (ms)
%     config     — struct with fields:
%                    .folderPattern, .Kp, .Kd, .SSMDim
%     lag        — embedding time‐stepping lag. 
%     overEmbed  — embedding over‐dimension
%     indTest    — vector of test trajectory indices
%     sliceInt   — [tStart, tEnd] for slicing trajectories

    %% 1) Locate interpolation folder (SNIPER, exact k labeinl)
    [kLabel, ~] = formatKLabel(k_new);  % preserve digits if string
    
    baseDir = fullfile('IMInfoRDInfoInterpolated', ['Ts' kLabel 'ms_FurutaTimeDelay']);

    if ~exist(baseDir, 'dir')
        error('Folder not found: %s', baseDir);
    end

    %% 2) Load and clean xData
    xDataFile = fullfile(baseDir, ['xDataKp_15.5_Kd_5.45.mat']);
    if ~isfile(xDataFile)
        error('xData file not found: %s', xDataFile);
    end

    S = load(xDataFile, 'xData');
    xData = cleanZeroRows(S.xData);   % your cleaning utility


    %% 3) Number of trajectories
    [nTraj, ~] = size(xData);
    if any(indTest > nTraj) || any(indTest < 1)
        error('indTest contains indices outside 1..%d.', nTraj);
    end

    %% 4) Load IMInfo & RDInfo
    iPath = fullfile(baseDir, 'IMInfo.mat');
    rPath = fullfile(baseDir, 'RDInfo.mat');
    
    if ~isfile(iPath)
        error('Missing IMInfo file: %s', iPath);
    end
    if ~isfile(rPath)
        error('Missing RDInfo file: %s', rPath);
    end
    
    iS     = load(iPath);   IMInfo = iS.IMInfo;
    rS     = load(rPath);   RDInfo = rS.RDInfo;

    fprintf('Using interpolated IMInfo & RDInfo at k = %.2f \n', k_new);

    %% 5) Embed & advect
    % We build delay-coordinate embeddings of dimension config.SSMDim + overEmbed,
    % stepping by "lag" samples, then slice to [tStart, tEnd], and advect.
    
    [yData, embOpts] = coordinatesEmbedding( ...
        xData,                      ... % {t, x(t)}
        config.SSMDim,              ... % intrinsic dim (e.g., 2)
        'OverEmbedding', overEmbed, ...
        'TimeStepping',  lag);
    
    % Slice to the requested time interval
    yTrunc = sliceTrajectories(yData, sliceInt);
    
    % Advect on the interpolated manifold & reduced dynamics in reduced
    % coordinates eta
    [yRec] = advectRedCoord(IMInfo, RDInfo, yTrunc);


%% 7) Compute error on test set (NMTE or average DTW)
    results = []; 
    thresh    = 10000;           % FFT-peak diff threshold to switch metric
    switch upper(NMTEorDTW)
        case 'NMTE'
            % Per-trajectory NMTE in [0,1]
            trajErrors = computeTrajectoryErrors(yRec, yTrunc, 1);
            err = mean(trajErrors(indTest));
            errorMetric = 'NMTE';
            fprintf('Error (NMTE) over test set = %.2f%%\n', err*100);
    
        case 'DTW'
            % DTW_NMTE for the test indices (results has fields: errorPct, usedNMTE)
            results = DTW_NMTE(yTrunc, yRec, indTest, false, false, thresh);
            usedNMTE = [results.usedNMTE];
    
            if any(usedNMTE)
                % Some fell back to NMTE → switch overall metric to NMTE
                fprintf(['Some trajectories fell back to NMTE (FFT dev > %d%%). ', ...
                         'Consider increasing ''thresh''.\n'], thresh);
                trajErrors = computeTrajectoryErrors(yRec, yTrunc, 1);
                err = mean(trajErrors(indTest));
                errorMetric = 'NMTE';
                fprintf('Switching to NMTE: Error (NMTE) over test set = %.2f%%\n', err*100);
            else
                % All used DTW → average DTW error over test set
                dtwErrs = [results.errorPct];
                err = mean(dtwErrs);
                errorMetric = 'DTW';
                fprintf('Error (DTW) average over test set = %.2f%%\n', err*100);
            end
    
        otherwise
            error('NMTEorDTW must be ''NMTE'' or ''DTW''');
    end


    %% 8) Figure creation & per-trajectory plots
    hFig = figure('Color','w','Units','normalized','Position',[0.1 0.1 0.8 0.6]);
    opts = cell2struct(figOpts(2:2:end), figOpts(1:2:end), 2);
    
    N = numel(indTest);
    
    singleW = 0.85 / N;
    gap     = 0.06;
    startL  = 0.18 + (0.7 - (singleW*N + gap*(N-1))) / 2;
    
    nmte_for = @(idx) computeTrajectoryErrors(yRec(idx,:), yTrunc(idx,:), 1);
    
    for ii = 1:N
        trIdx = indTest(ii);
    
        left = startL + (ii-1)*(singleW + gap);
        pos  = [left, 0.2, singleW, 0.7];
    
        ax = axes('Parent',        hFig, ...
                  'Position',      pos, ...
                  'FontName',      opts.FontName, ...
                  'FontSize',      opts.FontSizeAxes, ...
                  'LineWidth',     opts.AxesLineWidth, ...
                  'Color',         opts.BackgroundColor, ...
                  'Box',           opts.Box);
        grid(ax, opts.Grid);
        ax.TickLabelInterpreter = 'latex';
        hold(ax, 'on');
    
        t      = yRec{trIdx,1}(:);
        x_full = yTrunc{trIdx,2}(1,:);
        x_ssm  = yRec{trIdx,2}(1,:);
        
        % Plot "Full" (truncated measurement) vs "SSM" (reconstruction)
        plot(ax, t, x_full, ...
             '-', ...
             'LineWidth', opts.PlotLineWidth, ...
             'Color',     'black', ...
             'DisplayName','Full');
    
        plot(ax, t, x_ssm, ...
             '-', ...
             'LineWidth', opts.PlotLineWidth, ...
             'Color',     'red', ...
             'DisplayName','SSM');
    
        xlabel(ax, '$t\ \mathrm{[s]}$', ...
               'Interpreter', opts.Interpreter, ...
               'FontName',    opts.FontName, ...
               'FontSize',    opts.FontSizeLabels);
    
        if ii == 1
            ylabel(ax, '$\theta [deg]$', ...
                   'Interpreter', opts.Interpreter, ...
                   'FontName',    opts.FontName, ...
                   'FontSize',    opts.FontSizeLabels);
        end
    
        if ii == 1 || ii == 2
            lg = legend(ax, ...
                        'Location',    'best', ...
                        'Interpreter', opts.Interpreter, ...
                        'FontSize',    opts.FontSizeLabels, ...
                        'Orientation', 'horizontal', ...
                        'NumColumns',  2);
            lg.ItemTokenSize = [45, 45];
            lg.Box           = 'off';
        end
            
                
        %---------------- FFT (BOTTOM-RIGHT, SEMILOGY) ----------------%
        x1 = x_full(:);
        x2 = x_ssm(:);
        tt = t(:);
        
        good = isfinite(tt) & isfinite(x1) & isfinite(x2);
        tt = tt(good);
        x1 = x1(good);
        x2 = x2(good);
        
        if numel(tt) >= 8
            dt   = mean(diff(tt));
            Fs   = 1/dt;
            Nsig = numel(tt);
        
            Xf1 = fft(x1);
            Xf2 = fft(x2);
        
            P2_1 = abs(Xf1 / Nsig);
            P1_1 = P2_1(1:floor(Nsig/2)+1);
            if numel(P1_1) > 2
                P1_1(2:end-1) = 2*P1_1(2:end-1);
            end
        
            P2_2 = abs(Xf2 / Nsig);
            P1_2 = P2_2(1:floor(Nsig/2)+1);
            if numel(P1_2) > 2
                P1_2(2:end-1) = 2*P1_2(2:end-1);
            end
        
            f = Fs * (0:floor(Nsig/2)) / Nsig;
        
            pMax = max([P1_1(:); P1_2(:); 1]);
            epsY = 1e-12 * pMax;
            P1_1(P1_1 <= 0) = epsY;
            P1_2(P1_2 <= 0) = epsY;
        
            axPos = ax.Position;
            insetPos = [ ...
                axPos(1) + axPos(3)*0.04, ...
                axPos(2) + axPos(4)*0.08, ...
                axPos(3)*0.20, ...
                axPos(4)*0.1 ];
        
            axFFT = axes( ...
                'Parent', hFig, ...
                'Units','normalized', ...
                'Position', insetPos, ...
                'FontName', opts.FontName, ...
                'LineWidth', 0.8, ...
                'Color', 'none', ...
                'Box', 'on', ...
                'XColor','k', ...
                'YColor','k' );
        
            hold(axFFT,'on');
        
            axFFT.FontSize   = 20;
            axFFT.TickLength = [0.015 0.015];
            axFFT.TickDir    = 'in';
        
            axFFT.YScale = 'log';
            grid(axFFT,'on');
            axFFT.XMinorGrid = 'on';
            axFFT.YMinorGrid = 'on';
        
            axFFT.GridAlpha      = 0.20;
            axFFT.MinorGridAlpha = 0.10;
            axFFT.GridLineStyle      = '-';
            axFFT.MinorGridLineStyle = ':';
        
            axFFT.TickLabelInterpreter = 'latex';
            axFFT.YAxis.Exponent = 0;
        
            semilogy(axFFT, f, P1_1, '-', 'LineWidth', 0.9, 'Color','black');
            semilogy(axFFT, f, P1_2, '-', 'LineWidth', 0.9, 'Color','red');
        
            xlim(axFFT, [0 15]);
            ylim(axFFT, [1e-3, 1e1]);
        
            axFFT.XLabel.String      = '$f\,[\mathrm{Hz}]$';
            axFFT.XLabel.Interpreter = 'latex';
            axFFT.XLabel.FontSize    = axFFT.FontSize - 1;
            axFFT.XLabel.Units       = 'normalized';
            axFFT.XLabel.Position(2)= -0.22;   
        
            axFFT.YLabel.String = '';
        
            axFFT.Title.String      = 'FFT';
            axFFT.Title.Interpreter = 'latex';
            axFFT.Title.FontWeight  = 'normal';
            axFFT.Title.FontSize    = axFFT.FontSize - 1;
            axFFT.Title.Units       = 'normalized';
            axFFT.Title.Position(2)= 1.01;     
        
            uistack(axFFT,'top');
            set(hFig,'CurrentAxes',ax);
        end


        % ---- trajectory error label (DTW vs NMTE) ----
        if exist('results','var') && ~isempty(results) && ii <= numel(results)
            if results(ii).usedNMTE == 0
                err_pct  = results(ii).errorPct * 100;
                metric_i = 'DTW';
            else
                % Fallback for this trajectory only → compute its NMTE
                err_pct  = nmte_for(trIdx) * 100;
                metric_i = 'NMTE';
            end
        else
            % NMTE branch (or results not available) → compute NMTE
            err_pct  = nmte_for(trIdx) * 100;
            metric_i = 'NMTE';
        end

        label = [ ...
            '$\mathrm{T_s = ' , num2str(kLabel,'%.2f') , ...
            '\,ms - ' , metric_i , ...
            '\ Error = ' , num2str(err_pct,'%.2f') , '\%}$' ...
        ];

        subtitle(ax, label, ...
            'Interpreter','latex', ...
            'FontSize',   opts.FontSizeTitle, ...
            'FontWeight','normal');

        tmin   = min(yTrunc{trIdx,1}(:));
        tmax   = max(yTrunc{trIdx,1}(:));
        ytrace = yTrunc{trIdx,2}(1,:);
        margin = 0.25 * max(1e-9, abs(max(ytrace)));
        ymin   = min(ytrace) - margin;
        ymax   = max(ytrace) + margin;
    
        xlim(ax, [tmin, tmax]);
        ylim(ax, [ymin, ymax]);
    end
    outDir = fullfile(pwd, 'Predictions');
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end

    outFile = fullfile(outDir, sprintf('Ts%2f.png', k_new));
    set(hFig, 'PaperPositionMode', 'auto');
    print(hFig, outFile, '-dpng', '-r300');

end

function s = localBuildEtaLatex(coeffs, exps)
    deg = sum(exps(:,1:2), 2);
    [~, ord] = sort(deg);
    coeffs = coeffs(ord);
    exps   = exps(ord,:);

    terms = cell(numel(coeffs),1);
    for ii = 1:numel(coeffs)
        c  = coeffs(ii);
        e1 = exps(ii,1);
        e2 = exps(ii,2);

        if ii>1 && c>=0
            sign_str = '+';
        elseif c<0
            sign_str = '-';
        else
            sign_str = '';
        end

        cabs = abs(c);
        if (e1==0 && e2==0)
            c_str = num2str(cabs,'%.5g');
        else
            if abs(cabs-1) < 1e-14
                c_str = '';
            else
                c_str = num2str(cabs,'%.5g');
            end
        end

        mono = '';
        if e1>0
            mono = ['\eta_1' localSup(e1)];
        end
        if e2>0
            mono = [mono '\eta_2' localSup(e2)];
        end
        if isempty(mono)
            mono = '1';
        end
        terms{ii} = [sign_str c_str mono];
    end
    rhs = strjoin(terms, ' ');
    s = ['$\text{ } = ' rhs '$'];
end

function s = localSup(n)
    if n<=1, s=''; else, s=['^{' num2str(n) '}']; end
end

