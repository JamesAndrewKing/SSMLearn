function results = DTW_NMTE(yTrunc, yRec, indTest, applyShift, doPlot, thresh)
% DTW_NMTE  Aligns and compares one or more trajectories with optional
%           global shift, plotting, and FFT-deviation threshold.
%
%   results = DTW_NMTE(yTrunc, yRec, indTest)
%     analyzes each index in vector indTest with no shift, no plotting,
%     and default FFT threshold = 30%.
%
%   results = DTW_NMTE(..., applyShift, doPlot, thresh)
%     applyShift (logical): apply global shift if true (default=false)
%     doPlot   (logical): show diagnostic plots if true (default=false)
%     thresh   (scalar):   FFT-peak deviation threshold in % (default=30)
%
% INPUTS:
%   yTrunc   – cell array, each row {t, x_ref}
%   yRec     – cell array, each row {t, x_pred}
%   indTest  – vector of row indices to analyze
%
% OUTPUT:
%   results – struct array with fields:
%       .index        – test index
%       .errorPct     – error (DTW or NMTE) 
%       .shiftLag     – samples of global shift applied
%       .usedNMTE     – true if NMTE fallback was used
%       .deviations   – vector of FFT-peak deviations (unitless)
%       .maxDeviation – maximum FFT-peak deviation in %
%       .reason       – explanation of choice

    % set defaults
    if nargin<4 || isempty(applyShift), applyShift = false; end
    if nargin<5 || isempty(doPlot),    doPlot    = false; end
    if nargin<6 || isempty(thresh),    thresh    = 30;    end

    nTests = numel(indTest);
    results(nTests) = struct( ...
        'index',[], 'errorPct',[], 'shiftLag',[], ...
        'usedNMTE',[], 'deviations',[], 'maxDeviation',[], 'reason','' );

    for k = 1:nTests
        idx = indTest(k);

        % 1) sampling
        t_ref = yTrunc{idx,1}(:);
        dt    = t_ref(2)-t_ref(1);
        fs    = 1/dt;

        % 2) signals
        x1      = yTrunc{idx,2}(1,:);
        x2_orig = yRec{idx,2}(1,:);

        % 3) global shift
        if applyShift
            [x2, shiftLag] = globalShift(x1, x2_orig);
        else
            x2       = x2_orig;
            shiftLag = 0;
        end

        % 4) FFT 
        [X1, freqs] = fftSpectrum(x1, fs);
        [X2, ~]     = fftSpectrum(x2, fs);
        nPeaks      = 3;
        [useNMTE, deviations] = decideFallbackPeaks(X1, X2, freqs, nPeaks, thresh);
        maxDev = max(deviations)*100;  

        % 5) compute error 
        if ~useNMTE
            errLabel = 'DTW error';
            reason = sprintf('DTW chosen');
                %'DTW chosen: max FFT deviation = %.2f%% < %d%%.', maxDev, thresh);
                
            [errorPct, pathX, pathY, refWarped, predWarped] = ...
                computeDTW(x1, x2, t_ref, fs);
            methodLabel = sprintf('DTW error = %.2f%%', errorPct*100);
        else
            errLabel = 'NMTE error';
            offIdx = find(deviations*100 > thresh, 1);
            reason = sprintf( ...
                'NMTE fallback: peak %d deviation = %.2f%% > %d%%.', ...
                offIdx, deviations(offIdx)*100, thresh);
            errorPct     = computeNMTE(x1, x2_orig);
            pathX = []; pathY = []; refWarped = []; predWarped = [];
            methodLabel = sprintf('NMTE error = %.2f%%', errorPct*100);
        end

        % 5b) print 
        fprintf('Test %d: %s (%s = %.2f%%)\n', ...
                idx, reason, errLabel, errorPct*100);

        % 6) optional plot
        if doPlot
            plotWorkflow( ...
                idx, t_ref, x1, x2_orig, x2, ...
                pathX, pathY, refWarped, predWarped, ...
                X1, X2, freqs, ...
                shiftLag, useNMTE, methodLabel, deviations, applyShift, reason, maxDev ...
            );
        end

        % 7) store
        results(k).index        = idx;
        results(k).errorPct     = errorPct;
        results(k).shiftLag     = shiftLag;
        results(k).usedNMTE     = useNMTE;
        results(k).deviations   = deviations;
        results(k).maxDeviation = maxDev;
        results(k).reason       = reason;
        results(k).errorLabel   = errLabel;
    end
end

%% Sub‐functions

function [x2s, lag] = globalShift(x1, x2)
    [c, lags] = xcorr(x1, x2);
    [~, idx]  = max(c);
    lag       = lags(idx);
    x2s       = circshift(x2, lag);
end

function errPct = computeNMTE(x1, x2)
    err    = mean(abs(x1 - x2)) / max(abs(x1));
    errPct = err * 100;
end

function [errPct, pX, pY, rw, pw] = computeDTW(x1, x2, t, fs)
    N   = numel(t);
    dx1 = gradient(x1, 1/fs);
    dx2 = gradient(x2, 1/fs);
    C   = zeros(N);
    for i = 1:N
        for j = 1:N
            C(i,j) = ((x1(i)-x2(j))^2 + (t(i)-t(j))^2) * abs(dx1(i)-dx2(j));
        end
    end
    D       = cumulativeCost(C);
    [pX,pY] = backtrackDTW(D);
    rw      = x1(pX);
    pw      = x2(pY);
    errPct  = mean(abs(rw - pw)) / max(abs(pw));
end

function D = cumulativeCost(C)
    N = size(C,1);
    D = zeros(N);
    D(1,1) = C(1,1);
    for i = 2:N, D(i,1) = C(i,1) + D(i-1,1); end
    for j = 2:N, D(1,j) = C(1,j) + D(1,j-1); end
    for i = 2:N
        for j = 2:N
            D(i,j) = C(i,j) + min([D(i-1,j),D(i,j-1),D(i-1,j-1)]);
        end
    end
end

function [pX,pY] = backtrackDTW(D)
    N = size(D,1);
    i=N; j=N; pX=i; pY=j;
    while i>1 || j>1
        if      i==1, j=j-1;
        elseif  j==1, i=i-1;
        else
            [~, idx] = min([D(i-1,j),D(i,j-1),D(i-1,j-1)]);
            if idx==1, i=i-1;
            elseif idx==2, j=j-1;
            else, i=i-1; j=j-1; end
        end
        pX=[i; pX]; pY=[j; pY];
    end
end

function [X, freqs] = fftSpectrum(x, fs)
    N     = numel(x);
    X     = fft(x);
    freqs = (0:N-1)*(fs/N);
end

function [useFB, dev] = decideFallbackPeaks(X1, X2, freqs, nPeaks, thresh)
    N       = numel(X1);
    half    = 1:floor(N/2);
    m1      = abs(X1(half));
    m2      = abs(X2(half));
    fHalf   = freqs(half);
    [~,l1]  = findpeaks(m1,'SortStr','descend','NPeaks',nPeaks);
    [~,l2]  = findpeaks(m2,'SortStr','descend','NPeaks',nPeaks);
    pf1 = nan(nPeaks,1); pf2 = nan(nPeaks,1);
    pf1(1:numel(l1)) = fHalf(l1);
    pf2(1:numel(l2)) = fHalf(l2);
    m1  = isnan(pf1)&~isnan(pf2); pf1(m1)=pf2(m1);
    m2  = isnan(pf2)&~isnan(pf1); pf2(m2)=pf1(m2);
    valid = ~isnan(pf1)&~isnan(pf2);
    dev   = nan(nPeaks,1);
    dev(valid) = abs(pf1(valid)-pf2(valid))./abs(pf1(valid));
    useFB = any(dev*100 > thresh);
    useFB = logical(useFB);
end

function plotWorkflow(idx, t, x1, x2o, x2s, pX, pY, rw, pw, X1, X2, freqs, lag, fb, lbl, dev, sh, reason)
    Nfft = numel(X1);
    fh   = freqs(1:floor(Nfft/2));

    figure('Color','w','Position',[200 100 800 1100]);
    sgtitle(sprintf('Test %d – %s', idx, reason), ...
            'FontSize',18, 'Interpreter','none');

    % Panel 1
    ax1 = subplot(5,1,1);
    plot(t,x1,'k','LineWidth',1.4); hold on;
    plot(t,x2o,'r--','LineWidth',1.2); grid on;
    title('1) Raw Overlay','FontSize',16); xlabel('Time [s]'); ylabel('Amp');
    legend('Full','Prediction','Location','best'); set(ax1,'FontSize',12);

    % Panel 2
    ax2 = subplot(5,1,2);
    if sh
        plot(t,x1,'k','LineWidth',1.4); hold on;
        plot(t,x2s,'b--','LineWidth',1.2); grid on;
        title(sprintf('2) Global Shift (lag=%d)',lag),'FontSize',16);
        legend('Full','Shifted','Location','best');
    else
        text(0.5,0.5,'No Global Shift','FontSize',14,'HorizontalAlignment','center');
        axis off; title('2) No Global Shift','FontSize',16);
    end
    xlabel('Time [s]'); ylabel('Amp'); set(ax2,'FontSize',12);

    % Panel 3
    ax3 = subplot(5,1,3);
    if ~fb
        plot(pX,pY,'.-','LineWidth',1.2); grid on;
        xlabel('Full idx'); ylabel('Pred idx');
    else
        text(0.3,0.5,'DTW skipped','FontSize',14); axis off;
    end
    title('3) DTW Path','FontSize',16); set(ax3,'FontSize',12);

    % Panel 4
    ax4 = subplot(5,1,4);
    if ~fb
        plot(t(pX),rw,'k','LineWidth',1.4); hold on;
        plot(t(pX),pw,'r--','LineWidth',1.2); grid on;
        legend('Full warped','Pred warped','Location','best');
        xlabel('Warped Time'); ylabel('Amp');
    else
        plot(t,x1,'k','LineWidth',1.4); hold on;
        plot(t,x2o,'r--','LineWidth',1.2); grid on;
        legend('Full','Prediction','Location','best');
        xlabel('Time [s]'); ylabel('Amp');
    end
    title(['4) ' lbl],'FontSize',16); set(ax4,'FontSize',12);

    % Panel 5
    ax5 = subplot(5,1,5);
    plot(fh,abs(X1(1:numel(fh))),'k','LineWidth',1.4); hold on;
    plot(fh,abs(X2(1:numel(fh))),'r--','LineWidth',1.2); grid on;
    xlim([0 max(fh)]);
    title(sprintf('5) FFT Comparison (max dev=%.2f%%)',max(dev)*100),'FontSize',16);
    xlabel('Freq [Hz]'); ylabel('Mag');
    legend('Full spec','Pred spec','Location','best'); set(ax5,'FontSize',12);
end
