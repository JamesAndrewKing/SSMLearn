function BoA(interpStruct, mu_values, config, ICrho1rho2)
% Corrected_rho1rho2PhasePortraitsOverTs
%   Plot phase portraits rho1-rho2 for the 4D SSM-reduced normal form dynamics 
%   in polar coordinates as a function of the delay parameter T_s, 
%   and compute eigenvalues at the 2-torus.
%
%   INPUTS:
%       interpStruct(1:2) : struct array with polynomial data (coeffs/exponents)
%       mu_values            : vector of sampling times T_s (ms)
%       config.interpMethodDynamicsPP : 'linear' or 'spline'
%       ICrho1rho2: IC within and outside basin of attraction (BoA) of the
%           stable 2-torus

    outDir  = fullfile(pwd, 'BoA');
    if ~exist(outDir,'dir'), mkdir(outDir); end
    pdfFile = fullfile(outDir, 'BoA.pdf');

    uniTol      = 6.5e-2;
    TsThreshold = [31.45, 31.72];
    tspan                 = 4000;
    tspanManifoldsStable  = 5000;
    tspanManifoldsUnstable= 5000;
    xLim = [0 0.6];
    yLim = [0 0.3];

    if ~isfield(config, 'axesPosition')
        config.axesPosition = [];
    end

    for k = 1:numel(mu_values)
        Ts = mu_values(k);

        % --- Vector field and Jacobian -----------------------------------
        c1 = getCoefficients(interpStruct(1), Ts, config);
        c2 = getCoefficients(interpStruct(2), Ts, config);
        e1 = interpStruct(1).exponents;
        e2 = interpStruct(2).exponents;

        odefun = @(t,R)[ ...
            c1' * ( R(1).^e1(:,1) .* R(2).^e1(:,2) ); ...
            c2' * ( R(1).^e2(:,1) .* R(2).^e2(:,2) )  ...
        ];

        Jac = @(R) localJac(R, c1, c2, e1, e2);

        % --- Base phase portrait (no manifolds) --------------------------
        plotpp(odefun, uniTol, ...
               'tspan',tspan, 'xLim',xLim, 'yLim',yLim, ...
               'xPlotNum',10,'yPlotNum',10,'arrowSize',12, ...
               'arrowDensity',2,'quiverDensity',60, ...
               'plotArrows',true,'plotEPs',true, ...
               'plotQuiver',true,'plotNonSaddleTrajectory',false);
        hold on;

        EPs = evalin('base','EP_set'); % nFP x 2

        % --- Classify equilibria -----------------------------------------
        tol = 1e-8;
        isZero1 = abs(EPs(:,1)) < tol;
        isZero2 = abs(EPs(:,2)) < tol;

        EP_origin       = EPs( isZero1 &  isZero2, :);
        EPs_limCycle    = EPs( xor(isZero1, isZero2), :);          % one and only one zero coord
        EPs_bothNonZero = EPs(~isZero1 & ~isZero2, :);
        EP_x            = EPs( isZero2 & ~isZero1, :);             % on x-axis only

        % --- Torus FP (both coordinates nonzero) -------------------------
        hThisTorus = [];
        torusLabel = '';

        if ~isempty(EPs_bothNonZero)
            ep_nz  = EPs_bothNonZero(1,:);
            J_nz   = Jac(ep_nz.');
            lambda = eig(J_nz);

            if all(real(lambda) < 0)
                torusLabel = 'Stable 2-torus';
                torusColor = [0.50, 1.00, 0.50];  % light green
            else
                torusLabel = 'Unstable 2-torus';
                torusColor = [0.50, 0.00, 0.00];  % dark red
            end

            hThisTorus = plot(ep_nz(1), ep_nz(2), 'o', ...
                'MarkerFaceColor',torusColor, ...
                'MarkerEdgeColor','k', ...
                'MarkerSize',22,'LineWidth',1.5);
        end

        % --- Stable/unstable manifolds of EPs_limCycle -------------------
        delta     = 1e-4*(xLim(2)-xLim(1));
        opts      = odeset('RelTol',1e-12,'AbsTol',1e-12);
        tolOrigin = 0.5e-2;

        hStable   = [];
        hUnstable = [];

        for iEP = 1:size(EPs_limCycle,1)
            ep = EPs_limCycle(iEP,:);
            J  = Jac(ep.');
            [V,D] = eig(J);

            for jV = 1:2
                lambda = D(jV,jV);
                if abs(imag(lambda)) > 1e-10   % skip complex eigenvalues
                    continue;
                end
                v = real(V(:,jV)).';

                for sgn = [+1, -1]
                    x0 = ep + sgn*delta*v;

                    % Unstable manifold: integrate forward
                    if real(lambda) > 0
                        [tU, Xu] = ode15s(odefun, [0 tspanManifoldsUnstable], x0, opts);
                        r2 = Xu(:,1).^2 + Xu(:,2).^2;
                        idx = find(r2 < tolOrigin^2, 1);
                        if ~isempty(idx)
                            Xu = Xu(1:idx, :);
                            tU = tU(1:idx); 
                        end
                        h = plot(Xu(:,1), Xu(:,2), '-', 'Color',[0 0 1], 'LineWidth',2);
                        if isempty(hUnstable), hUnstable = h; end
                    end

                    % Stable manifold: integrate backward
                    if real(lambda) < 0
                        [tS, Xs] = ode15s(odefun, [0 -tspanManifoldsStable], x0, opts);
                        r2 = Xs(:,1).^2 + Xs(:,2).^2;
                        idx = find(r2 < tolOrigin^2, 1);
                        if ~isempty(idx)
                            Xs = Xs(1:idx, :);
                            tS = tS(1:idx);
                        end
                        h = plot(Xs(:,1), Xs(:,2), '-', 'Color',[1 0 0], 'LineWidth',2);
                        if isempty(hStable), hStable = h; end
                    end
                end
            end
        end

        % --- Plot EPs with different markers -----------------------------
        if ~isempty(EP_origin)
            hOrigin = plot(EP_origin(:,1), EP_origin(:,2), 'o', ...
                'MarkerFaceColor',[0.70, 0.40, 0.70], ...
                'MarkerEdgeColor','k', ...
                'MarkerSize',22,'LineWidth',1.5);
            uistack(hOrigin, 'top');
        else
            hOrigin = [];
        end

        if ~isempty(EPs_limCycle)
            hLimCycle = plot(EPs_limCycle(:,1), EPs_limCycle(:,2), 'o', ...
                'MarkerFaceColor',[1.00, 0.50, 0.00], ...
                'MarkerEdgeColor','k', ...
                'MarkerSize',22,'LineWidth',1.5);
            uistack(hLimCycle, 'top');
        else
            hLimCycle = [];
        end

        % --------------------- “Unstable 3-torus” ------------------------
        hUnstable3 = [];
        if Ts > TsThreshold(1) && Ts < TsThreshold(2)
            if isempty(EP_x)
                disp('No FP found on x-axis → skip Unstable 3-torus');
            else
                tspanManifoldsStable = 5000;   % 
                ep_x = EP_x(1,:);

                J_x        = Jac(ep_x.');
                [Vx, Dx]   = eig(J_x);
                realParts  = real(diag(Dx));
                [~, idxSt] = min(realParts);
                v_stable   = real(Vx(:, idxSt)).';

                x0_x = ep_x + delta * v_stable;
                [tSx, Xsx] = ode15s(odefun, [0 -tspanManifoldsStable], x0_x, opts);

                tMin          = min(tSx);
                desiredWindow = 200;
                cutoff        = tMin + desiredWindow;
                idx_window    = find(tSx <= cutoff);

                if isempty(idx_window)
                    X_to_plot = Xsx;
                else
                    X_to_plot = Xsx(idx_window, :);
                end

                hUnstable3 = plot(X_to_plot(:,1), X_to_plot(:,2), '-', ...
                                  'Color',[0.0, 0.50, 0.0], 'LineWidth',3.5);
            end
        end
        hold off

        % --------------------- plot IC1, IC2, IC3  -----------------------
        if ~isempty(ICrho1rho2)
            hold on;
            % IC1: blue
            hIC1 = plot(ICrho1rho2(1,1), ICrho1rho2(1,2), 'o', ...
                        'MarkerSize',22, ...
                        'MarkerFaceColor',[0.0000, 0.4470, 0.7410], ...
                        'MarkerEdgeColor','k', 'LineWidth',1.5);
            % IC2: orange
            hIC2 = plot(ICrho1rho2(2,1), ICrho1rho2(2,2), 'o', ...
                        'MarkerSize',22, ...
                        'MarkerFaceColor',[0.40, 0.22, 0.08], ...
                        'MarkerEdgeColor','k', 'LineWidth',1.5);
            % IC3: dark yellow
            hIC3 = plot(ICrho1rho2(3,1), ICrho1rho2(3,2), 'o', ...
                        'MarkerSize',22, ...
                        'MarkerFaceColor',[0.9290, 0.6940, 0.1250], ...
                        'MarkerEdgeColor','k', 'LineWidth',1.5);
            uistack([hIC1, hIC2, hIC3], 'top');
            hold off;
        else
            hIC1 = []; hIC2 = []; hIC3 = [];
        end

        % ------------------------- Figure -----------------------
        set(gcf,'Color','w','Units','normalized','Position',[0.05,0.05,0.8,0.8]);
        axesPosition = [0.1 0.13 0.8 0.8];
        if ~isempty(axesPosition)
            set(gca, 'Position', axesPosition);
        end

        set(gca,'FontSize',35,'Box','on','GridLineStyle',':','TickLabelInterpreter','latex');
        xlabel('$$\rho_1$$','Interpreter','latex');
        ylabel('$$\rho_2$$','Interpreter','latex');
        title(sprintf('$$T_s = %.2f\\,\\mathrm{ms}$$', Ts),'Interpreter','latex');

        % ---------------------------- Legend ------------------------------
        legendHandles = [];
        legendLabels  = {};

        addEntry = @(h,label) ...
            ( ~isempty(h) && (legendHandles(end+1) == h) ); 

        if ~isempty(hOrigin)
            legendHandles(end+1) = hOrigin;
            legendLabels{end+1}  = 'Saddle fixed point';
        end
        if ~isempty(hLimCycle)
            legendHandles(end+1) = hLimCycle;
            legendLabels{end+1}  = 'Saddle-type limit cycle';
        end
        if ~isempty(hThisTorus)
            legendHandles(end+1) = hThisTorus;
            legendLabels{end+1}  = torusLabel;
        end
        if ~isempty(hUnstable3)
            legendHandles(end+1) = hUnstable3;
            legendLabels{end+1}  = 'Unstable 3-torus';
        end
        if ~isempty(hStable)
            legendHandles(end+1) = hStable;
            legendLabels{end+1}  = 'Stable manifolds';
        end
        if ~isempty(hUnstable)
            legendHandles(end+1) = hUnstable;
            legendLabels{end+1}  = 'Unstable manifolds';
        end
        if ~isempty(hIC1)
            legendHandles(end+1) = hIC1;
            legendLabels{end+1} = 'IC1';
        end
        if ~isempty(hIC2)
            legendHandles(end+1) = hIC2;
            legendLabels{end+1} = 'IC2';
        end
        if ~isempty(hIC3)
            legendHandles(end+1) = hIC3;
            legendLabels{end+1} = 'IC3';
        end

        if ~isempty(legendHandles)
            legend(legendHandles, legendLabels, ...
                   'Interpreter','latex','Location','northeast');
        end

        % ---------------------------- Save --------------------------------
        if k == 1
            exportgraphics(gcf,pdfFile,'ContentType','vector','Resolution',300);
        else
            exportgraphics(gcf,pdfFile,'Append',true,'ContentType','vector','Resolution',300);
        end
    end
end
