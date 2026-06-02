% SSM-based model reduction through oblique projection of a 2-dof
% nonlinear oscillator chain.

clear all
close all

%%

n = 2;
k1 = 1;
k2 = 3 + 0.325;
c1 = 0.05;
alpha = 0.5;
factor = @(beta_alpha) 1/2 * beta_alpha;
c2_fun = @(beta_alpha) c1 * factor(beta_alpha);
beta_alpha_red = 0.4;
c2 = c2_fun(beta_alpha_red);
[F,A,M,C,K,F_nl] = SP(k1,k2,c1,c2,alpha,0,0);

T_tilde = [1 0; -1 1];
M_t = M/T_tilde;
C_t = C/T_tilde;
K_t = K/T_tilde;

T = [1 0 0 0; -1 1 0 0; 0 0 1 0; 0 0 -1 1];
A_t = T*A/T;

F_t = @(t,eta) A_t * eta + T*F_nl(t,T\eta);
[V_t,D_t,lambda_t] = eigSorted(A_t);


tSpan_decay = linspace(0,400,1e4);
[t_t, x_t, t_t_off, x_t_off, x0_ssm_t] = plot_SSMTool(F_t,tSpan_decay,V_t);

kmean = 1;       
[amp_ssm,freq_ssm,damp_ssm,time_ssm] = PFFk(t_t,x_t(1,:),kmean);
[amp_off,freq_off,damp_off,time_off] = PFFk(t_t_off,x_t_off(1,:),kmean);

lim = 1e-3;
[freq_spline_ssm, amp_spline_ssm] = bc_spline(freq_ssm, amp_ssm, lim);
[freq_spline_off, amp_spline_off] = bc_spline(freq_off, amp_off, lim);

f = figure;
f.Position = [1028,132,455,734];
hold on 
grid on 
plot(freq_spline_ssm, amp_spline_ssm,'color',[0.07,0.62,1.00],'LineWidth',4,'DisplayName','Trajectory on primary SSM')
plot(freq_spline_off, amp_spline_off,'-k','LineWidth',4,'DisplayName','Trajectory on fractional SSM')

xlim([0.98 1.08])
xlabel('Frequency [rad/s]','Interpreter','Latex')
ylabel('$|x_1|$','Interpreter','Latex')
set(gca,'FontSize',30,'TickLabelinterpreter','latex')
h = get(gca,'Children');
set(gca,'Children',[h(2) h(1)], ...
    'OuterPosition',[0.017728808895572,0.021298756734844,0.930686646616658,0.907894174771129], ...
    'InnerPosition',[0.213186813186813,0.136557610241821,0.654945054945055,0.725840209921667], ...
    'Position',[0.213186813186813,0.136557610241821,0.654945054945055,0.725840209921667]);
legend([h(2) h(1)],'Location','NW','Interpreter','latex','Position',[0.075229572070824,0.889168935736186,0.850496061555631,0.09441417049647])

xData = cell(1,2);
xData{1,1} = t_t_off;
xData{1,2} = x_t_off;

xOutput = @(x) x(1,:);
endTime = xData{1,1}(end);

automatic = false;
if automatic
    indplot = 1;
    [startTime, indStartTime, SSMDim] = SSM_startTime(xData(1,:),indplot);
    sliceInt = [startTime, endTime];
else
    SSMDim = 2;
    sliceInt = [10, endTime];
end

overEmbed = 0;
ShiftStep = 1;
oblique_projection = true;

if ~automatic 
    lim = 2e-7; 
    indStartTime = regimeLinear(xData,lim);
end

[yData, P, V_trunc_slow,data_non_projected,data_non_projected_trunc] = obliqueProjection_fullPhaseSpace(xData,indStartTime,SSMDim, overEmbed, ShiftStep, 'flag_end',0,'oblique_projection',oblique_projection);

%% computation of SSM model with linear oblique projection through minimization of the oscillations of the backbone curve (Bettini et al., Chaos (2025))

yDataTrunc = sliceTrajectories(yData, sliceInt);

indTrain = 1;
indTest = 1;

% manifold parametrization
SSMOrder = 5;

% compute the reduced coordinates as the oblique projection of the
% coordinates in the full observable space onto the slow subspace 

[IMInfo, ~, ~] = IMGeometry(yDataTrunc(indTrain,:), SSMDim, 1,'V_e',V_trunc_slow);
SSMChart_P = @(x) IMInfo.parametrization.tangentSpaceAtOrigin'*P*x;
etaData = projectTrajectories(IMInfo, yData);
etaDataTrunc = projectTrajectories(IMInfo, yDataTrunc);

yData_original = data_non_projected;
yData_original_Trunc = sliceTrajectories(yData_original, sliceInt);
[IMInfo, SSMChart, SSMFunction] = IMGeometry(yData_original_Trunc(indTrain,:), SSMDim, SSMOrder, 'reducedCoordinates', etaDataTrunc(indTrain,:));
IMInfo.chart.map = SSMChart_P;

plotSSMWithTrajectories(IMInfo, 1, yData_original_Trunc(indTrain,:))
view(-100,20); zlabel('$u \, [$m$]$','Interpreter','latex')

% manifold reduced dynamics 
ROMOrder = 5;
RDInfo = IMDynamicsFlow(etaDataTrunc(indTrain,:), ...
    'R_PolyOrd', ROMOrder, 'style', 'normalform');

omega_0 = imag(lambda_t(1));
zData = transformTrajectories(RDInfo.inverseTransformation.map, etaData);
rhoCal = abs(zData{indTest(1),2}(1,1));
amplitudeFunction = @(x) x(1,:);
BBCInfo = backboneCurves(IMInfo, RDInfo, amplitudeFunction, rhoCal);


kmean = 1;       
[amp_sol,freq_sol,damp_sol,time_sol] = PFFk(t_t_off, xOutput(x_t_off),kmean);
[amp_z,freq_z,damp_z,time_z] = PFFk(t_t_off,xOutput(yData{1,2}),kmean);

lim = 1e-3;
[freq_spline_sol, amp_spline_sol] = bc_spline(freq_sol, amp_sol, lim);
[freq_spline_z, amp_spline_z] = bc_spline(freq_z, amp_z, lim);
fig = figure;
fig.Position = [476,255,560,617];
hold on 
grid on 
plot(freq_spline_sol, amp_spline_sol,'Color',[0.65,0.65,0.65],'LineWidth',4,'DisplayName','Data')
% plot(freq_spline_z, amp_spline_z,'k-','LineWidth',4,'DisplayName','Projected data')
plot(BBCInfo.frequency, BBCInfo.amplitude,'color',[0.47,0.67,0.19],'LineWidth',4,'Displayname','ROM prediction')
xlabel('Frequency [rad/s]','Interpreter','Latex')
ylabel('$|q_1|$','Interpreter','Latex')
set(gca,'FontSize',30,'TickLabelInterpreter','Latex', ...
    'OuterPosition',[0,0,1,1.05]);
xlim([0.96 1.09])
legend('location','NW','Interpreter','latex')

fig = figure;
fig.Position = [476,255,560,617];
hold on 
grid on 
plot(freq_spline_sol, amp_spline_sol,'Color',[0.65,0.65,0.65],'LineWidth',4,'DisplayName','Backbone from data')

xlabel('Frequency [rad/s]','Interpreter','Latex')
ylabel('$|q_1|$','Interpreter','Latex')
set(gca,'FontSize',30,'TickLabelInterpreter','Latex', ...
    'OuterPosition',[0,0,1,1.05]);
xlim([omega_0-0.5 omega_0+0.5])
xlim([0.96 1.09])
legend('location','NW','Interpreter','latex')

omegaSpan_bottom = 0.98;
omegaSpan_top = 1.08;

% FRC from the reduced order model 
amp_lim = 1;
amplitudeCal = [0.0435 0.108709 0.2165 0.322 0.43 0.533 0.635];
forcingSpan = zeros(1,length(amplitudeCal));
for iCal = 1:length(amplitudeCal)
   [~,pos] = min(abs(BBCInfo.amplitude-amplitudeCal(iCal)));
   forcingSpan(iCal) = -BBCInfo.amplitudeNormalForm(pos)*BBCInfo.damping(pos);
end
[IMInfoF,RDInfoF] = forcedSSMROM(IMInfo,RDInfo,'nForcingFrequencies',1);
FRCData = analyticalFRC(IMInfoF,RDInfoF,forcingSpan,amplitudeFunction);

for ii = 1:length(amplitudeCal)
    
    forced_freq = FRCData.(['F' num2str(ii)]).Freq(:);
    forced_amp = FRCData.(['F' num2str(ii)]).Amp(:);
    
    index_plot = find(forced_freq>=omegaSpan_bottom & forced_freq<=omegaSpan_top & forced_amp<amp_lim);
    forced_freq_plot = forced_freq(index_plot);
    forced_amp_plot = forced_amp(index_plot);
    
    if ii == 1
        plot(forced_freq_plot, forced_amp_plot,'.r','LineWidth',1,'MarkerSize',15,'DisplayName','FRC from SSM model')
    else
        plot(forced_freq_plot, forced_amp_plot,'.r','LineWidth',1,'MarkerSize',15,'HandleVisibility','off')
    end

end


% generate forced response curves from the original system 
freq_vect = linspace(omegaSpan_bottom,omegaSpan_top,50);
amp_vect = [0.001 0.0025 0.005 0.0075 0.01 0.0125 0.015];

for jj = 1:length(amp_vect)
    ampl_FRC = zeros(size(freq_vect));
    freq_FRC = zeros(size(freq_vect));
    amp_curr = amp_vect(jj);
    for ii = 1:length(freq_vect)
        Omega = freq_vect(ii);
        F_forced = @(t,x) F_t(t,x) + amp_curr * cos(Omega*t);
        [t_forced, x_forced] = ode45(F_forced, tSpan_decay, x0_ssm_t);
        t_forced = transpose(t_forced);
        x_forced = transpose(x_forced);
 
        signal = x_forced;
        data = cell(1,2);
        data{1,1} = t_forced;
        data{1,2} = signal;

        kmean = 1;
        [amp_frc,freq_frc,~,~] = PFFk(data{1,1},data{1,2}(1,:),kmean);
        amp_frc = amp_frc(~isnan(freq_frc));
        freq_frc = freq_frc(~isnan(freq_frc));
        ampl_FRC(ii) = amp_frc(end);
        freq_FRC(ii) = 2*pi*freq_frc(end);
    
    end
    
    if jj == 1
        plot(freq_FRC, ampl_FRC,'k.','LineWidth',1,'MarkerSize',15,'DisplayName','FRC from data')
    else
        plot(freq_FRC, ampl_FRC,'k.','LineWidth',1,'MarkerSize',15,'HandleVisibility','off')
    end

end

plot(BBCInfo.frequency, BBCInfo.amplitude,'color',[0.47,0.67,0.19],'LineWidth',4,'Displayname','Backbone from SSM model')

h = get(gca,'Children');
legend([h(4) h(1) h(2) h(3)]);

%% computation of SSM model with linear oblique projection through foliation invariance


% estimation of the orthogonal complement of the fast subspace for the initial condition of the optimization process

t = horzcat(xData{:,1});
X = horzcat(xData{:,2});
% compute the derivative of the data
iStart = 1;
for iTraj = 1:size(xData,1); iStart(iTraj+1) = iStart(iTraj)+size(xData{iTraj,1},2); end
Xdot = []; Xnew = [];
for iTraj = 1:size(xData,1)
    [xdot,xnew] = ftd(X(:,iStart(iTraj):iStart(iTraj+1)-1),t);
    Xdot = [Xdot, xdot]; Xnew = [Xnew, xnew];
end

A_est = Xdot / Xnew;
A_est_adj = transpose(A_est);
[V_adj, lambda_adj] = eigSorted(A_est_adj);

B_adj = [real(V_adj(:,1)) imag(V_adj(:,1))];

% Compute the linear oblique projection from the invariance equation of the foliation 
SSMDim = 2;

% rearrange data
t = horzcat(xData{:,1});
X = horzcat(xData{:,2});

% now compute the tangent space from data via SVD
mfddim = 2;
[u,s,v] = svds(X, mfddim);
V_e = max(abs(v'),[],2).'.*u*s;

% impose the constraint that V^T * V = I
V_tg = orthogonalizeGramSchmidt(V_e);

% compute the derivative of the data
iStart = 1;
for iTraj = 1:size(xData,1); iStart(iTraj+1) = iStart(iTraj)+size(xData{iTraj,1},2); end
Xdot = []; Xnew = [];
for iTraj = 1:size(xData,1)
    [xdot,xnew] = ftd(X(:,iStart(iTraj):iStart(iTraj+1)-1),t);
    Xdot = [Xdot, xdot]; Xnew = [Xnew, xnew];
end

% optimization process
B0 = V_tg;
% B0 = B_adj;
[nB,mB] = size(B0);


% approximate the slowest eigenvalues (SVD/DMD)
dt = mean(mean(diff(t,[],2)));

X1 = X(:,1:end-1);
X2 = X(:,2:end);

[U,S,V] = svds(X1,4);
U = U(:,1:2*SSMDim);
S = S(1:2*SSMDim,1:2*SSMDim);
V = V(:,1:2*SSMDim);
S_tilde = U.'*X2*V*pinv(S);
[E,~,Lambda] = eigSorted(S_tilde);
lambda_A_from_S_tilde = 1/dt*log(Lambda);  
[~,index_sort] = sort(abs(real(lambda_A_from_S_tilde)));
lambda_trunc = lambda_A_from_S_tilde(index_sort);

R_tilde0 = [real(lambda_trunc(1)) imag(lambda_trunc(1));
            -imag(lambda_trunc(1)) real(lambda_trunc(1))];

[nR,mR] = size(R_tilde0);
ROMOrder = 5;
phi_0 = phi_mv(inv(transpose(B0)*V_tg) * transpose(B0) * Xnew, 2:ROMOrder);
N0 = zeros(size(R_tilde0,1), size(phi_0,1));
[nN,mN] = size(N0);

x0 = [B0(:); R_tilde0(:); N0(:)];
J = @(X) OblProjMat_nonlin(X,Xdot,Xnew,nB,mB,nR,mR,nN,mN,ROMOrder,V_tg);


[aa0,bb0] = idemP(x0,V_tg,nB,mB);

% constraint 
nonlin_constraint = @(X) idemP(X,V_tg,nB,mB);

options = optimoptions('fminunc', 'Display', 'iter','MaxFunctionEvaluation',1e5);
[x_opt, fval] = fminunc(J,x0,options);

xB = x_opt(1:nB*mB);
B = reshape(xB,nB,mB);
xR_tilde = x_opt(1+nB*mB:nB*mB+nR*mR);
R_tilde = reshape(xR_tilde,nR,mR);
xN = x_opt(1+nB*mB+nR*mR:end);
N = reshape(xN,nN,mN);

P_opt = V_tg * inv(transpose(B) * V_tg) * transpose(B);

[aa,bb] = idemP(x_opt,V_tg,nB,mB);


R = R_tilde;
Rmap = @(t,xi) R*xi + N*phi_mv(xi, 2:ROMOrder);

% transformation to normal form 

redData = cell(1,2);
redData{1,1} = t;
redData{1,2} = transpose(V_tg) * P_opt * X;
 
RDInfo = NormalFormTransformation(R,N,SSMDim,redData,'R_PolyOrd',ROMOrder,'style', 'normalform');

% SSM parametrization 
Xi = transpose(V_tg) * P_opt * X;
SSMOrder = 5;
phi_M0 = phi_mv(Xi,2:SSMOrder);
M0 = zeros(size(X,1),size(phi_M0,1));
J_param = @(x) LinOblParam(x,Xnew,V_tg,P_opt,SSMOrder);
nonlin_constraint_param = @(x) orthogV_tPM(x,V_tg,P_opt);
options = optimoptions('fmincon', 'Display', 'iter','MaxFunctionEvaluation',1e5);
[M_opt, ~] = fmincon(J_param, M0, [], [], [], [], [], [], nonlin_constraint_param, options);

x_ssm = @(xi) V_tg * xi + M_opt * phi_mv(xi, 2:SSMOrder);

IMInfo.chart.map = @(x) transpose(V_tg) * P_opt * x;
IMInfo.parametrization.map = @(q) x_ssm(q);
IMInfo.parametrization.polynomialOrder = SSMOrder;
IMInfo.parametrization.dimension = SSMDim;
IMInfo.parametrization.tangentSpaceAtOrigin = V_tg;
IMInfo.parametrization.nonlinearCoefficients = M_opt;


% check the backbone curve 

indTest = 1;
omega_0 = imag(lambda_t(1));
zData = transformTrajectories(RDInfo.inverseTransformation.map, redData);
rhoCal = abs(zData{indTest(1),2}(1,1));
amplitudeFunction = @(x) x(1,:);
BBCInfo = backboneCurves(IMInfo, RDInfo, amplitudeFunction, rhoCal);

kmean = 1;       
[amp_sol,freq_sol,damp_sol,time_sol] = PFFk(t_t_off, xOutput(x_t_off),kmean);
[amp_f,freq_f,damp_f,time_f] = PFFk(t_t_off,xOutput(P_opt * X),kmean);

lim = 1e-3;
[freq_spline_sol, amp_spline_sol] = bc_spline(freq_sol, amp_sol, lim);
[freq_spline_f, amp_spline_f] = bc_spline(freq_f, amp_f, lim);
fig = figure;
fig.Position = [476,255,560,617];
hold on 
grid on 
plot(freq_spline_sol, amp_spline_sol,'Color',[0.65,0.65,0.65],'LineWidth',4,'DisplayName','Data')
plot(BBCInfo.frequency, BBCInfo.amplitude,'color',[0.47,0.67,0.19],'LineWidth',4,'Displayname','ROM prediction')
xlabel('Frequency [rad/s]','Interpreter','Latex')
ylabel('$|q_1|$','Interpreter','Latex')
set(gca,'FontSize',30,'TickLabelInterpreter','Latex', ...
    'OuterPosition',[0,0,1,1.05]);
xlim([0.96 1.09])
legend('location','NW','Interpreter','latex')

% check nonautonomous system (forced response curves)

fig = figure;
fig.Position = [476,255,560,617];
hold on 
grid on 
plot(freq_spline_sol, amp_spline_sol,'Color',[0.65,0.65,0.65],'LineWidth',4,'DisplayName','Backbone from data')

xlabel('Frequency [rad/s]','Interpreter','Latex')
ylabel('$|q_1|$','Interpreter','Latex')
set(gca,'FontSize',30,'TickLabelInterpreter','Latex', ...
    'OuterPosition',[0,0,1,1.05]);
xlim([omega_0-0.5 omega_0+0.5])
xlim([0.96 1.09])
legend('location','NW','Interpreter','latex')

omegaSpan_bottom = 0.98;
omegaSpan_top = 1.08;

% FRC from the reduced order model 
amp_lim = 1;
amplitudeCal = [0.0435 0.108709 0.2165 0.322 0.43 0.537 0.64];
forcingSpan = zeros(1,length(amplitudeCal));
for iCal = 1:length(amplitudeCal)
   [~,pos] = min(abs(BBCInfo.amplitude-amplitudeCal(iCal)));
   forcingSpan(iCal) = -BBCInfo.amplitudeNormalForm(pos)*BBCInfo.damping(pos);
end
[IMInfoF,RDInfoF] = forcedSSMROM(IMInfo,RDInfo,'nForcingFrequencies',1);
FRCData = analyticalFRC(IMInfoF,RDInfoF,forcingSpan,amplitudeFunction);

for ii = 1:length(amplitudeCal)
    
    forced_freq = FRCData.(['F' num2str(ii)]).Freq(:);
    forced_amp = FRCData.(['F' num2str(ii)]).Amp(:);
    
    index_plot = find(forced_freq>=omegaSpan_bottom & forced_freq<=omegaSpan_top & forced_amp<amp_lim);
    forced_freq_plot = forced_freq(index_plot);
    forced_amp_plot = forced_amp(index_plot);
    
    if ii == 1
        plot(forced_freq_plot, forced_amp_plot,'.r','LineWidth',1,'MarkerSize',15,'DisplayName','FRC from SSM model')
    else
        plot(forced_freq_plot, forced_amp_plot,'.r','LineWidth',1,'MarkerSize',15,'HandleVisibility','off')
    end

end

% generate forced response curves from the original system 
freq_vect = linspace(omegaSpan_bottom,omegaSpan_top,50);
amp_vect = [0.001 0.0025 0.005 0.0075 0.01 0.0125 0.015];

for jj = 1:length(amp_vect)
    ampl_FRC = zeros(size(freq_vect));
    freq_FRC = zeros(size(freq_vect));
    amp_curr = amp_vect(jj);
    for ii = 1:length(freq_vect)
        Omega = freq_vect(ii);
        F_forced = @(t,x) F_t(t,x) + amp_curr * cos(Omega*t);
        [t_forced, x_forced] = ode45(F_forced, tSpan_decay, x0_ssm_t);
        t_forced = transpose(t_forced);
        x_forced = transpose(x_forced);
 
        signal = x_forced;
        data = cell(1,2);
        data{1,1} = t_forced;
        data{1,2} = signal;

        kmean = 1;
        [amp_frc,freq_frc,~,~] = PFFk(data{1,1},data{1,2}(1,:),kmean);
        amp_frc = amp_frc(~isnan(freq_frc));
        freq_frc = freq_frc(~isnan(freq_frc));
        ampl_FRC(ii) = amp_frc(end);
        freq_FRC(ii) = 2*pi*freq_frc(end);
    
    end
    
    if jj == 1
        plot(freq_FRC, ampl_FRC,'k.','LineWidth',1,'MarkerSize',15,'DisplayName','FRC from data')
    else
        plot(freq_FRC, ampl_FRC,'k.','LineWidth',1,'MarkerSize',15,'HandleVisibility','off')
    end

end

plot(BBCInfo.frequency, BBCInfo.amplitude,'color',[0.47,0.67,0.19],'LineWidth',4,'Displayname','Backbone from SSM model')

h = get(gca,'Children');
legend([h(4) h(1) h(2) h(3)]);



%% Additional functions
function [F,A,M,C,K,F_nl] = SP(k1,k2,c1,c2,alpha,epsilon,Omega)

M = eye(2);
C = [c1 + c2, -c2;
    -c2, c1+c2];
K = [k1 + k2, -k2;
    -k2, k1+k2];

f_nl = @(t,x) [alpha * x(1).^3; 0];
f_ext = @(t,x) [epsilon * cos(Omega * t); 0];

A = [zeros(2) eye(2);
    -M\K -M\C];

F_nl = @(t,x) [zeros(2,1); -M\f_nl(t,x)] + [zeros(2,1); M\f_ext(t,x)];

F = @(t,x) A * x + [zeros(2,1); -M\f_nl(t,x)] + [zeros(2,1); M\f_ext(t,x)];

end

function plotSSM_ssmtool()

    load('data_oscillator/XX.mat','XX');
    load('data_oscillator/YY.mat','YY');
    load('data_oscillator/ZZ.mat','ZZ');
    
    % customFigure; 
    classicColors = colororder;
    fig = figure;
    hold on 
    grid on 
    fig.Position = [696,337,560,491];
    h = surf(XX,YY,ZZ,'HandleVisibility','off');
    h.FaceColor = classicColors(1,:);
    h.FaceColor = [0.47,0.67,0.19];
    h.EdgeColor = 'none';
    h.FaceAlpha = .5;
    xlabel('$x_1 \, [$m$]$','Interpreter','latex','FontSize',30,'Position',[0.104356912417099,-1.207746549280131,-0.237632362684135]); 
    zlabel('$x_2 \, [$m$]$','Interpreter','latex','FontSize',30,'Position',[-1.467103409674007,1.14325138688222,-0.034514954675319]); 
    ylabel('$x_3\, [$m/s$]$','Interpreter','latex','FontSize',30,'Position',[-1.259382664235982,-0.112477314059568,-0.240349125976296]); 
    view(-20.122788542926202,34.822508198240442)
    set(gca,'FontSize',30,'TickLabelInterpreter','latex', ...
        'OuterPosition',[0.050396865838005,0.068582337162595,0.936174462145399,0.786624455018657], ...
        'InnerPosition',[0.23954969276697,0.162583751291155,0.66545030723303,0.634255930702729], ...
        'Position',[0.23954969276697,0.162583751291155,0.66545030723303,0.634255930702729]);
     
end

function [freq_fit_spline, amp_fit_spline] = bc_spline(freq, amp, lim)
    freq_spline = flip(2*pi*freq(amp>lim));
    amp_spline = flip(amp(amp>lim));
    amp_spline = amp_spline(~isnan(freq_spline));
    freq_spline = freq_spline(~isnan(freq_spline));
    amp_fit_spline = linspace(amp_spline(1),amp_spline(end),1000);
    freq_fit_spline = spline(amp_spline,freq_spline,amp_fit_spline);
end

function [t_t, x_t, t_t_off, x_t_off, x0_ssm_t] = plot_SSMTool(F_t,tSpan_decay,V_t)

rho_train_t = 0.7;
theta_train_t = 0;
ztrain_t = rho_train_t * exp(1i * theta_train_t);
ztrain_t = [ztrain_t; conj(ztrain_t)];
plotSSM_ssmtool
load('data_oscillator/x0_ssm_t.mat','x0_ssm_t');
[t_t, x_t] = ode45(F_t, tSpan_decay, x0_ssm_t);
t_t = transpose(t_t);
x_t = transpose(x_t);
ratio_first = 1;
ratio_second = 2;
x_ssm_t_complex = V_t\x0_ssm_t;
x_ssm_t_complex_mag = abs(x_ssm_t_complex);
x_ssm_t_complex_phase = angle(x_ssm_t_complex);
new_rho = [x_ssm_t_complex_mag(1) * ratio_first; x_ssm_t_complex_mag(2) * ratio_second; x_ssm_t_complex_mag(3) * ratio_first; x_ssm_t_complex_mag(4) * ratio_second];
new_x_ssm_t_complex = new_rho .* cos(x_ssm_t_complex_phase) + 1i * new_rho .* sin(x_ssm_t_complex_phase);
new_x_ssm_t = V_t * new_x_ssm_t_complex;
x0_t_off = real(new_x_ssm_t);
[t_t_off, x_t_off] = ode45(F_t, tSpan_decay, x0_t_off);
t_t_off = transpose(t_t_off);
x_t_off = transpose(x_t_off);

plot3(x_t(1,:),x_t(3,:),x_t(2,:),'color',[0.07,0.62,1.00],'LineWidth',3,'DisplayName','Trajectory on primary SSM')
plot3(x_t_off(1,:),x_t_off(3,:),x_t_off(2,:),'k','LineWidth',3,'DisplayName','Trajectory on fractional SSM')

legend('location','NE','Interpreter','latex','Position',[0.163971949986049,0.838998539133578,0.691028050013951,0.14114053186234])

end

function [dXdt, Xtrunc] = ftd(X, t) % finite time difference
    ind = 5:size(X,2)-4; Xtrunc = X(:,ind);
    dX = 4/5*(X(:,ind+1)-X(:,ind-1)) - 1/5*(X(:,ind+2)-X(:,ind-2)) + ...
        4/105*(X(:,ind+3)-X(:,ind-3)) - 1/280*(X(:,ind+4)-X(:,ind-4));
    dXdt = dX./(t(ind+1)-t(ind));
end

function exps=exponents(d,k)
    B = repmat({0:max(k)},1,d);
    A = combvec(B{:}).';
    exps = A(ismember(sum(A,2), k),:);
    [~,ind] = sort(sum(exps,2));
    exps = exps(ind,:);
end

function u = phi_mv(xi, r) % return monomials
    x = reshape(xi, 1, size(xi, 1), []);
    exps = exponents(size(xi, 1),r);
    u = reshape(prod(x.^exps, 2), size(exps, 1), []);
end

function J = OblProjMat_nonlin(x,Xdot,Xnew,nB,mB,nR,mR,nN,mN,ROMOrder,V)

xB = x(1:nB*mB);
B = reshape(xB,nB,mB);

B = B / sqrt(trace(B'*B)); 


xR_tilde = x(1+nB*mB:nB*mB+nR*mR);
R_tilde = reshape(xR_tilde,nR,mR);

xN = x(1+nB*mB+nR*mR:end);
N = reshape(xN,nN,mN);

LHS = transpose(B) * Xdot;
RHS = (transpose(B)*V) * R_tilde * inv(transpose(B)*V) * transpose(B) * Xnew + (transpose(B)*V) * N * phi_mv(inv(transpose(B)*V) * transpose(B) * Xnew, 2:ROMOrder);


Err = LHS - RHS;

J = norm(Err);

end

function [c,ceq] = idemP(x,V,nB,mB)
xB = x(1:nB*mB);
B = reshape(xB,nB,mB);

c = [];
M = V * ((B' * V) \ B');
ceq = norm(M*M - M, 'fro');



end

function [c,ceq] = orthogV_tPM(x,V,P)
c = [];
ceq = transpose(V) * P * x;
end

function J = LinOblParam(x,X,V,P,SSMOrder)

Xi = transpose(V)*P*X;
Err = X - V*Xi - x*phi_mv(Xi,2:SSMOrder);
J = norm(Err);

end

function RDInfo = NormalFormTransformation(R,N,SSMDim,redData,varargin)

etaData = redData;

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
L2 = (1+options.c1*exp(-options.c2*t)).^(-2);
options = setfield(options,'L2',L2);

% reduced dynamics 
W_r = [R N];
k = size(W_r,1);
[V,D,d] = eigSorted(W_r(:,1:SSMDim));

[phi,Expmat] = multivariatePolynomial(k,1,options.R_PolyOrd);
W_r_known = W_r;
nCoefs = size(W_r_known,2);
l_opt = 0; Err = 0;
R = @(x) W_r*phi(x);
R_info = assembleStruct(@(x) W_r*phi(x),W_r,phi,Expmat,l_opt,Err);
options.l = l_opt;

Maps_info_opt=initialize_nf_flow(V,D,d,W_r,etaData,options);
Maps = dynamicsCoordChangeNF(Maps_info_opt,options);

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

RDInfo = struct('reducedDynamics',R_info,'inverseTransformation',...
    iT_info,'conjugateDynamics',N_info,'transformation',T_info,...
    'conjugacyStyle',options.style,'dynamicsType','flow',...
    'eigenvaluesLinPartFlow',d,'eigenvectorsLinPart',V);

end

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
    'SpecifyObjectiveGradient',true);
% Default case
if nargin_o == 5; options.R_PolyOrd = varargin_o{:};
    options.N_PolyOrd = varargin_o{:}; end
% Custom options
if nargin_o > 5
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