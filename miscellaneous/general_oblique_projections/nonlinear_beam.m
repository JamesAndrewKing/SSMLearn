% SSM-based model reduction through oblique projection of an experimental
% nonlinear beam

clear all
close all

%% load experimental data 

load('nonlinear_beam_experiments/decaying_data.mat')

fact_amp = 50*4;
signal_decay = fact_amp * dev4955.scopes.wave{1,1}.wave;
signal_net = signal_decay(round(length(signal_decay)*0.03):round(length(signal_decay)*0.7));
dt = dev4955.scopes.wave{1,1}.dt;
t_vect = linspace(0,length(signal_net)*dt, length(signal_net));

dt = diff(t_vect(1:2));

% backbone curve of experimental data 
index = length(t_vect);
kmean = 1;
Y = signal_net;
X = [ones(length(signal_net),1) t_vect.'];
B = X\Y;
signal_centered = signal_net - B(1);

signal_filtered = fliplr(bandpass(fliplr(signal_centered),[30 250],1/dt));
[amp,freq,damp,time] = PFFk(t_vect(1:index),signal_filtered(1:index),kmean);

f = figure;
f.Position = [406,328,590,538];
hold on 
grid on 
plot(2*pi*freq, amp,'color','k','LineWidth',4,'DisplayName','Backbone curve from experiment')

xlim([268 269.5])
ylim([0 100])
xlabel('Frequency [rad/s]','Interpreter','Latex')
ylabel('Amplitude [mm/s]','Interpreter','Latex')
set(gca,'FontSize',30,'TickLabelinterpreter','latex')
legend('location','N','Position',[0.163388475321107,0.931412638341626,0.731526778916181,0.067657993628633],'Interpreter','latex')

%% Preliminary computations

lim = 5e-9; % user dependent parameter
SSMDim = 2;

overEmbed = 1;
ShiftStep = 1;
norm = 0; % if norm = 0, then we apply the linear oblique projection. If norm = 1, then we employ the standard normal projection.
xData = cell(1,2);
xData{1,1} = t_vect;
xData{1,2} = transpose(signal_filtered);
[yData,~,V_trunc_slow,data_non_projected,data_non_projected_linear,~,B_min,lambda_trunc] = oblique_projection(xData,lim,SSMDim, overEmbed, ShiftStep, norm, 2);

yData_oscill = yData;
X = data_non_projected{1,2};

% data projected on tangent space 
VData = transpose(V_trunc_slow) * X;

% data projected on the orthogonal complement of the fast susbspace 
BData = transpose(B_min) * X;

%% computation of the oblique projection through the invariance of the foliation 

% rearrange data 
t = horzcat(data_non_projected{:,1});
X = horzcat(data_non_projected{:,2});

% tangent space
V_tg = V_trunc_slow;
% compute the derivative of the data
iStart = 1;
for iTraj = 1:size(data_non_projected,1); iStart(iTraj+1) = iStart(iTraj)+size(data_non_projected{iTraj,1},2); end
Xdot = []; Xnew = [];
for iTraj = 1:size(data_non_projected,1)
    [xdot,xnew] = ftd(X(:,iStart(iTraj):iStart(iTraj+1)-1),t);
    Xdot = [Xdot, xdot]; Xnew = [Xnew, xnew];
end


% estimation of the orthogonal complement of the fast subspace B

t_lin = horzcat(data_non_projected{:,1});
X_lin = horzcat(data_non_projected{:,2});

iStart = 1;
for iTraj = 1:size(data_non_projected,1); iStart(iTraj+1) = iStart(iTraj)+size(data_non_projected{iTraj,1},2); end
Xdot_lin = []; Xnew_lin = [];
for iTraj = 1:size(data_non_projected,1)
    [xdot_lin,xnew_lin] = ftd(X_lin(:,iStart(iTraj):iStart(iTraj+1)-1),t_lin);
    Xdot_lin = [Xdot_lin, xdot_lin]; Xnew_lin = [Xnew_lin, xnew_lin];
end

A_est = Xdot_lin / Xnew_lin;
A_est_adj = transpose(A_est);
[V_adj, lambda_adj] = eigSorted(A_est_adj);
B_adj = [real(V_adj(:,1)) imag(V_adj(:,1))];
B0 = B_adj;

P0 =  V_trunc_slow * inv(transpose(B0) * V_trunc_slow) * transpose(B0);

redData = transpose(V_trunc_slow) * P0 * X;
rData = transpose(V_trunc_slow) * yData_oscill{1,2};

[nB,mB] = size(B0);

R_tilde0 = [real(lambda_trunc(1)) -imag(lambda_trunc(1));
            imag(lambda_trunc(1)) real(lambda_trunc(1))];

[nR,mR] = size(R_tilde0);
ROMOrder = 5;
P0 = V_tg * inv(transpose(B0) * V_tg) * transpose(B0);
phi_0 = phi_mv(inv(transpose(B0)*V_tg) * transpose(B0) * Xnew, 2:ROMOrder);
N0 = zeros(size(R_tilde0,1), size(phi_0,1));
[nN,mN] = size(N0);

x0 = [B0(:); R_tilde0(:); N0(:)];
J = @(X) OblProjMat_nonlin(X,Xdot,Xnew,nB,mB,nR,mR,nN,mN,ROMOrder,V_tg);

nonlin_constraint = @(X) idemP(X,V_tg,nB,mB);

options = optimoptions('fminunc', 'Display', 'iter','MaxFunctionEvaluation',1e5);
[x_opt, fval] = fminunc(J,x0,options);

xB = x_opt(1:nB*mB);
B = reshape(xB,nB,mB);
xR_tilde = x_opt(1+nB*mB:nB*mB+nR*mR);
R_tilde = reshape(xR_tilde,nR,mR);
xN = x_opt(1+nB*mB+nR*mR:end);
N = reshape(xN,nN,mN);

P = V_tg * inv(transpose(B) * V_tg) * transpose(B);

[aa,bb] = idemP(x_opt,V_tg,nB,mB);

R = R_tilde;
Rmap = @(t,xi) R*xi + N*phi_mv(xi, 2:ROMOrder);

redData = cell(1,2);
redData{1,1} = t;
redData{1,2} = transpose(V_tg) * P * X;

yData = cell(1,2);
yData{1,1} = t;
yData{1,2} = P * X;

xOutput = @(x) x(1,:);
kmean = 1;       
[amp_z,freq_z,damp_z,time_z] = PFFk(t_vect,xOutput(yData{1,2}),kmean);

% refining the model, given the reduced coordinates found through the
% oblique porjection

endTime = yData{1,1}(end);
sliceInt = [1, endTime];
yDataTrunc = sliceTrajectories(yData, sliceInt);

indTrain = 1;
indTest = 1;

% manifold parametrization
SSMOrder = 7;

% compute the reduced coordinates as the oblique projection of the
% coordinates in the full observable space onto the slow subspace 

[IMInfo, ~, ~] = IMGeometry(yDataTrunc(indTrain,:), SSMDim, 1,'Ve',V_trunc_slow);
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
ROMOrder = 7;
RDInfo = IMDynamicsFlow(etaDataTrunc(indTrain,:), ...
    'R_PolyOrd', ROMOrder, 'style', 'normalform');

%% backbone curve 

zData = transformTrajectories(RDInfo.inverseTransformation.map, etaData);
rhoCal = abs(zData{indTest(1),2}(1,1));
amplitudeFunction = @(x) x(1,:);

BBCInfo = backboneCurves(IMInfo, RDInfo, amplitudeFunction, rhoCal);
subplot(121); ylabel('Amplitude [mm/s]','Interpreter','latex')
subplot(122); ylabel('Amplitude [mm/s]','Interpreter','latex')

f = figure;
f.Position = [705,132,641,617];
hold on 
grid on 
plot(2*pi*freq, amp,'Color',[0.65,0.65,0.65],'LineWidth',4,'DisplayName','Data')
plot(BBCInfo.frequency, BBCInfo.amplitude,'color',[0.47,0.67,0.19],'LineWidth',4,'Displayname','ROM prediction')
xlabel('Frequency [rad/s]','Interpreter','Latex')
ylabel('Amplitude [mm/s]','Interpreter','Latex')
set(gca,'FontSize',30,'TickLabelInterpreter','latex')
ylim([0 100])
xlim([268 269.5])
legend('location','NE','Interpreter','latex')

%% forced response curves 

f = figure;
f.Position = [705,132,641,617];
hold on 
grid on 
plot(2*pi*freq,amp,'Color',[0.65,0.65,0.65],'LineWidth',4,'DisplayName','Data')
plot(BBCInfo.frequency(BBCInfo.amplitude<110), BBCInfo.amplitude(BBCInfo.amplitude<110),'color',[0.39,0.83,0.07],'LineWidth',4,'Displayname','ROM prediction')
xlabel('Frequency [rad/s]','Interpreter','Latex')
ylabel('Amplitude [mm/s]','Interpreter','Latex')
set(gca,'FontSize',30,'TickLabelinterpreter','latex','Position',[0.158264687278054,0.136904373539905,0.613966201957516,0.788095626460096])
ylim([0 100])
xlim([268 269.5])
legend('location','NE','Interpreter','latex','Position',[0.5231,0.6549,0.466,0.2723]);

% extract data from forced experiment
load('nonlinear_beam_experiments/forced_data.mat');

X_data = forced_data.Results.X1_avg__V_.data;
Y_data = forced_data.Results.Y1_avg__V_.data;
freq_FRC = forced_data.Results.Frequency_of_MFLI_avg__Hz_.data;
% compute the amplitude
R_data = sqrt(X_data.^2 + Y_data.^2);
% transform to physical quantities
R_data_scope = sqrt(2) * R_data;
range = 10; % property of the scope: do not change
ampl_FRC = 4 * range * R_data_scope; % property of the scope: do not change 
phase_data = atan2(Y_data,X_data);


% from the model 

omegaSpan_bottom = 268.3;
omegaSpan_top = 269.1;
amplitudeCal = [38.41, 56.00, 71.95, 93.83];
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
    
    index_plot = find(forced_freq>=omegaSpan_bottom & forced_freq<=omegaSpan_top);
    forced_freq_plot = forced_freq(index_plot);
    forced_amp_plot = forced_amp(index_plot);
    
    if ii == 1
        plot(forced_freq_plot, forced_amp_plot,'.r','LineWidth',1,'MarkerSize',15,'DisplayName','FRC ROM prediction')
    else
        plot(forced_freq_plot, forced_amp_plot,'.r','LineWidth',1,'MarkerSize',15,'HandleVisibility','off')
    end

end

[amplitude, index] = unique(forced_data.Data.Amplitude_of_Excitation__V_.data);
ampl_FRC_proj = ampl_FRC;
freq_FRC_proj = 2*pi*freq_FRC;
n_length = length(freq_FRC)/length(amplitude);
plot(freq_FRC_proj(1:n_length), ampl_FRC_proj(1:n_length),'k.','LineWidth',1,'MarkerSize',15,'DisplayName','FRC data');
plot(freq_FRC_proj(n_length+1:2*n_length), ampl_FRC_proj(n_length+1:2*n_length),'k.','LineWidth',1,'MarkerSize',15,'HandleVisibility','off');
plot(freq_FRC_proj(2*n_length+1:3*n_length), ampl_FRC_proj(2*n_length+1:3*n_length),'k.','LineWidth',1,'MarkerSize',15,'HandleVisibility','off');
plot(freq_FRC_proj(3*n_length+1:4*n_length), ampl_FRC_proj(3*n_length+1:4*n_length),'k.','LineWidth',1,'MarkerSize',15,'HandleVisibility','off');

%% phase curves 

figure; hold on; grid on;
xlabel('Frequency [rad/s]','Interpreter','Latex')
ylabel('Phase [deg]','Interpreter','Latex')
set(gca,'FontSize',20,'TickLabelinterpreter','latex')
xlim([268.2 269.2])
ylim([-180 0]);
legend('location','NE','Interpreter','latex')

yticks([-180 -135 -90 -45 0 45 90]);
yticklabels({'-180','-135','-90','-45','0','45','90'});

% reduced order model
for ii = 1:length(amplitudeCal)
    
    forced_freq = FRCData.(['F' num2str(ii)]).Freq(:);
    forced_phase = 180/pi*FRCData.(['F' num2str(ii)]).Nf_Phs(:);
    
    index_plot = find(forced_freq>=omegaSpan_bottom & forced_freq<=omegaSpan_top);
    forced_freq_plot = forced_freq(index_plot);
    forced_phase_plot = forced_phase(index_plot);
    
    if ii == 1
        plot(forced_freq_plot, forced_phase_plot,'.r','LineWidth',1,'MarkerSize',10,'DisplayName','Phase ROM prediction')
    else
        plot(forced_freq_plot, forced_phase_plot,'.r','LineWidth',1,'MarkerSize',10,'HandleVisibility','off')
    end

end

% data
plot(freq_FRC_proj(1:n_length), 180/pi*(phase_data(1:n_length) - pi/2),'k.','LineWidth',1,'MarkerSize',10,'DisplayName','Phase data');
plot(freq_FRC_proj(n_length+1:2*n_length), 180/pi*(phase_data(n_length+1:2*n_length) - pi/2),'k.','LineWidth',1,'MarkerSize',10,'HandleVisibility','off');
plot(freq_FRC_proj(2*n_length+1:3*n_length), 180/pi*(phase_data(2*n_length+1:3*n_length) - pi/2),'k.','LineWidth',1,'MarkerSize',10,'HandleVisibility','off');
plot(freq_FRC_proj(3*n_length+1:4*n_length), 180/pi*(phase_data(3*n_length+1:4*n_length) - pi/2),'k.','LineWidth',1,'MarkerSize',10,'HandleVisibility','off');


%% Additional functions

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
ceq = det(V * inv(transpose(B) * V) * transpose(B) - eye(size(V,1)));

end







