% ========================================================================
% Script: TrajGenerator_TimeDelayAndSpatialDiscretization.m
% Purpose:
%   Simulate trajectories of the Furuta pendulum under delayed and spatially 
%   discretized PD controller  from random initial conditions within a 
%   small-radius hypersphere in the phase space around the fixed point to 
%   be stabilized (i.e. origin).
%
% Outputs:
%   Saves xData -> ./Data_Ts<Ts>_FurutaIntU/xDataKp_<Kp>_Kd_<Kd>.mat
% ========================================================================

clearvars; close all; clc

% --- Parallel setup ---
numCores = feature('numcores');
if isempty(gcp('nocreate')), parpool('local', numCores); end

% --- Parameters ---
T_final = 100;         % simulation time [s]
Ts      =  0.025;     % sampling interval [s]
intraTs = 0.001;      % internal integrator step [s]
r       = deg2rad(5); % IC radius [rad]

param.m    = 0.191;
param.Jp   = 0.00573;
param.Ja   = 0.00134;
param.l    = 0.15;
param.r    = 0.094;
param.g    = 9.81;
param.N    = 1.05;
param.K    = 1.12706;
param.b1   = 0.039;
param.b2   = 1.148 - param.K;
param.D2   = 1.5;     

fixed_Kp = 15.5
fixed_Kd = 5.45

%% --- Simulate trajectories ---
nIC = numCores;
results = cell(nIC,1);

parfor idx = 1:nIC
    [t_out, theta_out, thetadot_out, phidot_out] = ...
        simulateFurutaSingle_output(T_final, intraTs, Ts, param, fixed_Kp, fixed_Kd, r);
    results{idx} = struct('t', t_out, 'theta', theta_out, ...
                          'thetadot', thetadot_out, 'phidot', phidot_out);
end

%% --- Plots and save ---
nRows = ceil(sqrt(nIC));
nCols = ceil(nIC / nRows);
titleBase = sprintf('Furuta under time delay and spatial discretization: Kp=%g [V], Kd=%g [Vs], Ts =%g [ms] ICs in 4D ball (r=%.2f [deg])', fixed_Kp, fixed_Kd, Ts*1000, r*360/2/pi);

fields = {'theta',   '$\theta$ [deg]'; ...
          'thetadot','$\dot{\theta}$ [deg/s]'; ...
          'phidot',  '$\dot{\phi}$ [deg/s]'};

for f = 1:size(fields,1)
    fig = myfigure(); clf(fig);
    for k = 1:nIC
        subplot(nRows, nCols, k);
        S = results{k};
        if isfield(S, 't') && isfield(S, fields{f,1}) && ~isempty(S.t) && ~isempty(S.(fields{f,1}))
            plot(S.t, S.(fields{f,1}), 'LineWidth', 1.2); grid on;
            xlabel('Time [s]', 'Interpreter','latex','FontSize',16);
            ylabel(fields{f,2}, 'Interpreter','latex','FontSize',16);
            title(sprintf('Simulation %d', k), 'Interpreter','latex','FontSize',18);
        else
            axis off; text(0.5,0.5,sprintf('Missing %s (sim %d)', fields{f,1}, k), ...
                'HorizontalAlignment','center','Interpreter','latex');
        end
    end
    sgtitle(titleBase, 'Interpreter','latex','FontSize',20);
end

% --- Pack xData and save (rows: theta; thetadot; phidot) ---
xData = {};
for i = 1:nIC
    S = results{i};
    if isstruct(S) && ~isempty(S.t) && ~isempty(S.theta)
        xData(end+1,:) = {S.t(:), [S.theta(:)'; S.thetadot(:)'; S.phidot(:)']}; 
    end
end

folder = sprintf('Ts%gms_FurutaTimeDelaySpatialDiscretization', Ts*1000);
if ~exist(folder, 'dir'), mkdir(folder); end
filename = sprintf('xDataKp_%g_Kd_%g.mat', fixed_Kp, fixed_Kd);
save(fullfile(folder, filename), 'xData');

%% Functions

% -------------------------------------------------------------------------
% Function: simulateFurutaSingle_output
% Executes a trajectory simulation and returns the time vector and theta
% -------------------------------------------------------------------------
function [t_out, theta_out, thetadot_out, phidot_out] = simulateFurutaSingle_output(T_final, intraTs, Ts, param, Kp, Kd, r)
    N_steps = round(T_final / Ts);
    t_history = (0:N_steps)' * Ts;
    
    X_history = zeros(N_steps+1, 4);
    ode_options = odeset('RelTol',1e-6, 'AbsTol',1e-7, 'MaxStep',1e-3);
    t_fine = [];
    X_fine = [];
    
    theta0  = r * 2 * (rand - 0.5);
    phi0    = r * 2 * (rand - 0.5);
    dtheta0 = r * 2 * (rand - 0.5);
    dphi0   = r * 2 * (rand - 0.5);
    X_history(1,:) = [theta0, dtheta0, phi0, dphi0];
    theta_history = theta0;
    phi_history   = phi0;
    x_current = X_history(1,:)';
  
    for j = 1:N_steps
        time_fine_step = t_history(j):intraTs:t_history(j+1);
        if time_fine_step(end) < t_history(j+1)
            time_fine_step(end+1) = t_history(j+1);
        end
        
        if j < 3
            U_command = 0;
        else
            theta_jm1 = theta_history(end-1);
            theta_jm2 = theta_history(end-2);
            phi_jm1   = phi_history(end-1);
            phi_jm2   = phi_history(end-2);
            dtheta_est = (theta_jm1 - theta_jm2) / Ts;
            dphi_est   = (phi_jm1 - phi_jm2) / Ts;
            U_command = floor(-Kp * theta_jm1 - Kd * dtheta_est + param.D2 * dphi_est);  %integer part
        end
        
        [t_sol, x_sol] = ode23t(@(t,x) furutaDynamics(t,x,param,U_command), ...
                        time_fine_step, x_current, ode_options);

        if j == 1
            t_fine = [t_fine; t_sol];
            X_fine = [X_fine; x_sol];
        else
            t_fine = [t_fine; t_sol(2:end)];
            X_fine = [X_fine; x_sol(2:end,:)];
        end
        
        x_current = x_sol(end,:)';
        X_history(j+1,:) = x_current';
        theta_history = [theta_history, x_current(1)];
        phi_history   = [phi_history,   x_current(3)];
    end
    
    % Return the time vector and trajectory (in degrees)
    t_out = t_fine;
    theta_out = rad2deg(X_fine(:,1)); 
    thetadot_out = rad2deg(X_fine(:,2)); 
    phidot_out = rad2deg(X_fine(:,4)); 
end

% -------------------------------------------------------------------------
% Function: furutaDynamics
% Dynamics equations of the Furuta pendulum
% -------------------------------------------------------------------------
function dxdt = furutaDynamics(~, x, param, U_cmd)

theta  = x(1);
    dtheta = x(2);
    % phi    = x(3);   
    dphi   = x(4);
    
    cos_th = cos(theta);
    sin_th = sin(theta);
    sin2_th = sin_th^2;  
    
    m  = param.m;
    Jp = param.Jp;
    Ja = param.Ja;
    l  = param.l;
    r  = param.r;
    g  = param.g;
    b1 = param.b1;
    b2 = param.b2;
    N  = param.N;
    K  = param.K;
    
    mrl = m * r * l; 
    Jp_sc = Jp * sin_th * cos_th;  
    
    M = N * U_cmd - K * dphi;
    
     a = Jp;
    b = -mrl * cos_th;
    c = b; 
    d = Ja + Jp * sin2_th;
    
    RHS1 = Jp_sc * dphi^2 - b1 * dtheta + m * g * l * sin_th;
    RHS2 = M - (2 * Jp_sc * dtheta * dphi + b2 * dphi - mrl * dtheta^2);
    
    detA = a * d - b * c;
    dd1 = (RHS1 * d - b * RHS2) / detA;
    dd2 = (a * RHS2 - c * RHS1) / detA;

    dxdt = [ dtheta;
             dd1;
             dphi;
             dd2 ];
end