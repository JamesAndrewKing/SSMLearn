function [data_projected, P_min,V_trunc_slow,data_non_projected,data_non_projected_linear,index_time_linear, B_min, lambda_trunc] = oblique_projection(data,lim,SSMDim, overEmbed, ShiftStep,norm,flag_end)
% function that computes the linear oblique projection, based on the linear
% part of the data. The whole set of data in input is observed. 

    % identification of the linear regime of the data from the backbone
    % curves
    index_linear_traj = zeros(size(data,1),1);
    for jj = 1:size(data,1)
        t = data{jj,1};
        x = data{jj,2};

        index_linear_start_vect = zeros(size(x,1),1);
        for ii = 1:size(x,1)
            [amp,freq,~,time] = PFFk(t,x(ii,:),1); 
            signal_freq = freq(~isnan(freq));
            signal_amp = amp(~isnan(freq));
            signal_time = time(~isnan(freq));
            [index_linear_start_curr, ~] = fun_linear_regime(signal_freq,lim);
    
            index_linear_start_vect(ii) = index_linear_start_curr;
        end    
        index_linear_start_curr = round(mean(index_linear_start_vect));
        signal_time_linear = signal_time(index_linear_start_curr:end);
        [~,index_time_linear_curr] = min(abs(t-signal_time_linear(1)));

        index_linear_traj(jj) = round(mean(index_time_linear_curr));
    end
    index_time_linear = round(mean(index_linear_traj));

    time_mat = cell2mat(data(:,1));
    data_mat = cell2mat(data(:,2));
    
    % truncation of the final part of the data
    if flag_end == 0
        index_end = length(time_mat(1,:));
    else
        index_end = flag_end * index_time_linear;
    end

    time_mat_linear = time_mat(:,index_time_linear:index_end);   
    data_mat_linear = data_mat(:,index_time_linear:index_end);
    
    data_linear = cell(size(data));
    m = size(time_mat_linear,1);
    p = size(data{1,2},1);
    mp_times = ones(1,m);
    data_linear(:,1) = mat2cell(time_mat_linear,mp_times,size(time_mat_linear,2));
    mp = p * ones(1,m);
    data_linear(:,2) = mat2cell(data_mat_linear,mp,size(time_mat_linear,2));

    % computation of the linear oblique projection via minimization of the
    % oscillations of the backbone curve 
   
    [yData_delay, ~]=coordinatesEmbedding(data_linear,SSMDim,'OverEmbedding',overEmbed,'ShiftSteps',ShiftStep);

    
    N = size(yData_delay{1,2},1);

    % compute the tangent space 
    dt = mean(mean(diff(time_mat_linear,[],2)));
    X = [];
    Y = [];
    for ii = 1:1 %size(yData_delay,1)
        X = [X [yData_delay{ii,2}(:,1:end-1)]]; 
        Y = [Y [yData_delay{ii,2}(:,2:end)]];
    end
    [U,S,V] = svds(X,4);
    U = U(:,1:2*SSMDim);
    S = S(1:2*SSMDim,1:2*SSMDim);
    V = V(:,1:2*SSMDim);
    S_tilde = U.'*Y*V*pinv(S);
    [E,~,Lambda] = eigSorted(S_tilde);
    lambda_A_from_S_tilde = 1/dt*log(Lambda);  
    [~,index_sort] = sort(abs(real(lambda_A_from_S_tilde)));
    lambda_trunc = lambda_A_from_S_tilde(index_sort);
    E_sorted = E(:,index_sort);
    V_trunc = U * E_sorted;
    V_trunc_slow = [real(V_trunc(:,1)) imag(V_trunc(:,1))];
    V_trunc_fast = [real(V_trunc(:,3)) imag(V_trunc(:,3))];
    
    if norm == 1 % normal projection 
        P_min = eye(N);
    else % linear oblique projection 
        % minimization of the oscillations 
        var0 = compute_var(yData_delay);
        omega_0 = imag(lambda_trunc(1));
        J = @(B) minimize_oscillations_bc_variance(V_trunc_slow,B,yData_delay,var0);
        options = optimoptions('fminunc','Display','iter');
        B_0 = V_trunc_slow;
        [B_min,~] = fminunc(J,B_0,options);
    
        % linear oblique projection 
        P_min = V_trunc_slow*((B_min.'*V_trunc_slow)\B_min.');
    end

    [data_non_projected, ~]= coordinatesEmbedding(data,SSMDim,'OverEmbedding',overEmbed,'ShiftSteps',ShiftStep);
    data_projected = data_non_projected;
    for ii = 1:size(data,1)
        data_projected{ii,2} = P_min * data_non_projected{ii,2};
    end
    data_non_projected_linear = yData_delay;


end

%% additional functions 

function [index_linear_start, lim_value] = fun_linear_regime(signal,lim)
    lim_value = lim;
    jj = 0;
    mean_vect = zeros(1,length(signal));
    for ii = 1:length(signal)-1
        mean_curr = mean(signal(1+jj:end));
        mean_vect(ii) = mean_curr;
        jj = jj + 1;
    end
    mean_vect(end) = signal(end);
    vel_mean_vect = diff(mean_vect);

    vel_abs_mean_vect = abs(vel_mean_vect);
        [~,index_linear] = find(vel_abs_mean_vect<lim_value);

    while isempty(find(vel_abs_mean_vect<lim_value))
        lim_value = exp(log(lim_value)+1);
    end
    [~,index_linear] = find(vel_abs_mean_vect<lim_value);
    index_linear_start = index_linear(1);
end

function [var0, var0_vect] = compute_var(data)

var0_vect = [];
lim_pff = 1e-4;
for ii = 1:size(data{1,2},1)
    kmean = 1;
    [amp,freq,~,~] = PFFk(data{1,1},data{1,2}(ii,:),kmean);
    amp = amp(~isnan(freq));
    signal = freq(~isnan(freq));
    amp = amp(amp>lim_pff);
    signal = signal(amp>lim_pff);
    average = mean(signal);
    var0_curr = var(signal - average);
    var0_vect = [var0_vect var0_curr];
end
var0 = max(var0_vect);

end

function J = minimize_oscillations_bc_variance(V_slow,B,data,var0)
    J_vect = zeros(size(data,1),1);
    for ii = size(data,1)
        sol = data{ii,2};
        P = V_slow*((B.'*V_slow)\B.');
        z = P*sol;
        data_proj = cell(1,2);
        data_proj{1,1} = data{ii,1};
        data_proj{1,2} = z;
        [~, var_vect] = compute_var(data_proj);
        J_vect(ii) = sum(var_vect(~isnan(var_vect)))/var0;
    end
    J = mean(J_vect);
end
