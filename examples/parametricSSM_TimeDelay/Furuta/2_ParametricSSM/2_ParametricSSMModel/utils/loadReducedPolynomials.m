function polynomials = loadReducedPolynomials(config)
%LOADREDUCEDPOLYNOMIALS  Load or compute reduced‐order polynomial coefficients in polar form
%   polynomials = loadReducedPolynomials(config)
%     config.mu_values     — sampling-time Ts (i.e. the parameter) vector
%   Output: for each Ts
%     .coeffs    = {c_r1, c_r2, c_th1, c_th2}
%     .exponents = {e_r1, e_r2, e_th1, e_th2}

    Ts_vals = config.mu_values;
    nTs     = numel(Ts_vals);
    polynomials = struct('Ts',        cell(1,nTs), ...
                         'coeffs',    cell(1,nTs), ...
                         'exponents', cell(1,nTs));
    for i = 1:nTs
        Ts_val = Ts_vals(i);
        subFolder = sprintf(config.folderPattern, Ts_val);
        matFile   = fullfile('IMInfoRDInfoForTraining', subFolder, 'RDInfo.mat');
        fprintf('Loading from: %s', matFile);

        % Load RDInfo
        data   = load(matFile, 'RDInfo');
        RDInfo = data.RDInfo;

        % Extract coefficients and exponents
        C = RDInfo.conjugateDynamics.coefficients;  
        E = RDInfo.conjugateDynamics.exponents;     
        nMon = size(E,1);

        c_r1 = []; e_r1 = [];
        c_th1 = []; e_th1 = [];
        c_r2 = []; e_r2 = [];
        c_th2 = []; e_th2 = [];

        for j = 1:nMon
            p = E(j,1); 
            q = E(j,2);
            r = E(j,3); 
            s = E(j,4);
            % Radial z1: dot(rho1)/rho1
            a1 = C(1,j);
            expo1 = [p+r, q+s, 0, 0];
            if abs(real(a1))>eps
                c_r1(end+1)  = real(a1);
                e_r1(end+1,:) = expo1;
            end
            % Angular z1: dot(theta1)
            if abs(imag(a1))>eps
                c_th1(end+1)  = imag(a1);
                e_th1(end+1,:) = expo1-[1,0,0,0];
            end

            % Radial z2: dot(rho2)/rho2
            a2 = C(2,j);
            expo2 = [p+r, q+s, 0, 0];
            if abs(real(a2))>eps
                c_r2(end+1)  = real(a2);
                e_r2(end+1,:) = expo2;
            end
            % Angular z2: dot(theta2)
            if abs(imag(a2))>eps
                c_th2(end+1)  = imag(a2);
                e_th2(end+1,:) = expo2-[0,1,0,0];
            end
        end

        polynomials(i).Ts        = Ts_val;
        polynomials(i).coeffs    = {c_r1, c_r2, c_th1, c_th2};
        polynomials(i).exponents = {e_r1, e_r2, e_th1, e_th2};
    end
end
