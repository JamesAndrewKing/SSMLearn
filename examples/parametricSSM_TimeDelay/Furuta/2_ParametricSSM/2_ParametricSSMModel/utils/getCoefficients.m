function c = getCoefficients(eqStruct, Ts, config)
%GETCOEFFICIENTS Interpolates the coefficients C_data for a given Ts value
%   c = GETCOEFFICIENTS(eqStruct, Ts) returns a column vector c
%   containing the coefficients interpolated based on the value of Ts.
    if Ts < min(eqStruct.Ts_vals) || Ts > max(eqStruct.Ts_vals)
        error('Ts is outside the supported interval [%g, %g].', ...
              min(eqStruct.Ts_vals), max(eqStruct.Ts_vals));
    end

    method = config.interpMethodDynamicsPP;
    
    % Fallback to linear if the method is not recognized
    if ~ismember(method, {'linear','spline'})
        disp(['Unrecognized interpolation method: ', method, ...
              '. Falling back to ''linear'' interpolation.']);
        method = 'linear';
    end

    [p, ~] = size(eqStruct.C_data);
    c = zeros(p,1);
    for i = 1:p
        coeff_row = eqStruct.C_data(i,:);
        c(i) = interp1(eqStruct.Ts_vals, coeff_row, Ts, method);
    end
end