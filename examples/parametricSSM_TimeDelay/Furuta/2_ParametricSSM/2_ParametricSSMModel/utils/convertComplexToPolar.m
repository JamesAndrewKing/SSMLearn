function polarData = convertComplexToPolar(zzData)
% convertToPolar Converts complex data in Cartesian form to polar form
% using the transformation z_j = rho_j e^(iTheta_j)

    [nRows, nCols] = size(zzData);
    if nCols ~= 2
        error('Input must be an N×2 cell array.');
    end

    polarData = cell(nRows, 2);

    for i = 1:nRows
        time = zzData{i,1};
        cartData = zzData{i,2};
        [numRows, numPoints] = size(cartData);
        if mod(numRows, 2) ~= 0
            error('Cartesian data must have an even number of rows (real/imag pairs).');
        end
        numSignals = numRows / 2;

        polarMatrix = zeros(numRows, numPoints);
        for k = 1:numSignals
            z = cartData(k, :);
            rho = abs(z);
            theta = angle(z); % in radians

            polarMatrix(2*k-1, :) = rho;
            polarMatrix(2*k, :) = theta;
        end

        polarData{i,1} = time;
        polarData{i,2} = polarMatrix;
    end
end
