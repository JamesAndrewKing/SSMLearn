function interpStruct = structureForInterpolation(polynomials)
    Ts_vals = [polynomials.Ts];
    eqNames = {'rho1_dot','rho2_dot'};
    for k =1:length(Ts_vals)
        for eqIdx = 1:2
            Cfull = cell2mat( arrayfun(@(p)p.coeffs{eqIdx}(:), polynomials, ...
                                       'UniformOutput', false) );   
            interpStruct(eqIdx).eqName    = eqNames{eqIdx};
            interpStruct(eqIdx).exponents = polynomials(1).exponents{eqIdx}(:,:);
            interpStruct(eqIdx).Ts_vals   = Ts_vals;
            interpStruct(eqIdx).C_data    = Cfull(:,:); 
        end
    end
end
