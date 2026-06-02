function [label, val] = formatKLabel(k, ndp)
% FORMATKLABEL 
    if nargin < 2, ndp = 3; end

    if isstring(k) || ischar(k)
        label = char(k);
        label = regexprep(label, '(\.\d*?)0+$', '$1');
        label = regexprep(label, '\.$', '');
        val   = str2double(label);
    else
        kval  = round(k, ndp);
        label = compose("%." + ndp + "f", kval);
        label = regexprep(label, '(\.\d*?)0+$', '$1');
        label = regexprep(label, '\.$', '');
        label = char(label);
        val   = kval;
    end
end
