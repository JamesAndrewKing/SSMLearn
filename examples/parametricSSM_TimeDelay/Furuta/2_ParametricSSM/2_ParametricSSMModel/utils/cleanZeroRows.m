function xDataClean = cleanZeroRows(xData)
%CLEanzEROROWS  Remove any row of a cell‐array where every cell is a numeric all‐zeros array
%   xDataClean = cleanZeroRows(xData)
%     xData        — N×M cell array
%     xDataClean   — K×M cell array with rows removed where all entries are zero

    nRows = size(xData,1);
    keep  = false(nRows,1);

    for i = 1:nRows
        row = xData(i,:);
        % A cell is “zero” if it’s numeric and all its elements are exactly 0
        isZeroCell = cellfun(@(c) isnumeric(c) && all(c(:)==0), row);
        % keep row if at least one cell is nonzero
        keep(i) = ~all(isZeroCell);
    end

    xDataClean = xData(keep,:);
end
