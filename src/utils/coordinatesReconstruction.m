function xData = coordinatesReconstruction(yData,nCoordinates,embeddingOptions)
%COORDINATESRECONSTRUCTION Reconstruct signals from delay coordinates.
%   xData = coordinatesReconstruction(yData,nCoordinates,embeddingOptions)
%   reverses coordinatesEmbedding by averaging all delay-coordinate
%   estimates of each sample. At the ends, it uses whichever estimates are
%   available. yData may be a delay-coordinate matrix or {time,state} data.

if nargin < 3
    embeddingOptions = struct('ShiftSteps',1,'TimestampMode','first');
end
shift = embeddingOptions.ShiftSteps;

if isnumeric(yData)
    xData = reconstruct(yData,nCoordinates,shift);
    return
end

xData = cell(size(yData));
for iTrajectory = 1:size(yData,1)
    Y = yData{iTrajectory,2};
    xData{iTrajectory,2} = reconstruct(Y,nCoordinates,shift);

    dt = median(diff(yData{iTrajectory,1}));
    window = size(xData{iTrajectory,2},2)-size(Y,2);
    firstTime = yData{iTrajectory,1}(1);
    if strcmpi(embeddingOptions.TimestampMode,'last')
        firstTime = firstTime-window*dt;
    end
    xData{iTrajectory,1} = firstTime+(0:size(xData{iTrajectory,2},2)-1)*dt;
end
end

function signal = reconstruct(Y,nCoordinates,shift)
nDelays = size(Y,1)/nCoordinates;
assert(nDelays == round(nDelays), ...
    'The embedded dimension must be divisible by nCoordinates.');

nSamples = size(Y,2)+(nDelays-1)*shift;
signal = zeros(nCoordinates,nSamples,'like',Y);
count = zeros(1,nSamples);

for iDelay = 0:nDelays-1
    rows = (1:nCoordinates)+iDelay*nCoordinates;
    columns = (1:size(Y,2))+iDelay*shift;
    signal(:,columns) = signal(:,columns)+Y(rows,:);
    count(columns) = count(columns)+1;
end
signal = signal./count;
end
