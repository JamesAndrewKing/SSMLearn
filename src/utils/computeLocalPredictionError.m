function [predictionError, prediction] = computeLocalPredictionError( ...
    trainingStates, trainingTargets, testStates, testTargets, nNeighbors)
%COMPUTELOCALPREDICTIONERROR Held-out local-constant prediction error.
%   [error,prediction] = computeLocalPredictionError( ...
%       trainingStates,trainingTargets,testStates,testTargets,nNeighbors)
%   predicts each test target by averaging the targets of the nearest
%   training states. States and targets are arranged column-wise.
%
%   The returned error is
%
%       norm(testTargets-prediction,'fro')
%       ---------------------------------- .
%       norm(testTargets-mean(testTargets,2),'fro')
%
%   States are standardized using the training mean and standard deviation
%   before the neighbor search. The default number of neighbors is 20.
%   This is the local-constant prediction criterion of Ragwitz and Kantz:
%   Phys. Rev. E 65, 056201 (2002), doi:10.1103/PhysRevE.65.056201.

if nargin < 5
    nNeighbors = 20;
end

validateattributes(trainingStates,{'numeric'},{'2d','nonempty','finite'});
validateattributes(trainingTargets,{'numeric'},{'2d','nonempty','finite'});
validateattributes(testStates,{'numeric'},{'2d','nonempty','finite'});
validateattributes(testTargets,{'numeric'},{'2d','nonempty','finite'});
validateattributes(nNeighbors,{'numeric'}, ...
    {'scalar','integer','positive','<=',size(trainingStates,2)});

if size(trainingStates,1) ~= size(testStates,1)
    error('computeLocalPredictionError:StateDimension', ...
        'Training and test states must have the same dimension.');
end
if size(trainingTargets,1) ~= size(testTargets,1)
    error('computeLocalPredictionError:TargetDimension', ...
        'Training and test targets must have the same dimension.');
end
if size(trainingStates,2) ~= size(trainingTargets,2) || ...
        size(testStates,2) ~= size(testTargets,2)
    error('computeLocalPredictionError:SampleCount', ...
        'Each state must have one corresponding target.');
end

stateMean = mean(trainingStates,2);
stateScale = std(trainingStates,0,2);
stateScale(stateScale == 0) = 1;
zTraining = (trainingStates-stateMean)./stateScale;
zTest = (testStates-stateMean)./stateScale;

neighbors = knnsearch(zTraining.',zTest.','K',nNeighbors);
nTargets = size(trainingTargets,1);
nTest = size(testTargets,2);
neighborTargets = reshape(trainingTargets(:,neighbors.'), ...
    nTargets,nNeighbors,nTest);
prediction = reshape(mean(neighborTargets,2),nTargets,nTest);

reference = testTargets-mean(testTargets,2);
referenceNorm = norm(reference,'fro');
if referenceNorm == 0
    error('computeLocalPredictionError:ConstantTarget', ...
        'Normalized prediction error is undefined for constant test targets.');
end
predictionError = norm(testTargets-prediction,'fro') ...
    /referenceNorm;
end
