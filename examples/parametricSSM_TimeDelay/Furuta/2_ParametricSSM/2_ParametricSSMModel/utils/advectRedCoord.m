function [yRec, etaRec] = advectRedCoord(IMInfo, RDInfo, yData)
% [yRec, etaRec, zRec, zData] = advectInReducedSpace(IMInfo, RDInfo, yData)
% Advect the initial conditions in yData by the dynamics in RDInfo on the
% manifold IMInfo in reduced coordinates.
%   The reconstructed trajectories can be compared to the original ones
% with computeTrajectoryErrors to evaluate the model precision
%   
% INPUT
% IMInfo     struct         Manifold.
% RDInfo     struct         Reduced dynamics in map or flow form.
% yData      {nTraj x 2}    Trajectories. Only the initial condition
%                           and times are used for reconstruction.
% OUTPUT
% yRec       {nTraj x 2}    Advected trajectories in observable space.
% etaRec     {nTraj x 2}    Advected trajectories on the manifold.
% zRec       {nTraj x 2}    Advected trajectories in the conjugate
%                           dynamics.
% zData      {nTraj x 2}    Original yData transformed to conjugate
%                           dynamics space.

R = RDInfo.reducedDynamics.map;
etaData = projectTrajectories(IMInfo, yData);
if strcmp(RDInfo.dynamicsType, 'flow')
    etaRec = integrateFlows(R, etaData);
elseif strcmp(RDInfo.dynamicsType, 'map')
    etaRec = iterateMaps(R, etaData);
end

yRec = liftTrajectories(IMInfo, etaRec);