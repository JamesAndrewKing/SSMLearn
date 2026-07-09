function [IMInfo,V] = IMGeometryParaConFourier(yData,etaData,paramData,varargin)
%IMGeometryParaConFourier Parametric Fourier-Taylor geometry fit.
%
%   [IMInfo,V] = IMGeometryParaConFourier(yData,etaData,paramData,...)
%
%   Monomial/Taylor-basis counterpart of
%   IMGeometryParaConFourierLegendre. The call signature is identical; only
%   the regression basis changes. This is useful for direct A/B replacement
%   tests:
%
%      IMGeometryParaConFourier(...)
%      IMGeometryParaConFourierLegendre(...)
%
%   If origin_fixed is true, V(0,theta,delta)=0 is imposed by linear
%   equality constraints on the Fourier-Taylor coefficients.

[IMInfo,V] = IMGeometryParaConFourierLegendre(yData,etaData,paramData, ...
    varargin{:},'basis','monomial');
IMInfo.parametrization.basis = 'monomial';
end
