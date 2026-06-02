%% 2D problem 

alpha = 1;
beta = 2.7;
delta = 0.8;
mu = 0.2;
gamma = 0.15;

[F,A,V,D] = setsystem(alpha,beta,delta,mu,gamma);

V1 = V(:,1);
V2 = V(:,2);
tspan = 0:1e-3:10;

x01_vect = linspace(-1,1,4);

figure
axis equal
hold on 

xlim([-1.25 1.25]);
ylim([-1.25 1.25]);

plot([-1.25 1.25], [0 0],'k','LineWidth',1)
plot([0 0],[-1.25 1.25],'k','LineWidth',1)

set(gca,'FontSize',20,'TickLabelInterpreter', 'Latex');
xlab = xlabel('$x_1$','Interpreter','latex');
xlab.Position = [1.2512 0.0066 -1];
ylab = ylabel('$x_2$','Interpreter','latex');
ylab.Rotation = 0;
ylab.Position = [-0.0899 1.0774 -1];

for ii = 1:length(x01_vect)
    if ii == 1 || ii == length(x01_vect)
        x02_vect = linspace(-1,1,4);
    else
        x02_vect = [-1 1];
    end 
    for jj = 1:length(x02_vect)     
        x0 = [x01_vect(ii); x02_vect(jj)];
        [t,sol] = ode45(F,tspan,x0);
        t = transpose(t);
        sol = transpose(sol);
        plot(sol(1,:), sol(2,:), 'k','LineWidth',1);
        arrowh(sol(1,:),sol(2,:),'k',[],0.8*norm(x0),0);

    end
end

x_vect = -1.25:0.1:1.25;
y_vect = linspace(-1.25,1.25,length(x_vect)); 

ax = gca;

% analytic nonlinear projection and fiber 
b1 = 1;
b2 = delta/(beta-alpha);
b4 = -2*gamma/beta*(delta/(beta-alpha))^2;
b5 = 1/(2*beta-alpha)*(mu - gamma*(delta/(beta-alpha))^2*(2*delta/beta+(delta/(beta-alpha))));

P_x = @(x) b1*x(1,:) + b2*x(2,:) + b4*x(1,:).*x(2,:) + b5*x(2,:).^2;

% draw slow SSM
xvect = linspace(-1,1,100);
yvect = linspace(-1,1,100);
a1 = 0;
a2 = gamma/(beta-2*alpha);
y_ssm_xp = @(xp) a1*xp + a2*xp.^2;

% plot primary slow SSM
F_ssm = @(z) z(2) - y_ssm_xp(P_x(z));
x_curr_vect = linspace(-0.85,0.75,5e2);
y_ssm_min1_vect = [];
y_ssm_min2_vect = [];
for ii = 1:length(x_curr_vect)
    x_curr = x_curr_vect(ii);
    F_ssm_min = @(y) F_ssm([x_curr;y]);
    options = optimset('Display','off');
    y_ssm_min1 = fzero(F_ssm_min,0,options);
    y_ssm_min2 = fzero(F_ssm_min,1.5,options);

    plot(x_curr, y_ssm_min1,'.','color',[0.47,0.67,0.19],'MarkerSize',10)
    plot(x_curr, y_ssm_min2,'.','color',[0.47,0.67,0.19],'MarkerSize',10)

end

F_ssm_2 = @(z) z(2) - y_ssm_xp(P_x(z));
y_curr_vect = linspace(0.1,0.3,5e2);
x_ssm_min1_vect = [];
x_ssm_min2_vect = [];
for ii = 1:length(y_curr_vect)
    y_curr = y_curr_vect(ii);
    F_ssm_min = @(x) F_ssm_2([x;y_curr]);
    options = optimset('Display','off');
    x_ssm_min1 = fzero(F_ssm_min,1.5,options);
    plot(x_ssm_min1, y_curr,'.','color',[0.47,0.67,0.19],'MarkerSize',10)
end


% plot nonlinear fibers (analytic)

x_fb_vect = linspace(-0.75,0.75,5);
x_sample_vect = [x_fb_vect; zeros(size(x_fb_vect))];
x_sample_proj = P_x(x_sample_vect);

for jj = 1:length(x_sample_proj)
    x_fiber_vect = [];
    x_bar = x_sample_proj(jj);
    for ii = 1:length(yvect)
    
        y_curr = yvect(ii);
        F_min = @(x) P_x([x;y_curr])-x_bar;
        x_fiber = fzero(F_min,0);
        x_fiber_vect = [x_fiber_vect x_fiber];
    
    end
    
    plot(x_fiber_vect, yvect,'color',[0.93,0.69,0.13],'LineWidth',3)

end

% draw fast SSM
x_fiber_vect = [];
x_bar = 0;
for ii = 1:length(yvect)
    y_curr = yvect(ii);
    F_min = @(x) P_x([x;y_curr])-x_bar;
    x_fiber = fzero(F_min,0);
    x_fiber_vect = [x_fiber_vect x_fiber];
end
plot(x_fiber_vect, yvect,'color',[0.93,0.69,0.13],'LineWidth',3)


% generate trajectories 

x01 = [0.8; 0.2];
x02 = [-0.8; 0.2];
x03 = [0.8; 0.1];

x0_vect = [x01 x02 x03];
sol_mat = [];
t_mat = [];
xData = cell(size(x0_vect,2),2);
for ii = 1:size(x0_vect,2)
    x0 = x0_vect(:,ii);
    [t,sol] = ode45(F,tspan,x0);
    t = transpose(t);
    sol = transpose(sol);
    sol_mat = [sol_mat; sol];
    t_mat = [t_mat; t];
    xData{ii,1} = t;
    xData{ii,2} = sol;
end

% computation of the oblique projection and SSM parametrization through invariance of the foliation equation
% rearrange data
t = horzcat(xData{:,1});
X = horzcat(xData{:,2});

SSMDim = 1;

% compute the tangent space from data via SVD
mfddim = 1;
[u,s,v] = svds(X, mfddim);
V_e = max(abs(v'),[],2).'.*u*s;

% impose the constraint that V^T * V = I
V_tg = orthogonalizeGramSchmidt(V_e);

% compute the derivative of the data
iStart = 1;
for iTraj = 1:size(xData,1); iStart(iTraj+1) = iStart(iTraj)+size(xData{iTraj,1},2); end
Xdot = []; Xnew = [];
for iTraj = 1:size(xData,1)
    [xdot,xnew] = ftd(X(:,iStart(iTraj):iStart(iTraj+1)-1),t);
    Xdot = [Xdot, xdot]; Xnew = [Xnew, xnew];
end

% optimization process
B0 = V_tg;
[nB,mB] = size(B0);

R_tilde0 = -alpha;

[nR,mR] = size(R_tilde0);
ROMOrder = 2;
phi_0 = phi_mv(inv(transpose(B0)*V_tg) * transpose(B0) * Xnew, 2:ROMOrder);
N0 = zeros(size(R_tilde0,1), size(phi_0,1));
[nN,mN] = size(N0);

x0 = [B0(:); R_tilde0(:); N0(:)];
J = @(X) OblProjMat_nonlin(X,Xdot,Xnew,nB,mB,nR,mR,nN,mN,ROMOrder,V_tg);

% constraint 
nonlin_constraint = @(X) idemP(X,V_tg,nB,mB);

options = optimoptions('fmincon', 'Display', 'iter','MaxFunctionEvaluation',1e5);
[x_opt, fval] = fmincon(J, x0, [], [], [], [], [], [], nonlin_constraint, options);

xB = x_opt(1:nB*mB);
B = reshape(xB,nB,mB);
xR_tilde = x_opt(1+nB*mB:nB*mB+nR*mR);
R_tilde = reshape(xR_tilde,nR,mR);
xN = x_opt(1+nB*mB+nR*mR:end);
N = reshape(xN,nN,mN);

P_opt = V_tg * inv(transpose(B) * V_tg) * transpose(B);

[aa,bb] = idemP(x_opt,V_tg,nB,mB);

R = R_tilde;
Rmap = @(t,xi) R*xi + N*phi_mv(xi, 2:ROMOrder);

% plot data-driven fibers
p_fun = @(x) transpose(V_tg) * P_opt * x;
for jj = 1:length(x_sample_proj)
    x_fiber_vect = [];
    x_bar = x_sample_proj(jj);
    for ii = 1:length(yvect)
    
        y_curr = yvect(ii);
        F_min = @(x) p_fun([x;y_curr])-x_bar;
        x_fiber = fzero(F_min,0);
        x_fiber_vect = [x_fiber_vect x_fiber];
    
    end
    
    plot(x_fiber_vect, yvect,'--','color',[1.00,0.41,0.16],'LineWidth',2)

end

% parametrization of the SSM
Xi = transpose(V_tg) * P_opt * X;
SSMOrder = ROMOrder;
phi_M0 = phi_mv(Xi,2:SSMOrder);
M0 = zeros(size(X,1),size(phi_M0,1));
J_param = @(x) LinOblParam(x,Xnew,V_tg,P_opt,SSMOrder);
nonlin_constraint_param = @(x) orthogV_tPM(x,V_tg,P_opt);
options = optimoptions('fmincon', 'Display', 'iter','MaxFunctionEvaluation',1e5);
[M_opt, ~] = fmincon(J_param, M0, [], [], [], [], [], [], nonlin_constraint_param, options);

x_ssm = @(xi) V_tg * xi + M_opt * phi_mv(xi, 2:SSMOrder);

xi_vect = linspace(-1.3,1,1e2);
X_ssm = x_ssm(xi_vect);
plot(X_ssm(1,:),X_ssm(2,:),'--','color',[0.39,0.83,0.07],'linewidth',3)

% generate test trajectory
x0_test = [0.7; 0.4];
x0_vect_test = x0_test;
xDataTest = [];
for ii = 1:size(x0_vect_test,2)
    x0 = x0_vect_test(:,ii);
    [t,sol] = ode45(F,tspan,x0);
    t = transpose(t);
    sol = transpose(sol);

    plot(sol(1,:), sol(2,:),'color','k','LineWidth',2.5);
    plot(sol(1,1), sol(2,1),'.','color','k','LineWidth',2,'MarkerSize',20);

    xDataTest{ii,1} = t;
    xDataTest{ii,2} = sol;
end

% reduced order model 

xi0 = transpose(V_tg) * P_opt * x0_test;
[tRed,xiRed] = ode45(Rmap,tspan,xi0);
tRed = transpose(tRed);
xiRed = transpose(xiRed);
xRed = x_ssm(xiRed);


proj_test = P_opt * x0_test;
plot([x0_test(1) xRed(1)],[x0_test(2) xRed(2)],'k--','LineWidth',1)
plot(xRed(1,1),xRed(2,1),'.','color',[0.39,0.83,0.07],'LineWidth',1,'MarkerSize',20)

% Prediction of test trajectory: comparison between orthogonal and oblique projection 

[IMInfo, SSMChart, SSMFunction] = IMGeometry(xData, SSMDim, SSMOrder);
etaData = projectTrajectories(IMInfo, xData);
RDInfo = IMDynamicsFlow(etaData,'R_PolyOrd', ROMOrder);
[yRec_orth, etaRec_orth, zRec_orth] = advect(IMInfo, RDInfo, xDataTest);

% Compute errors
normedTrajDist = computeTrajectoryErrors(yRec_orth, xDataTest);
RRMSE_orth = mean(normedTrajDist)

f = figure;
f.Position = [476,252,726,620];
subplot(2,1,1)
hold on
grid on
set(gca,'FontSize',20,'TickLabelInterpreter','latex')
xlabel('$t$','Interpreter','latex')
ylabel('$x_1$','Interpreter','latex')
legend('location','SE','Interpreter','latex')

plot(t,sol(1,:),'k','LineWidth',3,'DisplayName','Test data')
plot(tRed, xRed(1,:), 'color',[0.30,0.75,0.93],'LineWidth',3,'DisplayName','SSM model (oblique projection)');
plot(yRec_orth{1,1}, yRec_orth{1,2}(1,:), 'color',[0.72,0.27,1.00],'LineWidth',3,'DisplayName','SSM model (orthogonal projection)');

subplot(2,1,2)
hold on
grid on
set(gca,'FontSize',20,'TickLabelInterpreter','latex')
xlabel('$t$','Interpreter','latex')
ylabel('$x_2$','Interpreter','latex')

plot(t,sol(2,:),'k','LineWidth',3,'DisplayName','Test data')
plot(tRed, xRed(2,:), 'color',[0.30,0.75,0.93],'LineWidth',3,'HandleVisibility','off');
plot(yRec_orth{1,1}, yRec_orth{1,2}(2,:), 'color',[0.72,0.27,1.00],'LineWidth',3,'HandleVisibility','off');

% compute the error 
yRec = cell(1,2);
yRec{1,1} = tRed;
yRec{1,2} = xRed;
data = cell(1,2);
data{1,1} = t;
data{1,2} = sol;
normedTrajDist = computeTrajectoryErrors(yRec, data);
RRMSE_obl = mean(normedTrajDist)


%% Additional functions

function [dXdt, Xtrunc] = ftd(X, t) % finite time difference
    ind = 5:size(X,2)-4; Xtrunc = X(:,ind);
    dX = 4/5*(X(:,ind+1)-X(:,ind-1)) - 1/5*(X(:,ind+2)-X(:,ind-2)) + ...
        4/105*(X(:,ind+3)-X(:,ind-3)) - 1/280*(X(:,ind+4)-X(:,ind-4));
    dXdt = dX./(t(ind+1)-t(ind));
end

function exps=exponents(d,k)
    B = repmat({0:max(k)},1,d);
    A = combvec(B{:}).';
    exps = A(ismember(sum(A,2), k),:);
    [~,ind] = sort(sum(exps,2));
    exps = exps(ind,:);
end

function u = phi_mv(xi, r) % return monomials
    x = reshape(xi, 1, size(xi, 1), []);
    exps = exponents(size(xi, 1),r);
    u = reshape(prod(x.^exps, 2), size(exps, 1), []);
end

function J = OblProjMat_nonlin(x,Xdot,Xnew,nB,mB,nR,mR,nN,mN,ROMOrder,V)

xB = x(1:nB*mB);
B = reshape(xB,nB,mB);

xR_tilde = x(1+nB*mB:nB*mB+nR*mR);
R_tilde = reshape(xR_tilde,nR,mR);

xN = x(1+nB*mB+nR*mR:end);
N = reshape(xN,nN,mN);

LHS = transpose(B) * Xdot;
RHS = (transpose(B)*V) * R_tilde * inv(transpose(B)*V) * transpose(B) * Xnew + (transpose(B)*V) * N * phi_mv(inv(transpose(B)*V) * transpose(B) * Xnew, 2:ROMOrder);


Err = LHS - RHS;

J = norm(Err);

end

function [c,ceq] = idemP(x,V,nB,mB)
xB = x(1:nB*mB);
B = reshape(xB,nB,mB);

c = [];
ceq = det(V * inv(transpose(B) * V) * transpose(B) - eye(size(V,1)));

end

function [c,ceq] = orthogV_tPM(x,V,P)
c = [];
ceq = transpose(V) * P * x;
end

function J = LinOblParam(x,X,V,P,SSMOrder)

Xi = transpose(V)*P*X;
Err = X - V*Xi - x*phi_mv(Xi,2:SSMOrder);
J = norm(Err);

end

function [F,A,V,d] = setsystem(lambda,mu,delta,eta,gamma)

A = [-lambda delta; 0 -mu];
F = @(t,x) A*x + [eta*x(2).^2;gamma*x(1).^2];
[V,D] = eig(A);
d = diag(D);
[d, ind] = sort(d,'descend');
V = V(:, ind);

end

function handle = arrowh(x,y,clr,ArSize,Where,l)
%-- errors
if nargin < 2
	error('Please give enough coordinates !');
end
if (length(x) < 2) || (length(y) < 2),
	error('X and Y vectors must each have "length" >= 2 !');
end
if (x(1) == x(2)) && (y(1) == y(2)),
	error('Points superimposed - cannot determine direction !');
end
if nargin <= 2
	clr = 'b';
end
if nargin <= 3
	ArSize = [100,100];
end
handle = [];
%-- check if variables left empty, deal width ArSize and Color
if isempty(clr)
	clr = 'b'; nonsolid = false;
elseif ischar(clr)
	if strncmp('e',clr,1) % for non-solid arrow heads
		nonsolid = true; clr = clr(2);
	else
		nonsolid = false;
	end
elseif isvector(clr)
	if length(clr) == 4 && clr(1) == 0  % for non-solid arrow heads
		nonsolid = true;
		clr = clr(2:end);
	else
		nonsolid = false;
	end
else
	error('COLOR argument of wrong type (must be either char or vector)');
end
if nargin <= 4
	if (length(x) == length(y)) && (length(x) == 2)
		Where = 100;
	else
		Where = 50;
	end
end
if isempty(ArSize)
	ArSize = [100,100];
end
if length(ArSize) == 2
	ArWidth = 0.75*ArSize(2)/100; % .75 to make arrows it a bit slimmer
else
	ArWidth = 0.75;
end
ArSize = ArSize(1);
%-- determine and remember the hold status, toggle if necessary
if ishold,
	WasHold = 1;
else
	WasHold = 0;
	hold on;
end
%-- start for-loop in case several arrows are wanted
for Loop = 1:length(Where),
	%-- if vectors "longer" then 2 are given we're dealing with time series
	if (length(x) == length(y)) && (length(x) > 2),
		j = floor(length(x)*Where(Loop)/100); %-- determine that location
		if j >= length(x), j = length(x) - 1; end
		if j == 0, j = 1; end
		x1 = x(j); x2 = x(j+1); y1 = y(j); y2 = y(j+1);
	else %-- just two points given - take those
		x1 = x(1); x2 = (1-Where/100)*x(1)+Where/100*x(2);
		y1 = y(1); y2 = (1-Where/100)*y(1)+Where/100*y(2);
	end
	%-- get axe ranges and their norm
	OriginalAxis = axis;
	Xextend = abs(OriginalAxis(2)-OriginalAxis(1));
	Yextend = abs(OriginalAxis(4)-OriginalAxis(3));
	%-- determine angle for the rotation of the triangle
	if x2 == x1, %-- line vertical, no need to calculate slope
		if y2 > y1,
			p = pi/2;
		else
			p= -pi/2;
		end
	else %-- line not vertical, go ahead and calculate slope
		%-- using normed differences (looks better like that)
		m = ( (y2 - y1)/Yextend ) / ( (x2 - x1)/Xextend );
		if x2 > x1, %-- now calculate the resulting angle
			p = atan(m);
		else
			p = atan(m) + pi;
		end
	end
	%-- the arrow is made of a transformed "template triangle".
	%-- it will be created, rotated, moved, resized and shifted.
	%-- the template triangle (it points "east", centered in (0,0)):
	xt = [1	-sin(pi/6)	-sin(pi/6)];
	yt = ArWidth*[0	 cos(pi/6)	-cos(pi/6)];
	%-- rotate it by the angle determined above:
	xd = []; yd = [];
	for i=1:3
		xd(i) = cos(p)*xt(i) - sin(p)*yt(i);
		yd(i) = sin(p)*xt(i) + cos(p)*yt(i);
	end
	%-- move the triangle so that its "head" lays in (0,0):
	xd = xd - cos(p);
	yd = yd - sin(p);
	%-- stretch/deform the triangle to look good on the current axes:
    if l == 0
	    xd = xd*Xextend*ArSize/6000;
	    yd = yd*Yextend*ArSize/6000;
    else
        xd = xd*Xextend*ArSize/5000;
	    yd = yd*Yextend*ArSize/5000;
    end
	%-- move the triangle to the location where it's needed
	xd = xd + x2;
	yd = yd + y2;
	%-- draw the actual triangle
	handle(Loop) = patch(xd,yd,clr,'EdgeColor',clr,'HandleVisibility','off');
	if nonsolid, set(handle(Loop),'facecolor','none'); end
end % Loops
%-- restore original axe ranges and hold status
axis(OriginalAxis);
if ~WasHold,
	hold off
end
%-- work done. good bye.
end
