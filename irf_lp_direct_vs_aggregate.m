%%  IRF via Local Projections: Direct Shock vs Aggregate Shock
%%  -------------------------------------------------------------------
%%  Compares two LP estimators:
%%    (1) Direct LP   – uses diagonal Omega_direct * eps_t as regressors
%%    (2) Aggregate LP – uses full Omega_true * eps_t as regressors,
%%                       then maps coefficients back to structural shocks
%%  Both are plotted against the true population IRF.
%%  Outcomes: Unemployment, Inflation
%%  Shocks:   FP, MP, TFP, Oil, Crisis (5 total)
%%  -------------------------------------------------------------------
clear; clc; rng(42);

%% ── Parameters ──────────────────────────────────────────────────────
T   = 600;   % sample size
K   = 4;     % number of regular structural shocks (Crisis handled separately)
N   = 2;     % number of outcomes (Unemployment, Inflation)
H   = 20;    % max horizon for IRFs
p   = 4;     % lag order used as LP controls
rho = 0.70;  % AR(1) persistence in the DGP

shock_names   = {'FP','MP','TFP','Oil','Crisis'};
outcome_names = {'Unemployment','Inflation'};

%% ── True structural parameters ───────────────────────────────────────
%  Omega_true (5×5): weights that combine structural shocks into
%  aggregate components.  Each row defines one aggregate component.
Omega_true = [0.80 0.20 0.10 0.30 0.00;
              0.10 0.90 0.20 0.10 0.30;
              0.20 0.10 0.70 0.20 0.10;
              0.30 0.20 0.10 0.80 0.20;
              0.00 0.10 0.20 0.10 0.90];  % 5×5

%  A0_true (2×5): contemporaneous responses of outcomes to aggregate
%  components.
A0_true = [ 0.50 -0.30  0.40 -0.20 -0.50;   % Unemployment
           -0.30  0.20 -0.20  0.30  0.40];   % Inflation

%  True IRF at horizon h: IRF(h) = rho^h * A0_true * Omega_true  (2×5)
true_irf = zeros(N, K+1, H+1);
for h = 0:H
    true_irf(:,:,h+1) = (rho^h) * A0_true * Omega_true;
end

%% ── DGP: simulate structural shocks and outcomes ─────────────────────
crises_shock        = zeros(T,1);
crises_shock(end,1) = 1;                      % one-time crisis at t = T

eps  = [randn(T, K), crises_shock];           % T×(K+1)  structural shocks
aggr = eps * Omega_true';                      % T×(K+1)  aggregate shocks

rng(42);   % reset seed so outcomes are reproducible

%  Y_t = rho*Y_{t-1} + (A0_true * aggr_t) + noise
%  This DGP is consistent with true_irf(h) = rho^h * A0_true * Omega_true
unemp     = zeros(T,1);
infl_rate = zeros(T,1);
unemp(1)     = 6;   % starting unemployment rate (%)
infl_rate(1) = 2;   % starting inflation rate (%)

for t = 2:T
    shock_effect = A0_true * aggr(t,:)';          % 2×1 contemporaneous impact
    unemp(t)     = 0.5 + rho*unemp(t-1)     + shock_effect(1) + 0.5*randn();
    infl_rate(t) = 0.5 + rho*infl_rate(t-1) + shock_effect(2) + 0.4*randn();
end

Y = [unemp, infl_rate];  % T×N

%% ── Diagonal (direct) mixing matrix ─────────────────────────────────
%  Omega_direct: same diagonal as Omega_true, off-diagonal set to zero.
%  This corresponds to the "direct shock" identification in the image:
%    agg_eps_t = Omega_direct * eps_t
Omega_direct = diag(diag(Omega_true));  % 5×5 diagonal matrix

%% ── OLS helper ───────────────────────────────────────────────────────
%  Returns OLS coefficients b and heteroskedasticity-robust (HC0) SEs.
%  Defined as a local function at the bottom of this script.
%  (See ols_hc0 at end of file.)

%% ── (1) Direct LP ────────────────────────────────────────────────────
%  Regressor: s_direct_t = Omega_direct * eps_t
%             In matrix form: S_direct = eps * Omega_direct  (T×(K+1))
%             (Omega_direct is symmetric so no transpose needed)
%
%  LP specification at horizon h:
%    Y_{t+h} - Y_{t-1} = c + beta_h * s_direct_t + gamma_h * Y_lags + u_t
%
%  IRF_dir(n,k,h) = beta_h[n,k]  (direct response of outcome n to shock k)

S_direct = eps * Omega_direct;  % T×(K+1)

IRF_dir    = zeros(N, K+1, H+1);
IRF_dir_se = zeros(N, K+1, H+1);

for h = 0:H
    t0   = p + 1;  t1 = T - h;
    Teff = t1 - t0 + 1;

    dep    = Y(t0+h : t1+h, :) - Y(t0-1 : t1-1, :);  % Teff×N
    X_sh   = S_direct(t0:t1, :);                       % Teff×(K+1)

    X_ctrl = zeros(Teff, N*p);
    for l = 1:p
        X_ctrl(:, (l-1)*N+1 : l*N) = Y(t0-l : t1-l, :);
    end

    X = [ones(Teff,1), X_sh, X_ctrl];  % Teff × (1+(K+1)+N*p)

    for n = 1:N
        [b, se]              = ols_hc0(X, dep(:,n));
        IRF_dir(n,:,h+1)    = b(2 : K+2)';
        IRF_dir_se(n,:,h+1) = se(2 : K+2)';
    end
end

%% ── (2) Aggregate LP ─────────────────────────────────────────────────
%  Regressor: s_aggr_t = Omega_true * eps_t
%             In matrix form: S_aggr = eps * Omega_true'  (T×(K+1))
%
%  LP specification at horizon h:
%    Y_{t+h} - Y_{t-1} = c + A_h * s_aggr_t + gamma_h * Y_lags + u_t
%
%  A_h is N×(K+1).  The structural IRF is:
%    IRF_agg(:,:,h+1) = A_h * Omega_true
%  because s_aggr = Omega_true * eps  =>  A_h * s_aggr = A_h*Omega_true * eps

S_aggr = eps * Omega_true';  % T×(K+1)

IRF_agg    = zeros(N, K+1, H+1);
IRF_agg_se = zeros(N, K+1, H+1);

for h = 0:H
    t0   = p + 1;  t1 = T - h;
    Teff = t1 - t0 + 1;

    dep    = Y(t0+h : t1+h, :) - Y(t0-1 : t1-1, :);  % Teff×N
    X_ag   = S_aggr(t0:t1, :);                         % Teff×(K+1)

    X_ctrl = zeros(Teff, N*p);
    for l = 1:p
        X_ctrl(:, (l-1)*N+1 : l*N) = Y(t0-l : t1-l, :);
    end

    X = [ones(Teff,1), X_ag, X_ctrl];  % Teff × (1+(K+1)+N*p)

    A_h    = zeros(N, K+1);
    A_h_se = zeros(N, K+1);
    for n = 1:N
        [b, se]     = ols_hc0(X, dep(:,n));
        A_h(n,:)    = b(2 : K+2)';
        A_h_se(n,:) = se(2 : K+2)';
    end

    % Map aggregate-space coefficients back to structural shocks
    IRF_agg(:,:,h+1)    = A_h    * Omega_true;
    IRF_agg_se(:,:,h+1) = A_h_se * abs(Omega_true);  % conservative delta-method
end

%% ── Plotting ─────────────────────────────────────────────────────────
hz  = 0:H;          % horizon axis
ci  = 1.645;        % 90% confidence interval multiplier

fig = figure('Units','normalized','Position',[0 0 1 1]);

for k = 1 : K+1
    for n = 1:N
        idx = (k-1)*N + n;
        ax  = subplot(K+1, N, idx);

        tr      = squeeze(true_irf(n,k,:));
        dir_pt  = squeeze(IRF_dir(n,k,:));
        agg_pt  = squeeze(IRF_agg(n,k,:));
        dir_lo  = dir_pt - ci * squeeze(IRF_dir_se(n,k,:));
        dir_hi  = dir_pt + ci * squeeze(IRF_dir_se(n,k,:));
        agg_lo  = agg_pt - ci * squeeze(IRF_agg_se(n,k,:));
        agg_hi  = agg_pt + ci * squeeze(IRF_agg_se(n,k,:));

        hold(ax,'on');

        % 90% CI shading
        fill([hz, fliplr(hz)], [dir_lo', fliplr(dir_hi')], ...
             [0.18 0.55 0.95], 'FaceAlpha',0.20, 'EdgeColor','none');
        fill([hz, fliplr(hz)], [agg_lo', fliplr(agg_hi')], ...
             [0.92 0.35 0.20], 'FaceAlpha',0.20, 'EdgeColor','none');

        % Point estimates and true IRF
        plot(hz, tr,     'k-',  'LineWidth',2.0);
        plot(hz, dir_pt, 'b--', 'LineWidth',1.5);
        plot(hz, agg_pt, 'r-',  'LineWidth',1.5);
        yline(0, 'Color',[0.5 0.5 0.5], 'LineStyle',':', 'LineWidth',0.8);

        hold(ax,'off');
        title(sprintf('%s  \\rightarrow  %s', shock_names{k}, outcome_names{n}), ...
              'FontSize',8,'FontWeight','bold');
        xlabel('Horizon (quarters)','FontSize',7);
        ylabel('Response','FontSize',7);
        xlim([0 H]);
        grid on;  box on;

        if idx == 1
            legend('Direct LP 90% CI','Aggr LP 90% CI', ...
                   'True IRF','Direct LP','Aggregate LP', ...
                   'Location','best','FontSize',6);
        end
    end
end

sgtitle('IRFs via Local Projections: Direct Shock vs Aggregate Shock', ...
        'FontSize',13,'FontWeight','bold');

saveas(fig, 'irf_direct_vs_aggregate.png');
fprintf('IRF plot saved to irf_direct_vs_aggregate.png\n');

%% ── Local function ───────────────────────────────────────────────────
function [b, se] = ols_hc0(X, y)
%OLS_HC0  OLS with heteroskedasticity-robust (HC0) standard errors.
%   [b, se] = OLS_HC0(X, y) returns the OLS coefficient vector b and a
%   vector of HC0 standard errors se.
    b     = (X'*X) \ (X'*y);
    e     = y - X*b;
    XtXi  = inv(X'*X);
    se    = sqrt(diag( XtXi * (X' * diag(e.^2) * X) * XtXi ));
end
