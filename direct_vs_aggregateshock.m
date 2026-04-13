%%  direct_vs_aggregateshock.m
%   Corrected MATLAB script: Direct LP vs Aggregate-Shock LP
%   =========================================================
%
%   Bugs fixed vs the original draft
%   ---------------------------------
%   1.  Variable naming conflict: `i` was used as both the inflation
%       array AND the inner loop index `for i = 1:N`, silently over-
%       writing the data.  Renamed to `infl` (inflation) and `unemp`
%       (unemployment).
%   2.  Dimension error in X_sh: `Omega_direct * eps(t0:t1,:)` multiplied
%       a 5×5 matrix by a Teff×5 matrix – invalid.  Fixed to
%       `eps(t0:t1,:) * Omega_direct` (Teff×5 × 5×5 = Teff×5).
%   3.  IRF_dir / IRF_dir_se had wrong 2nd dimension K (=4) instead of
%       K+1 (=5 shocks including Crisis).
%   4.  Direct LP extracted b(2:K+1) – only K=4 coefficients for 5 shocks.
%       Fixed to b(2:K+2).
%   5.  aggr_hat = eps * B' where B = A0_true * Omega_true (2×5) gave T×2
%       instead of T×5.  Fixed to eps * Omega_true' (T×5).
%   6.  A_h in aggregate LP was zeros(N,K) (2×4) instead of zeros(N,K+1).
%   7.  Aggregate LP extracted b(2:K+1) – same off-by-one.  Fixed to
%       b(2:K+2).
%   8.  `Omega_est` was never defined.  Replaced with Omega_true.
%   9.  DGP did not incorporate aggregate shocks into Y: the AR(1)
%       simulation generated pure noise so LP estimates were meaningless.
%       Added A0_true(n,:) * aggr(t,:)' to each outcome equation.
%  10.  Plotting code was missing.  Added a 2×5 subplot IRF comparison.
%  11.  OLS function uses pinv / backslash to handle the nearly-singular
%       crisis-shock column (all zeros for h >= 1).

clear; clc; rng(42);

%% ── Parameters ──────────────────────────────────────────────────────
T   = 600;          % sample size
K   = 4;            % number of structural shocks (crisis added separately → K+1 total)
N   = 2;            % number of outcomes (unemployment, inflation)
H   = 20;           % max horizon
p   = 4;            % lag order for LP controls
rho = 0.70;         % AR(1) persistence for true IRF decay
shock_names   = {'FP','MP','TFP','Oil','Crisis'};
outcome_names = {'Unemployment','Inflation'};

%% ── True parameters ─────────────────────────────────────────────────
Omega_true = [0.80 0.20 0.10 0.30 0.00;
              0.10 0.90 0.20 0.10 0.30;
              0.20 0.10 0.70 0.20 0.10;
              0.30 0.20 0.10 0.80 0.20;
              0.00 0.10 0.20 0.10 0.90];   % 5×5

A0_true = [ 0.50 -0.30  0.40 -0.20 -0.50;   % unemployment responses (1×5)
           -0.30  0.20 -0.20  0.30  0.40];   % inflation responses    (1×5)
                                              % A0_true is 2×5

% True IRF(h) = rho^h * A0_true * Omega_true  →  2×5 at each h
true_irf = zeros(N, K+1, H+1);
for h = 0:H
    true_irf(:,:,h+1) = (rho^h) * A0_true * Omega_true;
end

%% ── DGP: simulate shocks and outcomes ───────────────────────────────
crises_shock        = zeros(T,1);
crises_shock(T)     = 1;                        % one-time shock at end

eps = [randn(T, K), crises_shock];              % T×(K+1)
aggr = eps * Omega_true';                       % T×5

% Reset seed before simulating outcome paths (mirrors original intent)
rng(42);

unemp = zeros(T,1);   % unemployment rate  (Bug 1 fix: was `u`)
infl  = zeros(T,1);   % inflation rate     (Bug 1 fix: was `i`)
unemp(1) = 6;
infl(1)  = 2;

% Bug 9 fix: incorporate aggregate shock into each outcome equation
for t = 2:T
    unemp(t) = 0.5 + 0.9*unemp(t-1) + A0_true(1,:)*aggr(t,:)' + 0.5*randn();
    infl(t)  = 0.5 + 0.8*infl(t-1)  + A0_true(2,:)*aggr(t,:)' + 0.4*randn();
end

Y = [unemp, infl];   % T×2

%% ── Omega_direct: diagonal of Omega_true ───────────────────────────
Omega_direct = diag(diag(Omega_true));   % 5×5 diagonal

%% ── OLS helper ───────────────────────────────────────────────────────
% Returns coefficients b and HC0 (heteroskedasticity-robust) SE.
% Uses pinv to handle rank-deficient X (e.g. crisis column all-zeros).
function [b, se] = ols_hc0(X, y)
    b     = X \ y;                          % least-squares coefficients
    resid = y - X * b;
    XtX_inv = pinv(X' * X);                 % Bug 11 fix: pinv for robustness
    Xe    = X .* resid;                     % Teff×ncols  (HC0 meat efficiently)
    meat  = Xe' * Xe;
    vcov  = XtX_inv * meat * XtX_inv;
    se    = sqrt(max(diag(vcov), 0));
end

%% ── Direct LP ────────────────────────────────────────────────────────
% Bug 3 fix: 2nd dim is K+1 (5 shocks), not K (4)
IRF_dir    = zeros(N, K+1, H+1);
IRF_dir_se = zeros(N, K+1, H+1);

for h = 0:H
    t0   = p+1;   t1 = T-h;
    Teff = t1 - t0 + 1;

    dep = Y(t0+h:t1+h,:) - Y(t0-1:t1-1,:);     % Teff×N

    % Bug 2 fix: eps(…) * Omega_direct  (Teff×5 × 5×5 = Teff×5)
    X_sh   = eps(t0:t1,:) * Omega_direct;        % Teff×(K+1)
    X_ctrl = zeros(Teff, N*p);
    for l = 1:p
        X_ctrl(:, (l-1)*N+1:l*N) = Y(t0-l:t1-l,:);
    end
    X = [ones(Teff,1), X_sh, X_ctrl];           % Teff×(1+(K+1)+N*p)

    for n = 1:N
        [b, se] = ols_hc0(X, dep(:,n));
        % Bug 4 fix: b(2:K+2) extracts K+1=5 coefficients
        IRF_dir(n,:,h+1)    = b(2:K+2)';
        IRF_dir_se(n,:,h+1) = se(2:K+2)';
    end
end

%% ── Aggregate LP ─────────────────────────────────────────────────────
% Bug 5 fix: aggr_hat = eps * Omega_true' (T×5), not eps * B' (T×2)
aggr_hat = eps * Omega_true';              % T×(K+1)

% Bug 6 fix: 2nd dim is K+1
IRF_agg   = zeros(N, K+1, H+1);
A_h_store = zeros(H+1, N, K+1);

for h = 0:H
    t0   = p+1;   t1 = T-h;
    Teff = t1 - t0 + 1;

    dep    = Y(t0+h:t1+h,:) - Y(t0-1:t1-1,:);
    X_ag   = aggr_hat(t0:t1,:);               % Teff×(K+1)
    X_ctrl = zeros(Teff, N*p);
    for l = 1:p
        X_ctrl(:, (l-1)*N+1:l*N) = Y(t0-l:t1-l,:);
    end
    X = [ones(Teff,1), X_ag, X_ctrl];

    A_h = zeros(N, K+1);                      % Bug 6 fix
    for n = 1:N
        b = X \ dep(:,n);
        % Bug 7 fix: b(2:K+2) for K+1=5 coefficients
        A_h(n,:) = b(2:K+2)';
    end
    A_h_store(h+1,:,:) = A_h;
    % Bug 8 fix: Omega_est → Omega_true
    IRF_agg(:,:,h+1) = A_h * Omega_true;
end

%% ── Plot IRFs ────────────────────────────────────────────────────────
horizons  = 0:H;
ci_factor = 1.96;   % 95 % CI multiplier

fig = figure('Name','IRF: Direct vs Aggregate LP', ...
             'Position',[100 100 1400 550]);

for n = 1:N
    for k = 1:(K+1)
        subplot(N, K+1, (n-1)*(K+1) + k);

        true_h = squeeze(true_irf(n,k,:))';
        dir_h  = squeeze(IRF_dir(n,k,:))';
        dir_se = squeeze(IRF_dir_se(n,k,:))';
        agg_h  = squeeze(IRF_agg(n,k,:))';

        % Shaded 95 % CI for Direct LP
        fill([horizons, fliplr(horizons)], ...
             [dir_h - ci_factor*dir_se, fliplr(dir_h + ci_factor*dir_se)], ...
             'b', 'FaceAlpha', 0.15, 'EdgeColor','none'); hold on;

        plot(horizons, true_h, 'k-',  'LineWidth', 2,   'DisplayName','True IRF');
        plot(horizons, dir_h,  'b--', 'LineWidth', 1.5, 'DisplayName','Direct LP');
        plot(horizons, agg_h,  'r-.', 'LineWidth', 1.5, 'DisplayName','Aggregate LP');
        yline(0, ':k', 'LineWidth', 0.7, 'HandleVisibility','off');

        xlim([0, H]);
        grid on;

        if n == 1
            title(shock_names{k}, 'FontWeight','bold');
        end
        if k == 1
            ylabel(outcome_names{n});
        end
        if n == N
            xlabel('Horizon');
        end
        if n == N && k == ceil((K+1)/2)
            legend('Location','best','FontSize',7);
        end
    end
end

sgtitle('IRF: True vs Direct LP vs Aggregate LP', 'FontWeight','bold');

saveas(fig, 'irf_direct_vs_aggregate.png');
fprintf('IRF plot saved to irf_direct_vs_aggregate.png\n');
