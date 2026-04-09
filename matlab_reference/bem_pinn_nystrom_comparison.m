clear; clc; close all;

%% ============================================================
% BEM/Nystrom + PINN/BINN (adaptive residual-based collocation)
% - BEM reference is fixed and unchanged
% - PINN collocation set grows adaptively after selected Adam phases
% - Quadrature, kernel convention, and residual definition are unchanged
%% ============================================================

cfg = struct;
cfg.seed = 0;

cfg.NpEdge = 12;
cfg.pGL = 16;
cfg.mColBase = 1;
cfg.mColRing = 1;
cfg.mColCorner = 1;
cfg.wBase = 1.0;
cfg.wRing = 1.0;
cfg.wCorner = 1.0;

cfg.useEquationScaling = false;
cfg.eqScaleMode = 'fixed';      % 'fixed' or 'auto'
cfg.eqScale = 10.0;             % used when eqScaleMode='fixed'
cfg.nSubNear = 0;
cfg.useNearPanelRefine = false;
cfg.nGrid = 201;

cfg.adamPhaseIters = 500;
cfg.adamPhaseLr    = 1e-4;
cfg.logEvery = 100;

cfg.useLBFGS = true;
cfg.lbfgsMaxIters = 3500;
cfg.lbfgsGradTol = 1e-8;
cfg.lbfgsStepTol = 1e-12;
cfg.lbfgsLineSearch = 'armijo-backtracking';
cfg.lbfgsMemory = 10;
cfg.lbfgsLogEvery = 25;
cfg.lbfgsAlpha0 = 1e-1;
cfg.lbfgsAlphaFallback = [1e-2, 1e-3];
cfg.lbfgsArmijoC1 = 1e-4;
cfg.lbfgsBacktrackBeta = 0.5;
cfg.lbfgsMaxBacktrack = 20;
cfg.lbfgsStepTolNeedsGrad = true;

cfg.useAdaptiveCollocation = false;
cfg.deltaAdapt = 0.5;
cfg.nNewPerPanel = 3;
cfg.adaptEveryAdam = 100;
cfg.adaptEveryLBFGS = 500;
cfg.maxNbFactor = 2.0;
cfg.duplicateTol = 1e-10;

cfg.gmresTol = 1e-12;
cfg.gmresMaxit = 300;
cfg.gmresRestart = [];
cfg.useDirectFallback = true;

cfg.u_exact = @(x,y) x.^2 - y.^2;

rng(cfg.seed);
P = Koch(1);
Nv = size(P,1);
fprintf('Vertices: %d\n', Nv);

%% ============================================================
% 1) Geometry and panelization
%% ============================================================
panels = build_uniform_panels(P, cfg.NpEdge);
Npan = numel(panels);

L_panel = zeros(Npan,1);
for pid = 1:Npan
    seg = panels{pid};
    L_panel(pid) = norm(seg(2,:) - seg(1,:));
end

fprintf('Panels: %d\n', Npan);
fprintf('Panel length min/max = %.3e / %.3e\n', min(L_panel), max(L_panel));

%% ============================================================
% 2) Standard quadrature (also Nyström nodes for BEM)
%% ============================================================
[Yq, wq, pan_id, s_on_panel, L_panel2, idxStd] = build_panel_gauss_polygon_with_index(panels, cfg.pGL);
Nq = size(Yq,2);

fprintf('Quadrature/Nystrom nodes Nq: %d\n', Nq);
fprintf('Quadrature weight min/max = %.3e / %.3e\n', min(wq), max(wq));

if any(abs(L_panel - L_panel2) > 1e-14)
    warning('Panel length mismatch in quadrature build.');
end

if cfg.useNearPanelRefine
    [YqR, wqR, pan_idR, idxRef] = build_panel_gauss_polygon_refined_with_index(panels, cfg.pGL, cfg.nSubNear); %#ok<NASGU>
    fprintf('Refined near rule: nSubNear=%d, NqR=%d\n', cfg.nSubNear, size(YqR,2));
else
    YqR = [];
    wqR = [];
    idxRef = [];
    fprintf('Refined near rule: disabled\n');
end

%% ============================================================
% 3) BEM/Nystrom reference solve (unchanged)
%% ============================================================
fprintf('\n=== Part A: BEM/Nystrom reference ===\n');
fprintf('Assembling BEM matrix (%d x %d)...\n', Nq, Nq);

[Vbem, corrBem, bemInfo] = assemble_nystrom_matrix( ...
    Yq, wq, pan_id, s_on_panel, L_panel, idxStd, ...
    YqR, wqR, idxRef, cfg.pGL, cfg.nSubNear, cfg.useNearPanelRefine);

fBem = cfg.u_exact(Yq(1,:).', Yq(2,:).');

fprintf('BEM assembly complete.\n');
fprintf('  corr min/max = %.3e / %.3e\n', min(corrBem), max(corrBem));
if cfg.useNearPanelRefine
    fprintf('  near replacement rows touched = %d\n', bemInfo.nRowsTouched);
end

[sigmaBem, flagBem, relresBem, iterBem, resvecBem] = gmres(Vbem, fBem, ...
    cfg.gmresRestart, cfg.gmresTol, cfg.gmresMaxit); %#ok<ASGLU>

if numel(iterBem) >= 2
    gmresIterCount = iterBem(2);
else
    gmresIterCount = iterBem;
end

fprintf('BEM GMRES flag       = %d\n', flagBem);
fprintf('BEM GMRES relres     = %.3e\n', relresBem);
fprintf('BEM GMRES iterations = %d\n', gmresIterCount);

if flagBem ~= 0 && cfg.useDirectFallback
    fprintf('BEM GMRES not fully converged. Using direct fallback V\\f.\n');
    sigmaBem = Vbem \ fBem;
end

%% ============================================================
% 4) PINN setup: panel classes, weights, and initial collocation
%% ============================================================
[~, isCornerPanel, isRingPanel] = corner_graded_collocation_counts( ...
    panels, P, cfg.mColBase, cfg.mColCorner, cfg.mColRing);
wPanel = build_panel_loss_weights(isCornerPanel, isRingPanel, cfg.wBase, cfg.wCorner, cfg.wRing);

mColInitPanel = cfg.mColBase * ones(Npan,1);
[XcInit, panInit, s0Init] = build_collocation_points_per_panel(panels, mColInitPanel);
collocInit = struct('Xc', XcInit, 'pan_of_xc', panInit, 's0_of_xc', s0Init);
NbInit = size(XcInit,1);

fprintf('\n=== Part B: PINN/BINN training (static or adaptive collocation) ===\n');
fprintf('Initial collocation diagnostics:\n');
fprintf('  total panels          = %d\n', Npan);
fprintf('  corner panels         = %d\n', nnz(isCornerPanel));
fprintf('  ring panels           = %d\n', nnz(isRingPanel));
fprintf('  initial Nb            = %d\n', NbInit);
fprintf('  mCol init min/max     = %d / %d\n', min(mColInitPanel), max(mColInitPanel));
fprintf('Adaptive settings:\n');
fprintf('  deltaAdapt            = %.3f\n', cfg.deltaAdapt);
fprintf('  nNewPerPanel          = %d\n', cfg.nNewPerPanel);
fprintf('  adaptEveryAdam        = %d\n', cfg.adaptEveryAdam);
fprintf('  adaptEveryLBFGS       = %d\n', cfg.adaptEveryLBFGS);
fprintf('  maxNbFactor           = %.2f (Nb <= %d)\n', cfg.maxNbFactor, floor(cfg.maxNbFactor * NbInit));

uW = unique(wPanel);
fprintf('Loss weight diagnostics:\n');
fprintf('  panel weight min/max  = %.3e / %.3e\n', min(wPanel), max(wPanel));
fprintf('  unique panel weights  = [%s]\n', num2str(uW.'));

staticData = struct;
staticData.Yq = Yq;
staticData.wq = wq;
staticData.idxStd = idxStd;
staticData.L_panel = L_panel;
staticData.YqR = YqR;
staticData.wqR = wqR;
staticData.idxRef = idxRef;
staticData.Npan = Npan;
staticData.useNearPanelRefine = cfg.useNearPanelRefine;
staticData.u_exact = cfg.u_exact;
staticData.P = P;
staticData.panels = panels;

%% ============================================================
% 5) Run PINN case (static or adaptive workflow)
%% ============================================================
rng(cfg.seed);
netAdaptive = build_pinn_network();
if cfg.useAdaptiveCollocation
    workflowTag = 'Adaptive';
    fprintf('\n--- Running %s PINN case ---\n', workflowTag);
    adaptiveOut = run_adaptive_training_case( ...
        netAdaptive, collocInit, wPanel, staticData, cfg, workflowTag);
else
    workflowTag = 'Static';
    fprintf('\n--- Running %s PINN case ---\n', workflowTag);
    adaptiveOut = run_static_training_case( ...
        netAdaptive, collocInit, wPanel, staticData, cfg, workflowTag);
end

sigmaPINN_q = adaptiveOut.sigmaFinal_q;
sigmaPINN_adam_q = adaptiveOut.sigmaAdam_q;
densityRelDiff = norm(sigmaPINN_q - sigmaBem, 2) / max(norm(sigmaBem,2), 1e-14);
densityRelDiffAdam = norm(sigmaPINN_adam_q - sigmaBem, 2) / max(norm(sigmaBem,2), 1e-14);

%% ============================================================
% 6) Interior reconstructions
%% ============================================================
bemOut = reconstruct_interior_solution(P, Yq, wq, sigmaBem, cfg.nGrid, cfg.u_exact);
pinnOut = reconstruct_interior_solution(P, Yq, wq, sigmaPINN_q, cfg.nGrid, cfg.u_exact);
pinnOutAdam = reconstruct_interior_solution(P, Yq, wq, sigmaPINN_adam_q, cfg.nGrid, cfg.u_exact);

fprintf('\n=== Interior error summary ===\n');
fprintf('BEM           Relative L2 = %.3e | Linf = %.3e\n', bemOut.relL2, bemOut.linf);
fprintf('PINN %s   Relative L2 = %.3e | Linf = %.3e\n', workflowTag, pinnOut.relL2, pinnOut.linf);
fprintf('PINN Adam-end Relative L2 = %.3e | Linf = %.3e\n', pinnOutAdam.relL2, pinnOutAdam.linf);

fprintf('\n=== Density comparison against BEM ===\n');
fprintf('%s Adam-end density rel diff  = %.3e\n', workflowTag, densityRelDiffAdam);
fprintf('%s final density rel diff     = %.3e\n', workflowTag, densityRelDiff);

%% ============================================================
% 7) Interactive visualization (no file output)
%% ============================================================
s_global = global_arclength_coordinate(panels, pan_id, s_on_panel);
[ss, perm] = sort(s_global);

% Density comparison
figure;
plot(ss, sigmaBem(perm), 'k-', 'LineWidth', 1.3); hold on;
plot(ss, sigmaPINN_q(perm), 'r--', 'LineWidth', 1.3);
grid on;
xlabel('boundary arclength s');
ylabel('\sigma(s)');
legend('BEM Nyström', sprintf('PINN %s', workflowTag), 'Location', 'best');
title(sprintf('Density Comparison (BEM vs PINN, %s)', workflowTag));

% Loss history
figure;
semilogy(1:numel(adaptiveOut.lossHistAdam), adaptiveOut.lossHistAdam, 'LineWidth', 1.3); hold on;
for k = 1:numel(adaptiveOut.adaptIters)
    xline(adaptiveOut.adaptIters(k), ':', adaptiveOut.adaptLabels{k}, ...
        'Color', [0.1 0.6 0.1], 'LabelVerticalAlignment', 'middle');
end
if ~isempty(adaptiveOut.lossHistLBFGS)
    semilogy(numel(adaptiveOut.lossHistAdam) + (1:numel(adaptiveOut.lossHistLBFGS)), ...
        adaptiveOut.lossHistLBFGS, 'LineWidth', 1.3);
    xline(numel(adaptiveOut.lossHistAdam), '--k', 'Adam->LBFGS', 'LabelVerticalAlignment', 'middle');
    legend('Adam', 'Adapt points', 'LBFGS', 'Location', 'best');
else
    legend('Adam', 'Adapt points', 'Location', 'best');
end
grid on;
xlabel('Training iteration');
ylabel('Loss');
title(sprintf('%s PINN Loss History', workflowTag));

figure;
imagesc(bemOut.xv, bemOut.yv, bemOut.Ugrid); axis image; set(gca,'YDir','normal');
title('BEM Interior Solution'); xlabel('x'); ylabel('y'); colorbar; hold on; plot_polygon(P);
apply_image_alpha_from_nan(bemOut.Ugrid);

figure;
imagesc(pinnOut.xv, pinnOut.yv, pinnOut.Ugrid); axis image; set(gca,'YDir','normal');
title('PINN Interior Solution'); xlabel('x'); ylabel('y'); colorbar; hold on; plot_polygon(P);
apply_image_alpha_from_nan(pinnOut.Ugrid);

figure;
imagesc(bemOut.xv, bemOut.yv, bemOut.Uexgrid); axis image; set(gca,'YDir','normal');
title('Exact Interior Solution'); xlabel('x'); ylabel('y'); colorbar; hold on; plot_polygon(P);
apply_image_alpha_from_nan(bemOut.Uexgrid);

figure;
imagesc(bemOut.xv, bemOut.yv, log10(abs(bemOut.Egrid)+1e-16)); axis image; set(gca,'YDir','normal');
title('BEM log10 Error'); xlabel('x'); ylabel('y'); colorbar; hold on; plot_polygon(P);
apply_image_alpha_from_nan(bemOut.Egrid);

figure;
imagesc(pinnOut.xv, pinnOut.yv, log10(abs(pinnOut.Egrid)+1e-16)); axis image; set(gca,'YDir','normal');
title('PINN log10 Error'); xlabel('x'); ylabel('y'); colorbar; hold on; plot_polygon(P);
apply_image_alpha_from_nan(pinnOut.Egrid);

%% ============================================================
% 8) Command-window metrics (interactive mode)
%% ============================================================
fprintf('\n=== Interactive Metrics ===\n');
fprintf('[Collocation]\n');
fprintf('  Nb initial            : %d\n', NbInit);
fprintf('  Nb final              : %d\n', adaptiveOut.NbFinal);
fprintf('  Adaptation steps      : %d\n', numel(adaptiveOut.nAddedPerStep));
fprintf('  Added points total    : %d\n', sum(adaptiveOut.nAddedPerStep));

fprintf('[Errors]\n');
fprintf('  PINN Relative L2      : %.6e\n', pinnOut.relL2);
fprintf('  PINN Linf             : %.6e\n', pinnOut.linf);
fprintf('  BEM  Relative L2      : %.6e\n', bemOut.relL2);
fprintf('  BEM  Linf             : %.6e\n', bemOut.linf);

fprintf('[Loss]\n');
fprintf('  Final Adam loss       : %.6e\n', adaptiveOut.finalAdamLoss);
if isnan(adaptiveOut.finalLBFGSLoss)
    fprintf('  Final LBFGS loss      : NaN\n');
else
    fprintf('  Final LBFGS loss      : %.6e\n', adaptiveOut.finalLBFGSLoss);
end
fprintf('  Final total loss      : %.6e\n', adaptiveOut.finalLoss);

fprintf('[Training]\n');
fprintf('  Total iterations      : %d\n', numel(adaptiveOut.lossHistCombined));
fprintf('  Adapt every Adam      : %d\n', cfg.adaptEveryAdam);
fprintf('  Adapt every LBFGS     : %d\n', cfg.adaptEveryLBFGS);


%% =====================================================================
% Local functions
%% =====================================================================

function panels = build_uniform_panels(P, NpEdge)
Nv = size(P,1);
panels = cell(Nv*NpEdge,1);
k = 1;
for e = 1:Nv
    a = P(e,:);
    b = P(mod(e,Nv)+1,:);
    for j = 1:NpEdge
        t0 = (j-1)/NpEdge;
        t1 = j/NpEdge;
        x0 = (1-t0)*a + t0*b;
        x1 = (1-t1)*a + t1*b;
        panels{k} = [x0; x1];
        k = k + 1;
    end
end
end

function [Yq, wq, pan_id, s_on_panel, L_panel, idxStd] = build_panel_gauss_polygon_with_index(panels, p)
[xi, wi] = gauss_legendre(p);

Npan = numel(panels);
L_panel = zeros(Npan,1);

Nq = Npan*p;
Yq = zeros(2, Nq);
wq = zeros(Nq, 1);
pan_id = zeros(Nq, 1);
s_on_panel = zeros(Nq, 1);
idxStd = cell(Npan,1);

k = 1;
for m = 1:Npan
    seg = panels{m};
    a = seg(1,:)';
    b = seg(2,:)';
    L = norm(b-a);
    L_panel(m) = L;

    idxStd{m} = k:(k+p-1);

    for j = 1:p
        t = (xi(j)+1)/2;
        y = (1-t)*a + t*b;

        Yq(:,k) = y;
        wq(k) = (L/2) * wi(j);
        pan_id(k) = m;
        s_on_panel(k) = t * L;
        k = k + 1;
    end
end
end

function [YqR, wqR, pan_idR, idxRef] = build_panel_gauss_polygon_refined_with_index(panels, pGL, nSub)
[xi, wi] = gauss_legendre(pGL);

Npan = numel(panels);
ptsPerPanel = nSub * pGL;
NqR = Npan * ptsPerPanel;

YqR = zeros(2, NqR);
wqR = zeros(NqR, 1);
pan_idR = zeros(NqR, 1);
idxRef = cell(Npan,1);

k = 1;
for m = 1:Npan
    seg = panels{m};
    a = seg(1,:)';
    b = seg(2,:)';

    idxRef{m} = k:(k+ptsPerPanel-1);

    for ss = 1:nSub
        tA = (ss-1)/nSub;
        tB = ss/nSub;

        aS = (1-tA)*a + tA*b;
        bS = (1-tB)*a + tB*b;
        Ls = norm(bS-aS);

        for j = 1:pGL
            t = (xi(j)+1)/2;
            y = (1-t)*aS + t*bS;

            YqR(:,k) = y;
            wqR(k) = (Ls/2) * wi(j);
            pan_idR(k) = m;
            k = k + 1;
        end
    end
end
end

function [mColPanel, isCornerPanel, isRingPanel] = corner_graded_collocation_counts( ...
         panels, P, mColBase, mColCorner, mColRing)
Npan = numel(panels);
isCornerPanel = false(Npan,1);

tolCorner = 1e-10 * max(1, max(abs(P(:))));
for pid = 1:Npan
    seg = panels{pid};
    a = seg(1,:);
    b = seg(2,:);
    dA = min(vecnorm(P - a, 2, 2));
    dB = min(vecnorm(P - b, 2, 2));
    if dA < tolCorner || dB < tolCorner
        isCornerPanel(pid) = true;
    end
end

isRingPanel = (circshift(isCornerPanel,1) | circshift(isCornerPanel,-1)) & ~isCornerPanel;

mColPanel = mColBase * ones(Npan,1);
mColPanel(isRingPanel) = mColRing;
mColPanel(isCornerPanel) = mColCorner;
end

function wPanel = build_panel_loss_weights(isCornerPanel, isRingPanel, wBase, wCorner, wRing)
Npan = numel(isCornerPanel);
wPanel = wBase * ones(Npan,1);
wPanel(isRingPanel) = wRing;
wPanel(isCornerPanel) = wCorner;
end

function [Xc, pan_of_xc, s0_of_xc] = build_collocation_points_per_panel(panels, mColPanel)
Npan = numel(panels);

if isscalar(mColPanel)
    mColPanel = mColPanel * ones(Npan,1);
end

Nb = sum(mColPanel);
Xc = zeros(Nb,2);
pan_of_xc = zeros(Nb,1);
s0_of_xc = zeros(Nb,1);

k = 1;
for pid = 1:Npan
    seg = panels{pid};
    a = seg(1,:)';
    b = seg(2,:)';
    L = norm(b-a);

    mThis = mColPanel(pid);
    [xi, ~] = gauss_legendre(mThis);
    tcol = (xi(:) + 1)/2;

    for j = 1:mThis
        t = tcol(j);
        x = (1-t)*a + t*b;
        Xc(k,:) = x.';
        pan_of_xc(k) = pid;
        s0_of_xc(k) = t * L;
        k = k + 1;
    end
end
end

function [Vmat, corr, info] = assemble_nystrom_matrix( ...
         Yq, wq, pan_id, s_on_panel, L_panel, idxStd, ...
         YqR, wqR, idxRef, pGL, nSubNear, useNearPanelRefine)

Nq = size(Yq,2);
Npan = numel(idxStd);
A = zeros(Nq, Nq);

% Standard quadrature contribution for all rows/nodes
for i = 1:Nq
    xi = Yq(:,i);
    r = vecnorm(Yq - xi, 2, 1);
    r = max(r, 1e-14);
    r(i) = 1;

    G = -(1/(2*pi)) * log(r);
    row = (G(:) .* wq(:)).';
    row(i) = 0;
    A(i,:) = row;
end

% Adjacent-panel replacement: subtract standard adjacent blocks, add refined-projected blocks.
if useNearPanelRefine
    TrefStd = refined_to_standard_projection_matrix(pGL, nSubNear); % (nSubNear*pGL) x pGL

    for i = 1:Nq
        pid = pan_id(i);
        im1 = mod(pid-2, Npan) + 1;
        ip1 = mod(pid,   Npan) + 1;
        adjPanels = [im1, ip1];

        for kk = 1:2
            padj = adjPanels(kk);
            js = idxStd{padj};
            jr = idxRef{padj};

            oldStd = A(i, js);

            yR = YqR(:, jr);
            rR = vecnorm(yR - Yq(:,i), 2, 1);
            rR = max(rR, 1e-14);
            GR = -(1/(2*pi)) * log(rR);
            kRef = GR(:) .* wqR(jr);

            ArefToStd = (kRef.' * TrefStd); % 1 x pGL
            A(i, js) = A(i, js) - oldStd + ArefToStd;
        end
    end
end

% Self-panel analytic correction
corr = zeros(Nq,1);
for i = 1:Nq
    pid = pan_id(i);
    js = idxStd{pid};

    I2 = self_panel_integral_log_kernel(L_panel(pid), s_on_panel(i));
    sumSelf = sum(A(i, js));
    corr(i) = I2 - sumSelf;
end

Vmat = A + diag(corr);

info = struct;
if useNearPanelRefine
    info.nRowsTouched = Nq;
else
    info.nRowsTouched = 0;
end
end

function [A, corr] = build_A_and_corr_multiColloc(Xc, pan_of_xc, s0_of_xc, Yq, wq, idxStd, L_panel)
Nb = size(Xc,1);
Nq = size(Yq,2);
A = zeros(Nb, Nq);
corr = zeros(Nb, 1);

for k = 1:Nb
    xk = Xc(k,:)';
    r = vecnorm(Yq - xk, 2, 1);
    r = max(r, 1e-14);
    G = -(1/(2*pi)) * log(r);
    A(k,:) = (G(:) .* wq(:)).';

    pid = pan_of_xc(k);
    js = idxStd{pid};

    I2 = self_panel_integral_log_kernel(L_panel(pid), s0_of_xc(k));
    sumSelf = sum(A(k, js));
    corr(k) = I2 - sumSelf;
end
end

function net = build_pinn_network()
layers = [
    featureInputLayer(2, 'Name','in')
    fullyConnectedLayer(80, 'Name','fc1')
    tanhLayer('Name','tanh1')
    fullyConnectedLayer(80, 'Name','fc2')
    tanhLayer('Name','tanh2')
    fullyConnectedLayer(80, 'Name','fc3')
    tanhLayer('Name','tanh3')
    fullyConnectedLayer(80, 'Name','fc4')
    tanhLayer('Name','tanh4')
    fullyConnectedLayer(1,  'Name','out')
];
net = dlnetwork(layers);
end

function out = run_static_training_case(net, collocInit, wPanel, staticData, cfg, caseName)
colloc = collocInit;
NbInit = size(colloc.Xc,1);

[op, opDiag] = build_pinn_operator_state(colloc, wPanel, staticData, cfg);
fprintf('%s | operator scaling: enabled=%d mode=%s eqScale=%.3e\n', ...
    caseName, cfg.useEquationScaling, cfg.eqScaleMode, op.eqScale);
fprintf('%s | mean|A| before/after = %.3e / %.3e\n', ...
    caseName, opDiag.meanAbsA_before, opDiag.meanAbsA_after);

nPhases = numel(cfg.adamPhaseIters);
totalIters = sum(cfg.adamPhaseIters);
lossHistAdam = zeros(totalIters,1);
trailingAvg = [];
trailingAvgSq = [];
it = 0;

adaptIters = zeros(0,1);
adaptLabels = cell(0,1);
nAddedPerStep = zeros(0,1);
panelIndicators = {};
selectedPanelsHistory = {};
collocHistory = struct('stage', {}, 'iter', {}, 'Nb', {}, 'Xc', {}, 'pan_of_xc', {}, 's0_of_xc', {});
collocHistory(end+1) = struct('stage', 'init', 'iter', 0, 'Nb', NbInit, ...
    'Xc', colloc.Xc, 'pan_of_xc', colloc.pan_of_xc, 's0_of_xc', colloc.s0_of_xc);

fprintf('--- Static Adam stage start ---\n');
fprintf('%s Adam phases: iters=[%s], lr=[%s]\n', caseName, ...
    num2str(cfg.adamPhaseIters), num2str(cfg.adamPhaseLr));

for iPhase = 1:nPhases
    nItPhase = cfg.adamPhaseIters(iPhase);
    lrPhase = cfg.adamPhaseLr(iPhase);
    fprintf('%s phase %d/%d | iters=%d | lr=%.1e | Nb=%d\n', ...
        caseName, iPhase, nPhases, nItPhase, lrPhase, op.Nb);

    for j = 1:nItPhase
        it = it + 1;
        [lossVal, grads, dbg] = dlfeval(@modelLossBINN_state, net, op);
        lossHistAdam(it) = double(gather(extractdata(lossVal)));

        [net, trailingAvg, trailingAvgSq] = adamupdate(net, grads, ...
            trailingAvg, trailingAvgSq, it, lrPhase);

        if mod(it, cfg.logEvery) == 0 || it == 1 || j == 1 || it == totalIters
            [gNorm, badGrad] = grad_norm_and_finite(grads);
            dbgNum = gather(extractdata(dbg));
            fprintf(['%s Adam %4d | wLoss=%.3e | uMSEsc=%.3e | uMSEunsc=%.3e | gradNorm=%.3e | badGrad=%d | lr=%.1e | phase=%d\n' ...
                     '              mean|Vstd|=%.3e mean|Vcorr|=%.3e mean|res|=%.3e nearSub/add=%.3e/%.3e\n'], ...
                     caseName, it, lossHistAdam(it), dbgNum(6), dbgNum(7), gNorm, badGrad, lrPhase, iPhase, ...
                     dbgNum(1), dbgNum(2), dbgNum(3), dbgNum(4), dbgNum(5));
        end
    end
end

finalAdamLoss = lossHistAdam(end);
sigmaAdam_q = forward(net, dlarray(staticData.Yq, 'CB'));
sigmaAdam_q = gather(extractdata(sigmaAdam_q(:)));
fprintf('--- Static Adam stage end | final loss = %.3e ---\n', finalAdamLoss);

lossHistLBFGS = zeros(0,1);
gradHistLBFGS = zeros(0,1);
lbfgsInfo = struct('reason', 'disabled', 'iterations', 0, 'lineSearchFailures', 0, 'blocks', 0);
if cfg.useLBFGS
    fprintf('--- Static LBFGS stage start ---\n');
    [net, lossHistLBFGS, gradHistLBFGS, infoBlock] = run_lbfgs_refinement(net, op, cfg);
    lbfgsInfo = infoBlock;
    lbfgsInfo.blocks = 1;
    fprintf('--- Static LBFGS stage end | iters=%d | reason=%s ---\n', ...
        lbfgsInfo.iterations, lbfgsInfo.reason);
end

if ~isempty(lossHistLBFGS)
    finalLoss = lossHistLBFGS(end);
    finalLBFGSLoss = lossHistLBFGS(end);
else
    finalLoss = finalAdamLoss;
    finalLBFGSLoss = NaN;
end

sigmaFinal_q = forward(net, dlarray(staticData.Yq, 'CB'));
sigmaFinal_q = gather(extractdata(sigmaFinal_q(:)));

out = struct;
out.caseName = caseName;
out.net = net;
out.finalLoss = finalLoss;
out.finalAdamLoss = finalAdamLoss;
out.finalLBFGSLoss = finalLBFGSLoss;
out.lossHistAdam = lossHistAdam;
out.lossHistLBFGS = lossHistLBFGS;
out.lossHistCombined = [lossHistAdam; lossHistLBFGS];
out.gradHistLBFGS = gradHistLBFGS;
out.lbfgsInfo = lbfgsInfo;
out.sigmaFinal_q = sigmaFinal_q;
out.sigmaAdam_q = sigmaAdam_q;
out.adaptIters = adaptIters;
out.adaptLabels = adaptLabels;
out.nAddedPerStep = nAddedPerStep;
out.panelIndicators = panelIndicators;
out.selectedPanelsHistory = selectedPanelsHistory;
out.collocHistory = collocHistory;
out.NbInit = NbInit;
out.NbFinal = size(colloc.Xc,1);
out.eqScaleLast = op.eqScale;
out.finalColloc = colloc;
end

function out = run_adaptive_training_case(net, collocInit, wPanel, staticData, cfg, caseName)
colloc = collocInit;
NbInit = size(colloc.Xc,1);
NbMax = floor(cfg.maxNbFactor * NbInit);

[op, opDiag] = build_pinn_operator_state(colloc, wPanel, staticData, cfg);
fprintf('%s | operator scaling: enabled=%d mode=%s eqScale=%.3e\n', ...
    caseName, cfg.useEquationScaling, cfg.eqScaleMode, op.eqScale);
fprintf('%s | mean|A| before/after = %.3e / %.3e\n', ...
    caseName, opDiag.meanAbsA_before, opDiag.meanAbsA_after);

nPhases = numel(cfg.adamPhaseIters);
totalIters = sum(cfg.adamPhaseIters);
lossHistAdam = zeros(totalIters,1);
trailingAvg = [];
trailingAvgSq = [];
it = 0;

adaptIters = zeros(0,1);
adaptLabels = cell(0,1);
nAddedPerStep = zeros(0,1);
panelIndicators = {};
selectedPanelsHistory = {};
collocHistory = struct('stage', {}, 'iter', {}, 'Nb', {}, 'Xc', {}, 'pan_of_xc', {}, 's0_of_xc', {});
collocHistory(end+1) = struct('stage', 'init', 'iter', 0, 'Nb', NbInit, ...
    'Xc', colloc.Xc, 'pan_of_xc', colloc.pan_of_xc, 's0_of_xc', colloc.s0_of_xc);

fprintf('--- Adaptive Adam stage start ---\n');
fprintf('%s Adam phases: iters=[%s], lr=[%s]\n', caseName, ...
    num2str(cfg.adamPhaseIters), num2str(cfg.adamPhaseLr));

for iPhase = 1:nPhases
    nItPhase = cfg.adamPhaseIters(iPhase);
    lrPhase = cfg.adamPhaseLr(iPhase);
    fprintf('%s phase %d/%d | iters=%d | lr=%.1e | Nb=%d\n', ...
        caseName, iPhase, nPhases, nItPhase, lrPhase, op.Nb);

    for j = 1:nItPhase
        it = it + 1;
        [lossVal, grads, dbg] = dlfeval(@modelLossBINN_state, net, op);
        lossHistAdam(it) = double(gather(extractdata(lossVal)));

        [net, trailingAvg, trailingAvgSq] = adamupdate(net, grads, ...
            trailingAvg, trailingAvgSq, it, lrPhase);

        if mod(it, cfg.logEvery) == 0 || it == 1 || j == 1 || it == totalIters
            [gNorm, badGrad] = grad_norm_and_finite(grads);
            dbgNum = gather(extractdata(dbg));
            fprintf(['%s Adam %4d | wLoss=%.3e | uMSEsc=%.3e | uMSEunsc=%.3e | gradNorm=%.3e | badGrad=%d | lr=%.1e | phase=%d\n' ...
                     '              mean|Vstd|=%.3e mean|Vcorr|=%.3e mean|res|=%.3e nearSub/add=%.3e/%.3e\n'], ...
                     caseName, it, lossHistAdam(it), dbgNum(6), dbgNum(7), gNorm, badGrad, lrPhase, iPhase, ...
                     dbgNum(1), dbgNum(2), dbgNum(3), dbgNum(4), dbgNum(5));
        end
        if mod(it, cfg.adaptEveryAdam) == 0
            [colloc, op, adaptInfo] = apply_residual_adaptation( ...
                net, colloc, op, wPanel, staticData, cfg, NbMax);
            adaptIters(end+1,1) = it; %#ok<AGROW>
            adaptLabels{end+1,1} = sprintf('adam@%d', it); %#ok<AGROW>
            nAddedPerStep(end+1,1) = adaptInfo.nAdded; %#ok<AGROW>
            panelIndicators{end+1} = adaptInfo.indicator; %#ok<AGROW>
            selectedPanelsHistory{end+1} = adaptInfo.selectedPanels; %#ok<AGROW>
            collocHistory(end+1) = struct('stage', 'adam', 'iter', it, 'Nb', adaptInfo.NbAfter, ... %#ok<AGROW>
                'Xc', colloc.Xc, 'pan_of_xc', colloc.pan_of_xc, 's0_of_xc', colloc.s0_of_xc);

            fprintf(['%s adapt@adam-%d | panels selected=%d | added=%d | Nb: %d -> %d | ' ...
                     'indicator min/max=%.3e / %.3e | mean|res|=%.3e\n'], ...
                    caseName, it, numel(adaptInfo.selectedPanels), adaptInfo.nAdded, ...
                    adaptInfo.NbBefore, adaptInfo.NbAfter, ...
                    min(adaptInfo.indicator), max(adaptInfo.indicator), adaptInfo.meanAbsRes);
        end
    end
end

finalAdamLoss = lossHistAdam(end);
sigmaAdam_q = forward(net, dlarray(staticData.Yq, 'CB'));
sigmaAdam_q = gather(extractdata(sigmaAdam_q(:)));
fprintf('--- Adaptive Adam stage end | final loss = %.3e ---\n', finalAdamLoss);

lossHistLBFGS = zeros(0,1);
gradHistLBFGS = zeros(0,1);
lbfgsInfo = struct('reason', 'disabled', 'iterations', 0, 'lineSearchFailures', 0);
if cfg.useLBFGS
    fprintf('--- Adaptive LBFGS stage start ---\n');
    lbfgsDone = 0;
    blockId = 0;
    lsFailTot = 0;
    lbfgsReason = 'maxIters';

    while lbfgsDone < cfg.lbfgsMaxIters
        blockId = blockId + 1;
        thisBlockIters = min(cfg.adaptEveryLBFGS, cfg.lbfgsMaxIters - lbfgsDone);
        cfgBlock = cfg;
        cfgBlock.lbfgsMaxIters = thisBlockIters;

        [net, lossBlock, gradBlock, infoBlock] = run_lbfgs_refinement(net, op, cfgBlock);
        lossHistLBFGS = [lossHistLBFGS; lossBlock]; %#ok<AGROW>
        gradHistLBFGS = [gradHistLBFGS; gradBlock]; %#ok<AGROW>
        lbfgsDone = lbfgsDone + numel(lossBlock);
        lsFailTot = lsFailTot + infoBlock.lineSearchFailures;

        if numel(lossBlock) < thisBlockIters
            lbfgsReason = infoBlock.reason;
        end

        if mod(lbfgsDone, cfg.adaptEveryLBFGS) == 0
            [colloc, op, adaptInfo] = apply_residual_adaptation( ...
                net, colloc, op, wPanel, staticData, cfg, NbMax);
            adaptIters(end+1,1) = totalIters + lbfgsDone; %#ok<AGROW>
            adaptLabels{end+1,1} = sprintf('lbfgs@%d', lbfgsDone); %#ok<AGROW>
            nAddedPerStep(end+1,1) = adaptInfo.nAdded; %#ok<AGROW>
            panelIndicators{end+1} = adaptInfo.indicator; %#ok<AGROW>
            selectedPanelsHistory{end+1} = adaptInfo.selectedPanels; %#ok<AGROW>
            collocHistory(end+1) = struct('stage', 'lbfgs', 'iter', lbfgsDone, 'Nb', adaptInfo.NbAfter, ... %#ok<AGROW>
                'Xc', colloc.Xc, 'pan_of_xc', colloc.pan_of_xc, 's0_of_xc', colloc.s0_of_xc);

            fprintf(['%s adapt@lbfgs-%d | panels selected=%d | added=%d | Nb: %d -> %d | ' ...
                     'indicator min/max=%.3e / %.3e | mean|res|=%.3e\n'], ...
                    caseName, lbfgsDone, numel(adaptInfo.selectedPanels), adaptInfo.nAdded, ...
                    adaptInfo.NbBefore, adaptInfo.NbAfter, ...
                    min(adaptInfo.indicator), max(adaptInfo.indicator), adaptInfo.meanAbsRes);
        end

        if numel(lossBlock) < thisBlockIters
            break;
        end
    end

    lbfgsInfo = struct('reason', lbfgsReason, 'iterations', lbfgsDone, ...
        'lineSearchFailures', lsFailTot, 'blocks', blockId);
    fprintf('--- Adaptive LBFGS stage end | iters=%d | reason=%s ---\n', ...
            lbfgsInfo.iterations, lbfgsInfo.reason);
end

if ~isempty(lossHistLBFGS)
    finalLoss = lossHistLBFGS(end);
    finalLBFGSLoss = lossHistLBFGS(end);
else
    finalLoss = finalAdamLoss;
    finalLBFGSLoss = NaN;
end

sigmaFinal_q = forward(net, dlarray(staticData.Yq, 'CB'));
sigmaFinal_q = gather(extractdata(sigmaFinal_q(:)));

out = struct;
out.caseName = caseName;
out.net = net;
out.finalLoss = finalLoss;
out.finalAdamLoss = finalAdamLoss;
out.finalLBFGSLoss = finalLBFGSLoss;
out.lossHistAdam = lossHistAdam;
out.lossHistLBFGS = lossHistLBFGS;
out.lossHistCombined = [lossHistAdam; lossHistLBFGS];
out.gradHistLBFGS = gradHistLBFGS;
out.lbfgsInfo = lbfgsInfo;
out.sigmaFinal_q = sigmaFinal_q;
out.sigmaAdam_q = sigmaAdam_q;
out.adaptIters = adaptIters;
out.adaptLabels = adaptLabels;
out.nAddedPerStep = nAddedPerStep;
out.panelIndicators = panelIndicators;
out.selectedPanelsHistory = selectedPanelsHistory;
out.collocHistory = collocHistory;
out.NbInit = NbInit;
out.NbFinal = size(colloc.Xc,1);
out.eqScaleLast = op.eqScale;
out.finalColloc = colloc;
end

function [op, diagInfo] = build_pinn_operator_state(colloc, wPanel, staticData, cfg)
[Acol, corrCol] = build_A_and_corr_multiColloc( ...
    colloc.Xc, colloc.pan_of_xc, colloc.s0_of_xc, ...
    staticData.Yq, staticData.wq, staticData.idxStd, staticData.L_panel);
fCol = staticData.u_exact(colloc.Xc(:,1), colloc.Xc(:,2));
wCol = wPanel(colloc.pan_of_xc);

meanAbsA_before = mean(abs(Acol(:)));
maxAbsA_before = max(abs(Acol(:)));
meanAbsf_before = mean(abs(fCol(:)));

if cfg.useEquationScaling
    if strcmpi(cfg.eqScaleMode, 'auto')
        eqScale = 1 / (meanAbsA_before + 1e-12);
    else
        eqScale = cfg.eqScale;
    end
else
    eqScale = 1.0;
end

Acol = eqScale * Acol;
corrCol = eqScale * corrCol;
fCol = eqScale * fCol;

op = struct;
op.Xc = colloc.Xc;
op.pan_of_xc = colloc.pan_of_xc;
op.s0_of_xc = colloc.s0_of_xc;
op.Nb = size(colloc.Xc,1);
op.Npan = staticData.Npan;
op.A_dl = dlarray(Acol);
op.corr_dl = dlarray(corrCol);
op.f_dl = dlarray(fCol);
op.wCol_dl = dlarray(wCol(:));
op.wColSum = sum(wCol);
op.eqScale = eqScale;
op.Xc_dl = dlarray(colloc.Xc.', 'CB');
op.Yq_dl = dlarray(staticData.Yq, 'CB');
op.idxStd = staticData.idxStd;
op.useNearPanelRefine = staticData.useNearPanelRefine;

if staticData.useNearPanelRefine
    op.YqR_dl = dlarray(staticData.YqR, 'CB');
    op.wqR_dl = dlarray(staticData.wqR);
    op.idxRef = staticData.idxRef;
else
    op.YqR_dl = dlarray(zeros(2,0), 'CB');
    op.wqR_dl = dlarray(zeros(0,1));
    op.idxRef = cell(staticData.Npan,1);
end

diagInfo = struct;
diagInfo.meanAbsA_before = meanAbsA_before;
diagInfo.maxAbsA_before = maxAbsA_before;
diagInfo.meanAbsf_before = meanAbsf_before;
diagInfo.meanAbsA_after = mean(abs(Acol(:)));
diagInfo.maxAbsA_after = max(abs(Acol(:)));
diagInfo.meanAbsf_after = mean(abs(fCol(:)));
diagInfo.eqScale = eqScale;
end

function [resAbs, resVec, dbgCore] = residual_abs_from_state(net, op)
[res, dbgCoreDl] = residual_vector_from_state(net, op);
resVec = gather(extractdata(res(:)));
resAbs = abs(resVec);
dbgCore = gather(extractdata(dbgCoreDl(:))).';
end

function indicator = panel_indicator_from_residual(resAbs, pan_of_xc, Npan)
pan = double(pan_of_xc(:));
vals = double(resAbs(:));
indicator = accumarray(pan, vals, [Npan, 1], @sum, 0);
end

function selectedPanels = select_panels_by_indicator(indicator, deltaAdapt)
tot = sum(indicator);
[indSort, ids] = sort(indicator, 'descend');
if isempty(ids)
    selectedPanels = zeros(0,1);
    return;
end
if tot <= 0
    selectedPanels = ids(1);
    return;
end
cumv = cumsum(indSort);
k = find(cumv >= deltaAdapt * tot, 1, 'first');
if isempty(k)
    k = numel(ids);
end
selectedPanels = ids(1:k);
selectedPanels = selectedPanels(indicator(selectedPanels) > 0);
if isempty(selectedPanels)
    selectedPanels = ids(1);
end
end

function [colloc, op, info] = apply_residual_adaptation( ...
         net, colloc, op, wPanel, staticData, cfg, NbMax)
[resAbs, ~, dbgCore] = residual_abs_from_state(net, op);
indicator = panel_indicator_from_residual(resAbs, colloc.pan_of_xc, staticData.Npan);
selectedPanels = select_panels_by_indicator(indicator, cfg.deltaAdapt);

NbBefore = size(colloc.Xc,1);
maxAddAllowed = max(0, NbMax - NbBefore);
[Xnew, panNew, s0New] = add_points_on_selected_panels( ...
    staticData.panels, colloc, selectedPanels, cfg.nNewPerPanel, ...
    maxAddAllowed, cfg.duplicateTol);

nAdded = size(Xnew,1);
if nAdded > 0
    colloc.Xc = [colloc.Xc; Xnew];
    colloc.pan_of_xc = [colloc.pan_of_xc; panNew];
    colloc.s0_of_xc = [colloc.s0_of_xc; s0New];
    [op, ~] = build_pinn_operator_state(colloc, wPanel, staticData, cfg);
end

info = struct;
info.indicator = indicator;
info.selectedPanels = selectedPanels;
info.nAdded = nAdded;
info.NbBefore = NbBefore;
info.NbAfter = size(colloc.Xc,1);
info.meanAbsRes = dbgCore(3);
end

function [Xnew, panNew, s0New] = add_points_on_selected_panels( ...
         panels, colloc, selectedPanels, nNewPerPanel, maxAddAllowed, duplicateTol)
Xnew = zeros(0,2);
panNew = zeros(0,1);
s0New = zeros(0,1);

if maxAddAllowed <= 0 || isempty(selectedPanels)
    return;
end

Xall = colloc.Xc;
for ii = 1:numel(selectedPanels)
    if size(Xnew,1) >= maxAddAllowed
        break;
    end
    pid = selectedPanels(ii);
    seg = panels{pid};
    a = seg(1,:)';
    b = seg(2,:)';
    L = norm(b-a);
    if L <= 0
        continue;
    end

    idxExist = (colloc.pan_of_xc == pid);
    tExist = colloc.s0_of_xc(idxExist) / L;
    nNeed = min(nNewPerPanel, maxAddAllowed - size(Xnew,1));
    tTol = max(1e-12, duplicateTol / max(L, 1e-14));

    tAdd = choose_new_t_values(tExist, nNeed, tTol);
    for jt = 1:numel(tAdd)
        t = tAdd(jt);
        x = (1-t)*a + t*b;
        d = vecnorm(Xall - x.', 2, 2);
        if isempty(d) || min(d) > duplicateTol
            Xnew(end+1,:) = x.'; %#ok<AGROW>
            panNew(end+1,1) = pid; %#ok<AGROW>
            s0New(end+1,1) = t * L; %#ok<AGROW>
            Xall(end+1,:) = x.'; %#ok<AGROW>
        end
        if size(Xnew,1) >= maxAddAllowed
            break;
        end
    end
end
end

function tAdd = choose_new_t_values(tExist, nNeed, tTol)
tAdd = zeros(0,1);
if nNeed <= 0
    return;
end

tExist = sort(tExist(:));
mExist = numel(tExist);
mTry = max(mExist + nNeed, 1);
maxTry = mExist + 16*nNeed + 32;

while numel(tAdd) < nNeed && mTry <= maxTry
    [xi, ~] = gauss_legendre(mTry);
    tCand = (xi(:) + 1)/2;
    tCand = sort(tCand);
    for k = 1:numel(tCand)
        tk = tCand(k);
        if all(abs(tk - tExist) > tTol) && all(abs(tk - tAdd) > tTol)
            tAdd(end+1,1) = tk; %#ok<AGROW>
            if numel(tAdd) >= nNeed
                break;
            end
        end
    end
    mTry = mTry + nNeed;
end

if numel(tAdd) < nNeed
    M = max(4*nNeed + mExist, 8);
    tEq = ((1:M)'/(M+1));
    for k = 1:numel(tEq)
        tk = tEq(k);
        if all(abs(tk - tExist) > tTol) && all(abs(tk - tAdd) > tTol)
            tAdd(end+1,1) = tk; %#ok<AGROW>
            if numel(tAdd) >= nNeed
                break;
            end
        end
    end
end
end


function [loss, grads, dbg] = modelLossBINN_state(net, op)
[res, dbgCore] = residual_vector_from_state(net, op);
res2 = res.^2;
resUnscaled = res / max(op.eqScale, 1e-14);
res2Unscaled = resUnscaled.^2;

loss = sum(op.wCol_dl .* res2) / op.wColSum;
grads = dlgradient(loss, net.Learnables);
dbg = [dbgCore; mean(res2); mean(res2Unscaled)];
end

function [res, dbgCore] = residual_vector_from_state(net, op)
Nb = op.Nb;
Npan = op.Npan;

sigma_std = forward(net, op.Yq_dl);
sigma_std = sigma_std(:);

sigma_c = forward(net, op.Xc_dl);
sigma_c = sigma_c(:);

if op.useNearPanelRefine
    sigma_ref = forward(net, op.YqR_dl);
    sigma_ref = sigma_ref(:);
end

Vstd = op.A_dl * sigma_std;
V = Vstd;
nearSubMag = dlarray(0.0);
nearAddMag = dlarray(0.0);

if op.useNearPanelRefine
    for k = 1:Nb
        pid = op.pan_of_xc(k);
        im1 = mod(pid-2, Npan) + 1;
        ip1 = mod(pid,   Npan) + 1;

        js1 = op.idxStd{im1};
        js2 = op.idxStd{ip1};

        cStd1 = sum(op.A_dl(k, js1).' .* sigma_std(js1));
        cStd2 = sum(op.A_dl(k, js2).' .* sigma_std(js2));
        V(k) = V(k) - cStd1 - cStd2;
        nearSubMag = nearSubMag + abs(cStd1) + abs(cStd2);

        xk = op.Xc_dl(:,k);

        jr1 = op.idxRef{im1};
        y1 = op.YqR_dl(:, jr1);
        r1 = vecnorm(y1 - xk, 2, 1);
        r1 = max(r1, 1e-14);
        G1 = -(1/(2*pi)) * log(r1);
        cRef1 = sum(G1(:) .* sigma_ref(jr1) .* op.wqR_dl(jr1));
        V(k) = V(k) + cRef1;
        nearAddMag = nearAddMag + abs(cRef1);

        jr2 = op.idxRef{ip1};
        y2 = op.YqR_dl(:, jr2);
        r2 = vecnorm(y2 - xk, 2, 1);
        r2 = max(r2, 1e-14);
        G2 = -(1/(2*pi)) * log(r2);
        cRef2 = sum(G2(:) .* sigma_ref(jr2) .* op.wqR_dl(jr2));
        V(k) = V(k) + cRef2;
        nearAddMag = nearAddMag + abs(cRef2);
    end
end

Vsig = V + op.corr_dl .* sigma_c;
res = Vsig - op.f_dl;

dbgCore = [
    mean(abs(Vstd));
    mean(abs(op.corr_dl .* sigma_c));
    mean(abs(res));
    nearSubMag / Nb;
    nearAddMag / Nb
];
end

function [net, lossHist, gradHist, info] = run_lbfgs_refinement(net, op, cfg)
[theta, meta] = net_to_vector(net);
[f, g, ~, net] = loss_and_grad_from_theta(theta, net, meta, op);

maxIters = cfg.lbfgsMaxIters;
lossHist = zeros(maxIters,1);
gradHist = zeros(maxIters,1);

S = zeros(numel(theta),0);
Y = zeros(numel(theta),0);
rho = zeros(0,1);

lsFailures = 0;
reason = 'maxIters';
kFinal = 0;
stalled = false;

for k = 1:maxIters
    kFinal = k;
    gNorm = norm(g);
    lossHist(k) = f;
    gradHist(k) = gNorm;

    if ~(isfinite(f) && isfinite(gNorm))
        reason = 'nonFinite';
        fprintf('LBFGS %4d | loss=%.3e | gradNorm=%.3e | alpha=0.0e+00 | stop=%s\n', ...
                k, f, gNorm, reason);
        break;
    end
    if gNorm < cfg.lbfgsGradTol
        reason = 'gradTol';
        fprintf('LBFGS %4d | loss=%.3e | gradNorm=%.3e | alpha=0.0e+00 | stop=%s\n', ...
                k, f, gNorm, reason);
        break;
    end

    if isempty(rho)
        p = -g;
    else
        Hg = lbfgs_two_loop(g, S, Y, rho);
        p = -Hg;
    end
    if p.' * g >= -1e-16
        p = -g;
    end

    alphaStarts = [cfg.lbfgsAlpha0, cfg.lbfgsAlphaFallback(:).'];
    [alpha, thetaNew, fNew, gNew, netNew, accepted] = lbfgs_line_search( ...
        theta, f, g, p, net, meta, op, cfg, alphaStarts);

    if ~accepted
        lsFailures = lsFailures + 1;
        % First recovery attempt: reset LBFGS memory and retry steepest descent.
        S = zeros(numel(theta),0);
        Y = zeros(numel(theta),0);
        rho = zeros(0,1);
        p = -g;
        [alpha, thetaNew, fNew, gNew, netNew, accepted] = lbfgs_line_search( ...
            theta, f, g, p, net, meta, op, cfg, cfg.lbfgsAlphaFallback(:).');
        if ~accepted
            stalled = true;
            reason = 'near-stationary / line-search stalled';
            fprintf('LBFGS %4d | loss=%.3e | gradNorm=%.3e | alpha=0.0e+00 | stop=%s\n', ...
                    k, f, gNorm, reason);
            break;
        end
    end

    s = thetaNew - theta;
    stepNorm = norm(s);
    if mod(k, cfg.lbfgsLogEvery) == 0 || k == 1 || k == maxIters
        fprintf('LBFGS %4d | loss=%.3e | gradNorm=%.3e | alpha=%.2e | stepNorm=%.2e\n', ...
                k, fNew, norm(gNew), alpha, stepNorm);
    end

    if norm(s) <= cfg.lbfgsStepTol * max(1, norm(theta))
        theta = thetaNew;
        net = netNew;
        f = fNew;
        g = gNew;
        lossHist(k) = f;
        gradHist(k) = norm(g);
        if cfg.lbfgsStepTolNeedsGrad && norm(g) >= cfg.lbfgsGradTol
            S = zeros(numel(theta),0);
            Y = zeros(numel(theta),0);
            rho = zeros(0,1);
            fprintf(['LBFGS %4d | loss=%.3e | gradNorm=%.3e | alpha=%.2e | ' ...
                     'small-step detected, resetting memory and continuing\n'], ...
                    k, f, norm(g), alpha);
            continue;
        else
            reason = 'stepTol';
            fprintf('LBFGS %4d | loss=%.3e | gradNorm=%.3e | alpha=%.2e | stop=%s\n', ...
                    k, f, norm(g), alpha, reason);
            break;
        end
    end

    y = gNew - g;
    ys = y.' * s;
    if ys > 1e-12 * max(1, norm(s) * norm(y))
        if size(S,2) == cfg.lbfgsMemory
            S(:,1) = [];
            Y(:,1) = [];
            rho(1) = [];
        end
        S(:,end+1) = s;
        Y(:,end+1) = y;
        rho(end+1) = 1 / ys;
    end

    theta = thetaNew;
    net = netNew;
    f = fNew;
    g = gNew;
end

lossHist = lossHist(1:kFinal);
gradHist = gradHist(1:kFinal);
if stalled && isempty(lossHist)
    kFinal = 0;
end
info = struct('reason', reason, 'iterations', kFinal, 'lineSearchFailures', lsFailures);
end

function Hg = lbfgs_two_loop(g, S, Y, rho)
m = size(S,2);
q = g;
alpha = zeros(m,1);

for i = m:-1:1
    alpha(i) = rho(i) * (S(:,i).' * q);
    q = q - alpha(i) * Y(:,i);
end

if m > 0
    sy = S(:,m).' * Y(:,m);
    yy = Y(:,m).' * Y(:,m);
    gamma = sy / max(yy, 1e-20);
    if ~(isfinite(gamma) && gamma > 0)
        gamma = 1.0;
    end
else
    gamma = 1.0;
end

r = gamma * q;
for i = 1:m
    beta = rho(i) * (Y(:,i).' * r);
    r = r + S(:,i) * (alpha(i) - beta);
end

Hg = r;
end

function [alpha, thetaNew, fNew, gNew, netNew, accepted] = lbfgs_line_search( ...
         theta, f, g, p, net, meta, op, cfg, alphaStarts)
c1 = cfg.lbfgsArmijoC1;
beta = cfg.lbfgsBacktrackBeta;
maxLS = cfg.lbfgsMaxBacktrack;

if isempty(alphaStarts)
    alphaStarts = cfg.lbfgsAlpha0;
end

alpha = alphaStarts(1);
gtp = g.' * p;
if gtp >= 0
    p = -g;
    gtp = g.' * p;
end

accepted = false;
thetaNew = theta;
fNew = f;
gNew = g;
netNew = net;

for iStart = 1:numel(alphaStarts)
    alphaTry = alphaStarts(iStart);
    for ls = 1:maxLS
        thetaTry = theta + alphaTry * p;
        [fTry, gTry, ~, netTry] = loss_and_grad_from_theta(thetaTry, net, meta, op);

        if ~(isfinite(fTry) && all(isfinite(gTry)))
            alphaTry = alphaTry * beta;
            if alphaTry < cfg.lbfgsStepTol
                break;
            end
            continue;
        end

        armijo = fTry <= f + c1 * alphaTry * gtp;
        if armijo
            accepted = true;
            thetaNew = thetaTry;
            fNew = fTry;
            gNew = gTry;
            netNew = netTry;
            alpha = alphaTry;
            break;
        end

        alphaTry = alphaTry * beta;
        if alphaTry < cfg.lbfgsStepTol
            break;
        end
    end
    if accepted
        break;
    end
end

if ~accepted
    alpha = 0;
end
end

function [theta, meta] = net_to_vector(net)
tbl = net.Learnables;
n = height(tbl);

meta.layer = cellstr(string(tbl.Layer));
meta.parameter = cellstr(string(tbl.Parameter));
meta.sz = cell(n,1);
meta.numel = zeros(n,1);
meta.cls = cell(n,1);

parts = cell(n,1);
for i = 1:n
    vi = gather(extractdata(tbl.Value{i}));
    meta.sz{i} = size(vi);
    meta.numel(i) = numel(vi);
    meta.cls{i} = class(vi);
    parts{i} = double(vi(:));
end

theta = vertcat(parts{:});
end

function net = vector_to_net(net, theta, meta)
T = net.Learnables;
offset = 1;
for i = 1:numel(meta.numel)
    ni = meta.numel(i);
    vals = theta(offset:(offset+ni-1));
    arr = reshape(vals, meta.sz{i});
    arr = cast(arr, meta.cls{i});
    T.Value{i} = dlarray(arr);
    offset = offset + ni;
end
net.Learnables = T;
end

function gvec = grads_to_vector(grads, meta)
n = height(grads);
parts = cell(n,1);
for i = 1:n
    gi = grads.Value{i};
    if isempty(gi)
        parts{i} = zeros(meta.numel(i),1);
    else
        gd = gather(extractdata(gi));
        parts{i} = double(gd(:));
    end
end
gvec = vertcat(parts{:});
end

function [f, gvec, dbgNum, netOut] = loss_and_grad_from_theta(theta, netIn, meta, op)
netOut = vector_to_net(netIn, theta, meta);
[lossVal, grads, dbg] = dlfeval(@modelLossBINN_state, netOut, op);
f = double(gather(extractdata(lossVal)));
gvec = grads_to_vector(grads, meta);
dbgNum = gather(extractdata(dbg));
end

function out = reconstruct_interior_solution(P, Yq, wq, sigma, nGrid, u_exact)
xv = linspace(-1,1,nGrid);
yv = linspace(-1,1,nGrid);
[Xg, Yg] = meshgrid(xv, yv);
mask = inpolygon(Xg, Yg, P(:,1), P(:,2));

Ugrid = nan(size(Xg));
Uexgrid = nan(size(Xg));

Xlist = Xg(mask);
Ylist = Yg(mask);
Uvals = zeros(numel(Xlist), 1);
Uex = u_exact(Xlist, Ylist);

for k = 1:numel(Xlist)
    xpt = [Xlist(k); Ylist(k)];
    r = vecnorm(Yq - xpt, 2, 1);
    r = max(r, 1e-14);
    G = -(1/(2*pi)) * log(r);
    Uvals(k) = sum(G(:) .* sigma(:) .* wq(:));
end

Ugrid(mask) = Uvals;
Uexgrid(mask) = Uex;
Egrid = Ugrid - Uexgrid;

err = Uvals - Uex;
relL2 = norm(err,2) / max(norm(Uex,2), 1e-14);
linf = norm(err,inf);

out = struct;
out.xv = xv;
out.yv = yv;
out.Ugrid = Ugrid;
out.Uexgrid = Uexgrid;
out.Egrid = Egrid;
out.relL2 = relL2;
out.linf = linf;
end

function plot_solution_triplet(xv, yv, Ugrid, Uexgrid, Egrid, P, ttl)
subplot(1,3,1);
imagesc(xv, yv, Ugrid); axis image; set(gca,'YDir','normal');
title('u_{num}'); xlabel('x'); ylabel('y'); colorbar; hold on; plot_polygon(P);
apply_image_alpha_from_nan(Ugrid);

subplot(1,3,2);
imagesc(xv, yv, Uexgrid); axis image; set(gca,'YDir','normal');
title('u_{exact}'); xlabel('x'); ylabel('y'); colorbar; hold on; plot_polygon(P);
apply_image_alpha_from_nan(Uexgrid);

subplot(1,3,3);
imagesc(xv, yv, log10(abs(Egrid)+1e-16)); axis image; set(gca,'YDir','normal');
title('log_{10}|error|'); xlabel('x'); ylabel('y'); colorbar; hold on; plot_polygon(P);
apply_image_alpha_from_nan(Egrid);

sgtitle(ttl);
end

function apply_image_alpha_from_nan(M)
h = findobj(gca, 'Type', 'Image');
if isempty(h)
    return;
end
if ismatrix(M)
    set(h(1), 'AlphaData', ~isnan(M));
end
end

function TrefStd = refined_to_standard_projection_matrix(pGL, nSubNear)
[xiStd, ~] = gauss_legendre(pGL);
[xiSub, ~] = gauss_legendre(pGL);

xiRef = zeros(nSubNear*pGL, 1);
k = 1;
for ss = 1:nSubNear
    tA = (ss-1)/nSubNear;
    tB = ss/nSubNear;
    for j = 1:pGL
        t = (xiSub(j)+1)/2;
        tGlobal = (1-t)*tA + t*tB;
        xiRef(k) = 2*tGlobal - 1;
        k = k + 1;
    end
end

TrefStd = barycentric_lagrange_matrix(xiStd(:), xiRef(:));
end

function L = barycentric_lagrange_matrix(xNodes, xEval)
n = numel(xNodes);
m = numel(xEval);

w = barycentric_weights(xNodes);
L = zeros(m, n);

tol = 1e-14;
for k = 1:m
    dx = xEval(k) - xNodes;
    [dmin, idx] = min(abs(dx));
    if dmin < tol
        L(k, idx) = 1;
    else
        tmp = w ./ dx;
        denom = sum(tmp);
        L(k, :) = (tmp / denom).';
    end
end
end

function w = barycentric_weights(x)
n = numel(x);
w = ones(n,1);
for j = 1:n
    for k = [1:j-1, j+1:n]
        w(j) = w(j) / (x(j) - x(k));
    end
end
end

function I2 = self_panel_integral_log_kernel(L, s0)
eps0 = 1e-16;
s0 = max(min(s0, L-eps0), eps0);
Lm = max(L - s0, eps0);
I2 = -(1/(2*pi)) * ( s0*log(s0) + Lm*log(Lm) - L );
end

function s = global_arclength_coordinate(panels, pan_id, s_on_panel)
Nq = numel(pan_id);
Npan = numel(panels);

L = zeros(Npan,1);
for k = 1:Npan
    seg = panels{k};
    L(k) = norm(seg(2,:) - seg(1,:));
end
offset = [0; cumsum(L(1:end-1))];

s = zeros(Nq,1);
for j = 1:Nq
    s(j) = offset(pan_id(j)) + s_on_panel(j);
end
end

function [gNorm, hasBad] = grad_norm_and_finite(grads)
g2 = 0.0;
hasBad = false;
for i = 1:height(grads)
    gi = gather(extractdata(grads.Value{i}));
    if any(~isfinite(gi(:)))
        hasBad = true;
    end
    g2 = g2 + sum(gi(:).^2);
end
gNorm = sqrt(g2);
end

function [x,w] = gauss_legendre(n)
beta = 0.5 ./ sqrt(1 - (2*(1:n-1)).^(-2));
T = diag(beta,1) + diag(beta,-1);
[V,D] = eig(T);
x = diag(D);
[x, idx] = sort(x);
V = V(:,idx);
w = 2*(V(1,:).^2).';
end

function plot_polygon(P)
plot([P(:,1); P(1,1)], [P(:,2); P(1,2)], 'k', 'LineWidth', 1.0);
end
