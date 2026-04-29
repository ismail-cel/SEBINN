function Tmat_corr = hypsing_correction_matrix(grid)
    n = grid.order;
    % Assemble
    Tmat_corr = spalloc(grid.N, grid.N, 3*n^2*grid.numpanels);
    for panel_idx = 1:grid.numpanels
        panel = get_panel(grid, panel_idx);
        za = panel.z_edges(1);
        zb = panel.z_edges(2);
        % Self correction
        wHcmp = wHinitZ(panel.z, panel.z, panel.dz, za, zb, true);
        Tmat_corr(n*(panel_idx-1) + (1:n), n*(panel_idx-1) + (1:n)) = wHcmp;
        % Neighbor corrections (from nb_panel to panel)
        for d = [-1 1]
            if isfield(grid, 'open_curve') && grid.open_curve
                nb_panel_idx = panel_idx - d;
                if nb_panel_idx == 0 || nb_panel_idx > grid.numpanels
                    continue
                end
            else
                nb_panel_idx = mod_panel_idx(grid, panel_idx - d);
            end
            nb_panel = get_panel(grid, nb_panel_idx);
            nb_za = nb_panel.z_edges(1);
            nb_zb = nb_panel.z_edges(2);
            wHcmp = wHinitZ(panel.z, nb_panel.z, nb_panel.dz, nb_za, nb_zb, false);
            Tmat_corr(n*(panel_idx-1) + (1:n), n*(nb_panel_idx-1) + (1:n)) = wHcmp;
        end
    end
    Tmat_corr = -imag(grid.n.*Tmat_corr/pi);
end

function wHcmp = wHinitZ(ztg,zsc,wzpsc,a,b,self)
%Calculates local compensation term wHcmp
    ngl = numel(zsc);
    c=(1-(-1).^(1:ngl))./(1:ngl);
    cc=(b-a)/2;
    Tr = @(z) (z-(b+a)/2)/cc;
    ztgtr = Tr(ztg);
    zsctr = Tr(zsc);
    if self
        closetg = (1:ngl).';
    else
        closetg = find(abs(ztgtr) < 2);
    end
    Nc = length(closetg);
    ztgtrc = ztgtr(closetg);
    wHcmpTemp = wzpsc.'./(zsc.'-ztg(closetg)).^2;
    P = zeros(Nc,ngl+1);
    R = zeros(Nc,ngl);
    if self
        wHcmpTemp(1:ngl+1:ngl^2) = 0;
        sgn = ones(Nc,1);
        sgn(imag(ztgtrc) < 0) = -1;
        argAdd = -sgn*pi*1i;
        P(:,1)=argAdd+log((1-ztgtrc)./(-1-ztgtrc));
    else
        P(:,1)=log((1-ztgtrc)./(-1-ztgtrc));
    end
    R(:,1) = -1./(1-ztgtrc)+1./(-1-ztgtrc);
    for k=1:ngl-1
        P(:,k+1)=ztgtrc.*P(:,k)+c(k);
        R(:,k+1) = -1./(1-ztgtrc)+(-1)^k./(-1-ztgtrc) + k*P(:,k);
    end
    V=zsctr.^(0:ngl-1);
    wHcmp = zeros(ngl);
    wHcmp(closetg,:) = (R/V)/cc-wHcmpTemp;
end
