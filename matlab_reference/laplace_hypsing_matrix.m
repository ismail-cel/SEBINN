function [T, Tcpx] = laplace_hypsing_matrix(g, args)
    % Integral equation is 1/2 + HLP (?)
    %
    % Very rough code at this point, but looks to be correct
    arguments
        g                       % Grid
        args.star = false       % Star interactions only, WITHOUT CORRECTIONS
    end

    N = g.N;
    op = @(z1, z2, dz1, n2) -imag(n2 .* dz1 ./ (z1-z2).^2) / pi;

    if args.star==false
        %T = ( g.dz(:).' ./(g.z(:).'-g.z(:)).^2 );
        T = op(g.z(:), g.z(:).', g.dz(:).', g.n(:));
        for j=1:N
            T(j,j) = 0;
        end
        Hmat_corr = hypsing_correction_matrix(g);
        T = T + Hmat_corr;
    else
        starind = g.starind;
        mem = numel(starind) * numel(starind{1})^2;
        %T = spalloc(N, N, mem);
        [I, J, K] = deal(zeros(mem, 1));
        p = 0; % Pointer
        for i=1:numel(starind)
            si = starind{i};
            si = si(:);
            block = op(g.z(si), g.z(si).', g.dz(si).', g.n(si));
            n = numel(si);
            block(1:n+1:end) = 0;
            % SLP(si, si) = block;

            % Sparse construction
            Ii = si * ones(size(si))'; Ii = Ii(:);
            Ji = ones(size(si)) * si'; Ji = Ji(:);
            I(p+(1:n^2)) = Ii;
            J(p+(1:n^2)) = Ji;
            K(p+(1:n^2)) = block(:);
            p = p + n^2;
        end
        T = sparse(I, J, K, N, N);
    end
end

