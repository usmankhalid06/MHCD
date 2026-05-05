function [D, X, Pi, Err] = DASDL(Y_all, sample_sets, K, rho, nIter, lambda1, lambda2)

    [m, n] = size(Y_all);
    C = length(sample_sets);

    fprintf('\n=== DASDL ===\n');
    fprintf('Data: %d × %d | Disorders: %d | K=%d | rho=%d | nIter=%d\n', ...
            m, n, C, K, rho, nIter);
    fprintf('lambda1=%.4f  lambda2=%.4f\n', lambda1, lambda2);

    %% ---- Initialisation ----
    D   = Y_all(:, 1:K);
    D   = D * diag(1 ./ sqrt(sum(D .* D)));
    X   = zeros(K, n);
    Pi  = ones(K, C) / C;   % uniform start — no prior disorder knowledge
    Err = zeros(1, nIter);

    fprintf('Iteration:     ');

    %% ====================================================================
    %  MAIN LOOP
    %% ====================================================================
    for iter = 1:nIter
        fprintf('\b\b\b\b\b%5i', iter);
        Dold = D;

        for j = 1:K
            X(j, :) = 0;
            E  = Y_all - D * X;
            xk = D(:, j)' * E;

            % Base threshold (same as ACSD)
            base_thr = rho ./ abs(xk);

            % Pi-modulated threshold modifier
            thr_modifier = ones(1, n);
            [max_pi, ~]  = max(Pi(j, :));

            if max_pi > (1/C + 0.02)   % Pi has diverged from uniform
                for c = 1:C
                    S_c = sample_sets{c};
                    % pi_kc > 1/C → modifier < 1 → lower threshold on S_c (fires more)
                    % pi_kc < 1/C → modifier > 1 → raise threshold on S_c (fires less)
                    thr_modifier(S_c) = thr_modifier(S_c) .* ...
                                        (1 - lambda1 * (Pi(j,c) - 1/C));
                end
                % Safety clamp: threshold modifier must stay positive
                thr_modifier = max(thr_modifier, 0.01);
            end

            thr    = base_thr .* thr_modifier;
            xk     = sign(xk) .* max(0, abs(xk) - thr/2);
            X(j,:) = xk;

            % BLOCK 1: Update dictionary atom (identical to ACSD)
            rInd = find(X(j, :));
            if ~isempty(rInd)
                num = E(:, rInd) * X(j, rInd)';
                nrm = norm(num);
                if nrm > 1e-10
                    D(:, j) = num / nrm;
                end
            end
        end

        Pi = update_pi(X, sample_sets, lambda2, K, C, n);

        %% ---- Convergence monitor ----
        Err(iter) = sqrt(trace((D - Dold)' * (D - Dold))) / ...
                    sqrt(trace(Dold' * Dold));

        if mod(iter, 5) == 0 || iter == 1
            pi_sharpness = mean(max(Pi, [], 2));
            n_transdiag  = sum(max(Pi, [], 2) < 0.5);
            recon_err    = norm(Y_all - D * X, 'fro')^2 / n;
            rhorsity     = 100 * nnz(~X) / numel(X);
            fprintf(' | Recon: %.2f | rhorse: %.1f%% | PiSharp: %.3f | Trans: %d', ...
                    recon_err, rhorsity, pi_sharpness, n_transdiag);
        end
        fprintf('\n');
        fprintf('Iteration:     ');
    end
    fprintf('\n');

    atom_energy     = mean(abs(X), 2);
    [~, sort_idx]   = sort(atom_energy, 'descend');
    dc_original_pos = find(sort_idx == 1);
    D  = D(:, sort_idx);
    X  = X(sort_idx, :);
    Pi = Pi(sort_idx, :);
    fprintf('  Atom sort: DC atom moved from position %d → row 1\n', dc_original_pos);

    print_summary(D, X, Pi, Y_all, K, C, n);
end



function Pi = update_pi(X, sample_sets, lambda2, K, C, n)

    lambda2 = max(lambda2, 1e-6);
    Pi_new  = zeros(K, C);

    for k = 1:K
        row_abs        = abs(X(k, :));
        total_activity = sum(row_abs);

        if total_activity < 1e-10
            Pi_new(k, :) = ones(1, C) / C;
            continue;
        end

        mean_overall = total_activity / n;
        log_scores   = zeros(1, C);

        for c = 1:C
            S_c           = sample_sets{c};
            n_c          = length(S_c);  % DD           = length(S_c);  
            mean_on_c     = sum(row_abs(S_c)) / n_c;
            preference    = mean_on_c / mean_overall;
            log_scores(c) = log(max(preference, 1e-10)) / lambda2;
        end

        % Numerically stable softmax
        log_scores   = log_scores - max(log_scores);
        p            = exp(log_scores);
        p            = max(p, 1e-8);
        Pi_new(k, :) = p / sum(p);
    end

    Pi = Pi_new;
end


%% ========================================================================
%  Print final summary
%% ========================================================================
function print_summary(D, X, Pi, Y_all, K, C, n)
    fprintf('\n=== DASDL Complete ===\n');
    fprintf('Reconstruction error : %.4f\n', norm(Y_all - D * X, 'fro')^2 / n);
    fprintf('rhorsity             : %.1f%%\n', 100 * nnz(~X) / numel(X));
    fprintf('Pi sharpness         : %.3f  (%.3f=uniform, 1.0=one-hot)\n', ...
            mean(max(Pi, [], 2)), 1/C);
    fprintf('Transdiagnostic atoms: %d / %d  (max Pi < 0.5)\n', ...
            sum(max(Pi, [], 2) < 0.5), K);

    fprintf('\nAtom Assignments (after energy sort):\n');
    for k = 1:K
        [max_pi, max_c] = max(Pi(k, :));
        if max_pi < 0.5
            [sorted_pi, sorted_c] = sort(Pi(k, :), 'descend');
            fprintf('  Atom %2d: TRANSDIAG  D%d(%.2f) + D%d(%.2f)\n', ...
                    k, sorted_c(1), sorted_pi(1), sorted_c(2), sorted_pi(2));
        else
            fprintf('  Atom %2d: Disorder %d  (pi=%.2f)\n', k, max_c, max_pi);
        end
    end
end