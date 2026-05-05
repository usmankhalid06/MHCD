clear; clc; close all;

%% ========================================================================
%  CONFIGURATION
%% ========================================================================
disorder_labels = {'Addiction','ADHD','Anxiety','Bipolar','BPD', ...
                   'Depression','Eating Disorder','Mental Health','PTSD'};
C      = 9;
nAlg   = 6;
algorithm_names = {'Group ICA','Clustering','Sparse ICA','ODL','ACSD','DASDL','DASDL+RPCA'};

model_files = {
    'midCovid_embeddings_MiniLM-L6.mat',
    'midCovid_embeddings_MiniLM-L12.mat',
    'midCovid_embeddings_mpnet.mat',
    'midCovid_embeddings_BGE.mat',
    'midCovid_embeddings_E5.mat',
    'midCovid_embeddings_GTE.mat'
};
dataset_name = 'midCovid';

model_short_names = {'MiniLM-L6','MiniLM-L12','mpnet','BGE','E5','GTE'};
nModels = length(model_files);
% load Dp_midCovid

%% ========================================================================
%  PARAMETERS
%% ========================================================================
nIter             = 30;
clust_params      = [8, 16, 64, 16];

%                    MiniLM-L6  MiniLM-L12  mpnet  BGE   E5   GTE
spa_per_model     = [18, 18, 16, 21, 23, 21];
spa_oca_per_model = [18, 18, 16, 21, 23, 21];
spa_odl_per_model = [ 4,  4,  4,  4,  4,  4];

K                 = 9 * (spa_per_model - 1) / 3;
dasdl_lambda1     = 0.7;
dasdl_lambda2    = 0.7;
dc_remove_models  = [4, 5, 6];   % BGE, E5, GTE


%% ========================================================================
%  STORAGE
%% ========================================================================
all_comorbidity = cell(nAlg, nModels);
all_activation  = cell(nAlg, nModels);
all_D           = cell(nAlg, nModels);
all_X           = cell(nAlg, nModels);
all_Pi          = cell(1, nModels);
all_K           = zeros(nModels, 1);
all_Err         = cell(1, nModels);  % add this before the loop

%% ========================================================================
%  MAIN LOOP  (one pass per embedding model)
%% ========================================================================
for m = 1:nModels
    fprintf('\n============================================================\n');
    fprintf('  MODEL %d/%d: %s\n', m, nModels, model_short_names{m});
    fprintf('============================================================\n');

    data = load(model_files{m});
    Y    = data.Y;
    all_K(m) = K(m);

    % Stack all classes into one matrix, track sample indices per disorder
    Y_all      = [];
    sample_sets = cell(C, 1);
    col_offset  = 0;
    for c = 1:C
        n_c = size(Y{c}, 2);
        Y_all = [Y_all, Y{c}];   %#ok<AGROW>
        sample_sets{c} = (col_offset + 1):(col_offset + n_c);
        col_offset = col_offset + n_c;
    end
    Yc = Y(1:C);

    emb_dim  = size(Y{1}, 1);
    spa_acsd = spa_per_model(m);
    spa_oca  = spa_oca_per_model(m);
    spa_odl  = spa_odl_per_model(m);
   


    %% ------------------------------------------------------------------
    %  Method 1: Group ICA
    fprintf('  Running Group ICA...\n');
    [D{1,m},~,~,XX] = gICA_exp(Yc, 9, 30, 30); %K(m)
    X{1,m} = [XX{1} XX{2} XX{3} XX{4} XX{5} XX{6} XX{7} XX{8} XX{9}];

    %% Method 2: Clustering
    fprintf('  Running Clustering...\n');
    [D{2,m}, X{2,m}] = kmeans_clustering(Y_all, clust_params(1), ...
            clust_params(2), clust_params(3), clust_params(4));

    %% Method 3: Sparse ICA
    fprintf('  Running Sparse ICA (spa=%.4f)...\n', spa_oca/emb_dim);
    [D{3,m}, X{3,m}, ~, ~] = LSICA(Y_all, K(m), spa_oca/emb_dim, nIter);

    %% Method 4: ODL
    fprintf('  Running ODL (spa=%d)...\n', spa_odl);
    [D{4,m}, X{4,m}, ~] = my_ODL(Y_all, K(m), nIter, spa_odl, nIter);

    % Dp = [[D{4,1};zeros(384,51)],[D{4,2};zeros(384,51)],D{4,3},D{4,4},D{4,5},D{4,6}]

    %% Method 5: ACSD
    fprintf('  Running ACSD (spa=%d)...\n', spa_acsd);
    [D{5,m}, X{5,m}, ~] = my_ACSD(Y_all, K(m), spa_acsd, nIter);

    %% Method 6: DASDL  (raw Pi-weighted comorbidity — RPCA applied after loop)
    fprintf('  Running DASDL (lambda2=%.2f, lambda5=%.2f)...\n', ...
        dasdl_lambda1, dasdl_lambda2);
    X{6,m} = zeros(K(m), size(Y_all,2));
    D{6,m}= [];
    Pi      = ones(K(m), C) / C;
    [D{6,m}, X{6,m}, Pi, all_Err{m}] = DASDL(Y_all, sample_sets, K(m), ...
            spa_acsd, nIter, dasdl_lambda1, dasdl_lambda2);

    %% Remove DC row 1 for BGE / E5 / GTE
    if ismember(m, dc_remove_models)
        fprintf('    Removing DC row 1 for %s\n', model_short_names{m});
        for k = 1:5
            if size(X{k,m},1) >= 1, X{k,m}(1,:) = []; end
        end
        if size(X{6,m},1) >= 1
            X{6,m}(1,:) = [];
            Pi(1,:)      = [];
        end
    end
    all_Pi{m} = Pi;

    %% Comorbidity: Methods 1-5  (X-based cosine similarity)
    for k = 1:5
        act = zeros(C, size(X{k,m},1));
        for c = 1:C
            act(c,:) = mean(abs(X{k,m}(:, sample_sets{c})), 2)';
        end
        act_norm = act ./ (vecnorm(act, 2, 2) + 1e-10);
        comorb   = act_norm * act_norm';
        comorb(eye(C)==1) = 0;
        all_comorbidity{k,m} = comorb;
        all_activation{k,m}  = act;
        all_D{k,m}           = D{k,m};
        all_X{k,m}           = X{k,m};
    end

    %% Comorbidity: Method 6 — DASDL Pi-weighted
    %  Pi(k,c) encodes how strongly atom k is associated with disorder c.
    %  Comorbidity(c1,c2) = sum_k Pi(k,c1)*Pi(k,c2)*act(c1,k)*act(c2,k)
    %  then symmetrise and normalise.
    K_eff     = size(X{6,m}, 1);
    act_dasdl = zeros(C, K_eff);
    for c = 1:C
        act_dasdl(c,:) = mean(abs(X{6,m}(:, sample_sets{c})), 2)';
    end

    comorb_dasdl = zeros(C, C);
    for c1 = 1:C
        for c2 = 1:C
            if c1 == c2, continue; end
            % Pi-weighted: only atoms active in BOTH disorders contribute
            weights = Pi(:,c1) .* Pi(:,c2) .* act_dasdl(c1,:)' .* act_dasdl(c2,:)';
            comorb_dasdl(c1,c2) = sum(weights);
        end
    end
    comorb_dasdl = (comorb_dasdl + comorb_dasdl') / 2;
    comorb_dasdl(eye(C)==1) = 0;
    mx = max(comorb_dasdl(:));
    if mx > 1e-10, comorb_dasdl = comorb_dasdl / mx; end

    all_comorbidity{6,m} = comorb_dasdl;
    all_activation{6,m}  = act_dasdl;
    all_D{6,m}           = D{6,m};
    all_X{6,m}           = X{6,m};

    fprintf('  Model %s complete.\n', model_short_names{m});

    for i =1:6
        percZeros(i,m) = 100 * nnz(~X{i,m}) / numel(X{i,m});
    end
    percZeros

end



%% ========================================================================
%  PLOT: all 6 methods per embedding model
%  To redo plots later without rerunning algorithms:
%    load('data_seasonal2019_orig.mat'); then run from here down.
%% ========================================================================
for m = 1:nModels
    fig = figure('Position', [50, 50, 2100, 820]);
    for k = 1:nAlg
        subplot_tight(2, 3, k, [0.10, 0.06]);
        mat = all_comorbidity{k,m};
        imagesc(mat); colorbar; colormap(gca,'parula'); caxis([0 max(mat(:))+1e-10]);
        title(algorithm_names{k}, 'FontSize',11,'FontWeight','bold');
        set(gca,'XTick',1:C,'YTick',1:C,...
            'XTickLabel',disorder_labels,'YTickLabel',disorder_labels,...
            'XTickLabelRotation',45,'FontSize',8);
        xlabel('Disorders'); ylabel('Disorders'); axis square;
        for i=1:C; for j=1:C; if i~=j
            val=mat(i,j); tc='k'; if val>max(mat(:))*0.6, tc='w'; end
            text(j,i,sprintf('%.2f',val),'HorizontalAlignment','center','Color',tc,'FontSize',6);
        end; end; end
    end
    sgtitle(sprintf('Comorbidity: %s  |  %s', dataset_name, model_short_names{m}),...
        'FontSize',14,'FontWeight','bold');
    fname = sprintf('khali1_%s_%s.png', dataset_name, model_short_names{m});
    exportgraphics(fig, fname, 'Resolution',300);
    fprintf('Saved %s\n', fname);
    drawnow;
end





%% ========================================================================
%  POST-LOOP FIGURE 1: ACSD across all 6 embedding models (2x3)
%% ========================================================================
fig_acsd = figure('Position', [50, 50, 1800, 1000]);
for m = 1:nModels
    subplot_tight(2, 3, m, [0.10, 0.06]);
    mat = all_comorbidity{5,m};
    imagesc(mat); colorbar; colormap(gca,'parula'); caxis([0 max(mat(:))+1e-10]);
    title(sprintf('ACSD: %s', model_short_names{m}), 'FontSize',11,'FontWeight','bold');
    set(gca,'XTick',1:C,'XTickLabel',disorder_labels,'XTickLabelRotation',45,...
        'YTick',1:C,'YTickLabel',disorder_labels,'FontSize',8);
    xlabel('Disorders'); ylabel('Disorders'); axis square;
    for i=1:C; for j=1:C; if i~=j
        val=mat(i,j); tc='k'; if val>max(mat(:))*0.6, tc='w'; end
        text(j,i,sprintf('%.2f',val),'HorizontalAlignment','center','FontSize',6,'Color',tc);
    end; end; end
end
sgtitle(sprintf('ACSD Comorbidity — %s (6 Embedding Models)', dataset_name),...
    'FontSize',14,'FontWeight','bold');
drawnow;
exportgraphics(fig_acsd, sprintf('khali1_acsd_%s.png', dataset_name), 'Resolution',300);
fprintf('Saved ACSD all-models figure\n');




%% ========================================================================
%  POST-LOOP FIGURE 2: DASDL (raw) across all 6 embedding models (2x3)
%% ========================================================================
fig_dasdl = figure('Position', [100, 100, 1800, 1000]);
for m = 1:nModels
    subplot_tight(2, 3, m, [0.10, 0.06]);
    mat = all_comorbidity{6,m};
    imagesc(mat); colorbar; colormap(gca,'parula'); caxis([0 max(mat(:))+1e-10]);
    title(sprintf('DASDL: %s', model_short_names{m}), 'FontSize',11,'FontWeight','bold');
    set(gca,'XTick',1:C,'XTickLabel',disorder_labels,'XTickLabelRotation',45,...
        'YTick',1:C,'YTickLabel',disorder_labels,'FontSize',8);
    xlabel('Disorders'); ylabel('Disorders'); axis square;
    for i=1:C; for j=1:C; if i~=j
        val=mat(i,j); tc='k'; if val>max(mat(:))*0.6, tc='w'; end
        text(j,i,sprintf('%.2f',val),'HorizontalAlignment','center','FontSize',6,'Color',tc);
    end; end; end
end
sgtitle(sprintf('DASDL Comorbidity — %s (6 Embedding Models)', dataset_name),...
    'FontSize',14,'FontWeight','bold');
drawnow;
exportgraphics(fig_dasdl, sprintf('khali1_dasdl_%s.png', dataset_name), 'Resolution',300);
fprintf('Saved DASDL all-models figure\n');





%% ========================================================================
%  COMPUTE: ACSD+Avg and DASDL+RPCA
%% ========================================================================

% ACSD+Avg: normalise each model then average
acsd_stack = zeros(C, C, nModels);
for m = 1:nModels
    a = all_comorbidity{5,m}; a(eye(C)==1) = 0;
    acsd_stack(:,:,m) = a / (max(a(:)) + 1e-10);
end
acsd_avg = mean(acsd_stack, 3); acsd_avg(eye(C)==1) = 0;

% DASDL+RPCA: stack raw DASDL, apply RobustPCA, use L
fprintf('\nRunning DASDL+RPCA across all %d models...\n', nModels);
M_dasdl = zeros(C*C, nModels);
for m = 1:nModels
    mat = all_comorbidity{6,m}; mat(eye(C)==1) = 0;
    mx = max(mat(:)); if mx > 1e-10, mat = mat/mx; end
    M_dasdl(:,m) = mat(:);
end
lambda = 0.05;
[L_dasdl, S_dasdl] = RobustPCA(M_dasdl, 2*lambda, lambda);

dasdl_rpca_avg = zeros(C, C);
for m = 1:nModels
    L_mat = reshape(L_dasdl(:,m), C, C);
    L_mat = (L_mat + L_mat') / 2;
    L_mat(eye(C)==1) = 0;
    L_mat = max(L_mat, 0);
    L_mat = L_mat / (max(L_mat(:)) + 1e-10);
    all_comorbidity{7,m} = L_mat;
    dasdl_rpca_avg = dasdl_rpca_avg + L_mat;
end
dasdl_rpca_avg = dasdl_rpca_avg / nModels;
dasdl_rpca_avg(eye(C)==1) = 0;

fprintf('DASDL+RPCA done.\n');
for m = 1:nModels
    fprintf('  %s: ||S||_F = %.4f\n', model_short_names{m}, norm(S_dasdl(:,m)));
end

%% ========================================================================
%  POST-LOOP FIGURE 3: ACSD+Avg vs DASDL+RPCA consensus (2 panels)
%% ========================================================================
fig_consensus = figure('Position', [150, 150, 1400, 580]);

subplot_tight(1, 2, 1, [0.10, 0.06]);
imagesc(acsd_avg); colorbar; colormap(gca,'parula'); caxis([0 1]);
set(gca,'XTick',1:C,'XTickLabel',disorder_labels,'XTickLabelRotation',45,...
    'YTick',1:C,'YTickLabel',disorder_labels,'FontSize',9);
title('ACSD+Avg — Consensus (6 Models)','FontWeight','bold','FontSize',12);
xlabel('Disorders'); ylabel('Disorders'); axis square;
for i=1:C; for j=1:C; if i~=j
    val=acsd_avg(i,j); tc='k'; if val>0.5, tc='w'; end
    text(j,i,sprintf('%.2f',val),'HorizontalAlignment','center','FontSize',7,'Color',tc);
end; end; end

subplot_tight(1, 2, 2, [0.10, 0.06]);
imagesc(dasdl_rpca_avg); colorbar; colormap(gca,'parula'); caxis([0 1]);
set(gca,'XTick',1:C,'XTickLabel',disorder_labels,'XTickLabelRotation',45,...
    'YTick',1:C,'YTickLabel',disorder_labels,'FontSize',9);
title('DASDL+RPCA — Consensus (6 Models)','FontWeight','bold','FontSize',12);
xlabel('Disorders'); ylabel('Disorders'); axis square;
for i=1:C; for j=1:C; if i~=j
    val=dasdl_rpca_avg(i,j); tc='k'; if val>0.5, tc='w'; end
    text(j,i,sprintf('%.2f',val),'HorizontalAlignment','center','FontSize',7,'Color',tc);
end; end; end

sgtitle(sprintf('Consensus Comorbidity: ACSD+Avg vs DASDL+RPCA — %s', dataset_name),...
    'FontSize',14,'FontWeight','bold');
drawnow;
exportgraphics(fig_consensus, sprintf('khali1_consensus_%s.png', dataset_name), 'Resolution',300);
fprintf('Saved consensus figure\n');

%% ========================================================================
%  Contrast ratio report
%% ========================================================================
mask_tri = triu(true(C), 1);
fprintf('\n=== Contrast Ratio (Q75/Q25) ===\n');
fprintf('%-12s  %8s  %8s\n', 'Model', 'ACSD', 'DASDL+RPCA');
fprintf('%s\n', repmat('-', 1, 40));
ratio = zeros(2, nModels);
for m = 1:nModels
    for ki = [5, 7]
        vals = all_comorbidity{ki,m}(mask_tri);
        q75  = quantile(vals, 0.75); q25 = quantile(vals, 0.25);
        idx  = ki - 4;
        ratio(idx,m) = mean(vals(vals>=q75)) / (mean(vals(vals<=q25)) + 1e-10);
    end
    fprintf('%-12s  %8.2f  %8.2f\n', model_short_names{m}, ratio(1,m), ratio(2,m));
end
fprintf('%s\n', repmat('-', 1, 40));
fprintf('%-12s  %8.2f  %8.2f\n', 'MEAN', mean(ratio(1,:)), mean(ratio(2,:)));

%% ========================================================================
%  Cross-model agreement (ACSD)
%% ========================================================================
fprintf('\n=== Cross-Model Agreement (ACSD) ===\n');
for i = 1:nModels
    for j = i+1:nModels
        r = corr(all_comorbidity{5,i}(mask_tri), all_comorbidity{5,j}(mask_tri));
        fprintf('  %s vs %s: %.3f\n', model_short_names{i}, model_short_names{j}, r);
    end
end

% exportgraphics(gcf, 'khali1.png', 'Resolution', 300);





%% ========================================================================
%  Cross-model agreement: SOCA, ODL, ACSD, DASDL
%% ========================================================================
method_names_sel = {'SOCA', 'ODL', 'ACSD', 'DASDL'};
method_idx_sel   = [3, 4, 5, 6];
mask_tri = triu(true(C), 1);

figure('Position', [50, 50, 1800, 1400]);
for mi = 1:4
    k = method_idx_sel(mi);
    r_matrix = zeros(nModels, nModels);
    for i = 1:nModels
        for j = 1:nModels
            v1 = all_comorbidity{k,i}; v1 = v1(mask_tri);
            v2 = all_comorbidity{k,j}; v2 = v2(mask_tri);
            r_matrix(i,j) = corr(v1, v2);
        end
    end
    off_diag = r_matrix(~eye(nModels, 'logical'));
    fprintf('\n=== Cross-Model Agreement (%s) ===\n', method_names_sel{mi});
    fprintf('%-12s', '');
    for i = 1:nModels, fprintf('%12s', model_short_names{i}); end
    fprintf('\n%s\n', repmat('-', 1, 12 + 12*nModels));
    for i = 1:nModels
        fprintf('%-12s', model_short_names{i});
        for j = 1:nModels, fprintf('%12.3f', r_matrix(i,j)); end
        fprintf('\n');
    end
    fprintf('Mean: %.3f\n', mean(off_diag));

    subplot(2, 2, mi);
    imagesc(r_matrix); colorbar; caxis([0 1]); colormap(gca, 'parula');
    set(gca,'XTick',1:nModels,'XTickLabel',model_short_names,'XTickLabelRotation',45,...
        'YTick',1:nModels,'YTickLabel',model_short_names,'FontSize',9);
    title(sprintf('%s  (mean r=%.3f)', method_names_sel{mi}, mean(off_diag)),...
        'FontWeight','bold','FontSize',11);
    for i=1:nModels; for j=1:nModels
        text(j,i,sprintf('%.2f',r_matrix(i,j)),...
            'HorizontalAlignment','center','FontSize',8,'FontWeight','bold','Color','k');
    end; end
end
sgtitle(sprintf('Cross-Model Agreement — %s', dataset_name),'FontSize',13,'FontWeight','bold');
exportgraphics(gcf, sprintf('cross_model_agreement_%s.png', dataset_name), 'Resolution',300);