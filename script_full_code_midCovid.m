% Generate and save embeddings for midCovid data
clear functions
clear; clc; close all;

%% Python Setup
try
    pyversion('C:\Users\mukhalid\anaconda3\envs\matlab_env\python.exe');
catch ME
    if contains(ME.message, 'Python is loaded')
        warning('Python already loaded.');
    else
        rethrow(ME);
    end
end

%% Configuration
ndata = 12;

cd D:\Codefiles\MATLAB_Codes\2026\Newpaper\Reddit
csv_files = {
    'addiction_post_features_tfidf_256.csv',
    'adhd_post_features_tfidf_256.csv',
    'anxiety_post_features_tfidf_256.csv',
    'bipolarreddit_post_features_tfidf_256.csv',
    'bpd_post_features_tfidf_256.csv',
    'depression_post_features_tfidf_256.csv',
    'EDAnonymous_post_features_tfidf_256.csv',
    'mentalhealth_post_features_tfidf_256.csv',
    'ptsd_post_features_tfidf_256.csv',
    'teaching_post_features_tfidf_256.csv',
    'jokes_post_features_tfidf_256.csv',
    'fitness_post_features_tfidf_256.csv'
};
cd D:\Codefiles\MATLAB_Codes\2026\Newpaper

dataset_labels = {'Addiction', 'ADHD', 'Anxiety', 'Bipolar', 'BPD', ...
                  'Depression', 'Eating Disorder', 'Mental Health', ...
                  'PTSD', 'Teaching', 'Jokes', 'Fitness'};

model_names = {
    'all-MiniLM-L6-v2',
    'all-MiniLM-L12-v2',
    'all-mpnet-base-v2',
    'BAAI/bge-base-en-v1.5',
    'intfloat/e5-base-v2',
    'thenlper/gte-base'
};
model_short_names = {'MiniLM-L6', 'MiniLM-L12', 'mpnet', 'BGE', 'E5', 'GTE'};
nModels = length(model_names);

%% Read CSV files (ONCE)
fprintf('Reading midCovid CSV files...\n');
all_posts = cell(ndata, 1);

for i = 1:ndata
    fprintf('  %s (%d/%d)...\n', dataset_labels{i}, i, ndata);
    
    if exist(csv_files{i}, 'file')
        opts = detectImportOptions(csv_files{i});
        data_table = readtable(csv_files{i}, opts);
        
        start_row = 2;
        end_row = min(1368, height(data_table));
        
        if width(data_table) >= 4
            post_texts = data_table{start_row:end_row, 4};
            valid_idx = ~ismissing(post_texts) & ~cellfun(@isempty, post_texts);
            post_texts = post_texts(valid_idx);
            
            for j = 1:length(post_texts)
                post_texts{j} = clean_reddit_post(post_texts{j});
            end
            
            fprintf('    %d valid posts\n', length(post_texts));
            all_posts{i} = post_texts;
        else
            all_posts{i} = {};
        end
    else
        fprintf('    ERROR: File not found: %s\n', csv_files{i});
        all_posts{i} = {};
    end
end

%% Generate and save embeddings for each model
for m = 1:nModels
    fprintf('\n=== Embedding with %s (%d/%d) ===\n', model_short_names{m}, m, nModels);
    
    Y = cell(ndata, 1);
    for i = 1:ndata
        fprintf('  %s...\n', dataset_labels{i});
        posts = all_posts{i};
        if isempty(posts)
            Y{i} = [];
            continue;
        end
        post_embeddings = get_sentence_embeddings(posts, model_names{m});
        Y{i} = post_embeddings';
        Y{i} = zscore(Y{i});
        fprintf('    Size: %d x %d\n', size(Y{i}));
    end
    
    save_name = sprintf('midCovid_embeddings_%s.mat', model_short_names{m});
    save(save_name, 'Y', '-v7.3');
    fprintf('  Saved: %s\n', save_name);
end

fprintf('\n=== ALL midCovid EMBEDDINGS SAVED ===\n');