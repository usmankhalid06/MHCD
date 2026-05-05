function [D,A]= kmeans_clustering(Y,ic,d,tau1,tau2)
tmpY = Y;
k44 = ic;
label4 = litekmeans_k(tmpY, k44, 5);
indexes4 = cell(k44,1);
for i = 1:k44
    indexes4{i} = find(label4 == i);
end
% the second stage clustering
labels = cell(k44,1);
knm = zeros(k44,1);

for i = 1:k44
    knm(i) = max(floor(length(indexes4{i})/d/d),1);
    labels{i} = litekmeans_k(tmpY(:,indexes4{i}), knm(i), 5);
end
k = sum(knm);

indexes = cell(k,1);
ijindex = 0;
for i = 1:k44
    for jj = 1:knm(i)
        ijindex = ijindex+1;
        tempindex = (labels{i} == jj);
        indexes{ijindex} = (indexes4{i}(tempindex));
    end
end
mean_cluster = zeros(size(tmpY,1),k);

for i = 1:k
%     size_cluster(i) = size((indexes{i}),1);
    mean_cluster(:,i) = mean(tmpY(:,indexes{i}),2);
%     [mean_cluster(:,i),~,~] = svds(tmpY(:,indexes{i}),1);
end

% Dq = rmmissing(mean_cluster,2);
% Dq= Dq*diag(1./sqrt(sum(Dq.*Dq)));
% 
% Xq = Dq'*Y;
% Xq = sign(Xq).*max(0, bsxfun(@minus,abs(Xq),spa/2));  %32
% Xp = Xq; Xp(abs(Xp)~=0)=1;  tw = sum(Xp); %nnz(Xp)/(size(Xp,1)*size(Xp,2));
% ind = [];
% if sum(tw)~=0
%     for i =1:size(Xq,1)
%         w(i,:)=double(Xq(i,:)~=0).*tw;
%         if sum(w(i,:)==1)==0
%             ind = [ind i];
%         end
%     end
%     Dq(:,ind) = [];
% else
%     fprintf('Sum of t_wei is zero');
% end


% Clean and normalize Dq
D = rmmissing(mean_cluster, 2);
D = D * diag(1 ./ sqrt(sum(D .* D)));
% Dq = zscore(Dq); 

% Sparse coding
A = D' * Y;  % Coefficients: K x V_2
A = sign(A) .* max(0, abs(A) - tau1/2);  % Thresholding

% Identify independent contributors
unique_contributors = any(A ~= 0 & sum(A ~= 0, 1) == 1, 2);  % Atoms with unique non-zero coefficients
if any(unique_contributors)
    D = D(:, unique_contributors);  % Keep only contributing atoms
else
    fprintf('No independent contributors found\n');
end


A = D'*Y;
A = sign(A).*max(0, bsxfun(@minus,abs(A),tau2/2));