function [T,S,U,Err]= LSICA(Y,K,spa,nIter)
 
% Prewhitening
[F, G, ~] = svds(Y, K);                     % Y: N × V
Xs = diag(1 ./ diag(G)) * F' * Y;            % Xs: K × V (spatially whitened)
U = ones(K,K);
S = U'*Xs;

fprintf('Iteration:     ');
for j= 1:nIter
    Uo =U;
    fprintf('\b\b\b\b\b%5i',j);
 
    tmp = U'*Xs;
    S = sign(tmp).*max(0, bsxfun(@minus,abs(tmp),spa/2));
    [F1, ~, G1] = svds(Xs*S',K);
    U = F1*G1';

    Err(j) = (sqrt(trace((U-Uo)'*(U-Uo)))/sqrt(trace(Uo'*Uo)));     
end

T = Y*S';

