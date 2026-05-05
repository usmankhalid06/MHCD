function [D, X, Err] = my_ODL(YY, K, nIter,lambda,nIter2)
    D = YY(:,1:K);D = D*diag(1./sqrt(sum(D.*D))); 
    X = zeros(size(D,2),size(YY,2));
    param.mode = 2;
    param.lambda = lambda;
    fprintf('Iteration:     ');
    for iter =1:nIter
        fprintf('\b\b\b\b\b%5i',iter);
        Y = YY; 
        Dold = D;
        
        % F1 = D'*D; G1 = D'*Y;
        % iiter = 0;
        % Xpp = X;
        % while (iiter < nIter)
        %     iiter = iiter + 1;
        %     for i =1:K
        %         xk = 1.0/F1(i,i) * (G1(i,:) - F1(i,:)*X) + X(i,:);
        %         thr = lambda./abs(xk);
        %         X(i,:) = sign(xk).*max(0, bsxfun(@minus,abs(xk),thr/2));
        %     end
        %     if (norm(X - Xpp, 'fro')/numel(D) < 1e-6)
        %         break;
        %     end
        %     Xpp = X;
        % end

        X = mexLasso(Y, D, param);

        F = X*X';
        E = Y*X';
        Dp = D;
        iter2 = 0;
        while (iter2 < nIter2)
            iter2 = iter2 + 1;
            for k = 1: size(D,2)
                if(F(k,k) ~= 0)
                    tmpD = 1.0/F(k,k) * (E(:,k) - D*F(:, k)) + D(:,k);
                    D(:,k) = tmpD/(max( norm(tmpD,2),1));
                end
            end
            if (norm(D - Dp, 'fro')/numel(D) <1e-9)
                break;
            end
            Dp = D;
        end
        Err(iter)= sqrt(trace((D-Dold)'*(D-Dold)))/sqrt(trace(Dold'*Dold));        
    end

end

