function [D,X,Err,A,B]= SDL(Y,Dp,Xp,nIter,K,zeta1,zeta2,lam)
    A = eye(size(Dp,2),K);
    B = zeros(K,size(Xp,1));
    D = Dp*A;
    X = zeros(K, size(Y,2));
    Err = zeros(1,nIter);
    
    fprintf('Iteration: ');
    for iter=1:nIter
        fprintf('\b\b\b\b\b%5i',iter);
        Dold = D;
        D = Dp*A;
        X = B*Xp;
        for j=1:size(D,2)
            X(j,:) = 0; A(:,j) = 0; B(j,:) = 0;
            E = Y-D*X;
            xk = D(:,j)'*E;
            thr = lam./abs(xk);
            xkk = sign(xk).*max(0, bsxfun(@minus,abs(xk),thr/2));
            Wx = xkk;
            Wx(Wx ~= 0) = 1;
            
            [~,bb1]= sort(abs(Xp*xkk'),'descend');
            ind1 = bb1(1:zeta2);
            B(j,ind1)= xkk*Xp(ind1,:)'/(Xp(ind1,:)*Xp(ind1,:)');
            X(j,:) = B(j,:)*Xp.*Wx;
            rInd = find(X(j,:));
            
            if (length(rInd)<1)
                [~,ind]= max(sum(Y-Dp*A*X.^2));
                D(:,j)= Y(:,ind)/norm(Y(:,ind));
            else
                % mag = X(j,rInd)*X(j,rInd)';
                % alpha = 0.01; 
                % tmp3 = alpha*D(:,j) + (1-alpha)*(1/mag)*E(:,rInd)*X(j,rInd)';
                tmp3 = E(:,rInd)*X(j,rInd)';
                [~,bb2]= sort(abs(Dp'*tmp3),'descend');
                ind2 = bb2(1:zeta1);
                A(ind2,j)= (Dp(:,ind2)'*Dp(:,ind2)+ 0.01*eye(length(ind2)))\Dp(:,ind2)'*tmp3;
                
                gradient = -2*E(:,rInd)*X(j,rInd)';
                pursuit_direction = gradient - Dp*(Dp'*gradient); % Orthogonal to Dp
                step_size = 0.1 * norm(tmp3) / (norm(pursuit_direction) + eps);
                A(:,j) = A(:,j)./norm(Dp*A(:,j));
                D(:,j) = Dp*A(:,j)+ step_size*pursuit_direction;
                D(:,j) = D(:,j)/norm(D(:,j));
            end
        end
        
        Err(iter) = (sqrt(trace((D-Dold)'*(D-Dold)))/sqrt(trace(Dold'*Dold)));
    end
end