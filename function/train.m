%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%The MATLAB code of IMLGL
%version 1.0
%Code based on DM2L multi-label learning algorithms.
%Data set: 1. http://mulan.sourceforge.net/datasets-mlc.html
%          2. http://www.uco.es/kdis/mllresources
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function model=train(J,Xtrain, Ytrain, lambda, lambda1, lambda2)
[~, m]=size(Ytrain);
[n, d]=size(Xtrain);
Ytrain=Ytrain.*J;
num_dim = d;
rho = 2^1;
Wt = (Xtrain'*Xtrain + rho*eye(num_dim)) \ (Xtrain'*Ytrain);
Rt = eye(m, m); 
Flag=true;
iteration=1;
A=ones(n,m);
M=eye(n,m);
while Flag && iteration <30
    %update label matrix
    Ybu=sign(Ytrain*Rt).*(A-J);
    Y=Ytrain+Ybu;
    for j=1:m
        u = (numel(find(Y(:,j)==-1))/numel(find(Y(:,j)==1)))^0.5;
        for i=1:n
            if Y(i,j) == -1
                M(i,j) = 1;
            else 
                M(i,j) = u;
            end
        end
    end
    X=cell(m, 1);
    %update label subset
    for i=1:m
        tempindex=find(Y(:, i)>0);
        X{i}=Xtrain(tempindex, :);    
    end
    
    [Wtplus, Rtplus]=optsurrogate(Xtrain, X, Ytrain, J, Wt, Rt, lambda, lambda1, lambda2,M);
    
    if norm(Wtplus-Wt, 'fro') <10^-3
        Flag=false;
    else
       iteration=iteration+1; 
       Wt=Wtplus;
       Rt=Rtplus;
    end  
end
model.W=Wtplus
model.R=Rtplus
end

