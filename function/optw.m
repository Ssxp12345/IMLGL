function [f,g] = optw(x,Data)
[d, m] = size(Data.W);
Wplus=reshape(x, d, m);
E = ones(size(Data.Ytrain));
HingeL = max((E - (Data.Ytrain* Data.R) .* (Data.Xtrain * Wplus)) , 0);
term1 = 1/2 * sum(sum(Data.M.*(HingeL.^2)));
term1R = Data.lambda1 * 1/2 * norm(Data.J .* Data.M.*(Data.Ytrain * Data.R - Data.Ytrain), 'fro')^2;
Sabs=0;
for i=1:m
  S=svd(Data.X{i}*Wplus, 'econ');
  Sabs=Sabs+sum(abs(S));
end
term2=Data.lambda*Sabs;
Normgradient=subgradient_nuclearnorm(Data.Xtrain*Data.W);
term3=Data.lambda*trace(Wplus'*Data.Xtrain'*Normgradient);
L = diag(sum(Data.R,2)) - Data.R;
term4 = Data.lambda2 * trace((Data.Xtrain*Wplus)*L*(Data.Xtrain*Wplus)');
f = term1+term2-term3+term4;
gterm1=Data.Xtrain' * (HingeL .* (-(Data.Ytrain * Data.R) .* Data.M));
gterm2=zeros(d, m);
for i=1:m
   gterm2=gterm2+Data.X{i}'*subgradient_nuclearnorm(Data.X{i}*Wplus);  
end
gterm2=Data.lambda*gterm2;
gterm3=Data.lambda*Data.Xtrain'*subgradient_nuclearnorm(Data.Xtrain*Data.W);
gterm4=2*Data.lambda2 * Data.Xtrain'*Data.Xtrain*Wplus* L;
g=gterm1+gterm2-gterm3+gterm4;
g=reshape(g, d*m, 1);
end
