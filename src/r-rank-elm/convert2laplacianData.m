function [H, T] = convert2laplacianData(H, T, qids);

for i=1:length(qids)
    qs = qids{i};
    n = length(qs);
    Hi = H(qs,:);
    Ti = T(qs,:);
    LHi = Hi - 1/n*ones(n,1)*(ones(1,n)*Hi);
    LTi = Ti - 1/n*ones(n,1)*(ones(1,n)*Ti);
    
%     meanHi = mean(Hi,1); LHi = Hi - repmat(meanHi, n, 1);
%     meanTi = mean(Ti,1); LTi = Ti - repmat(meanTi, n, 1);
    
    H(qs,:) = LHi;
    T(qs,:) = LTi;
end;


