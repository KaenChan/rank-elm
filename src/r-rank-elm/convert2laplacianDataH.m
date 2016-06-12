function H = convert2laplacianDataH(H, qids);

% LH = [];
% for i=1:length(qids)
%     qs = qids{i};
%     n = length(qs);
%     Hi = H(qs,:);
%     LHi = Hi - 1/n*ones(n,1)*(ones(1,n)*Hi);
%     LH = [LH; LHi];
% end;

for i=1:length(qids)
    qs = qids{i};
    n = length(qs);
    Hi = H(qs,:);
    LHi = Hi - 1/n*ones(n,1)*(ones(1,n)*Hi);
    H(qs,:) = LHi;
    % H(qs,:) = H(qs,:) - 1/n*ones(n,1)*(ones(1,n)*H(qs,:));
end;


