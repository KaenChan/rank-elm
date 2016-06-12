function model = os_rank_elm_pairwise_lh_train(X_train, Y_train, Q_train, X_vali, Y_vali, Q_vali, option)

%% input option
nHiddenNeurons = option.NumberofHiddenNeurons;
ActivationFunction    = option.ActivationFunction;
metric_type           = option.metric_type;
C                     = 2^option.C;
N0                    = option.N0;
Block                 = option.Block;
verbose = option.verbose;

if isfield(option, 'seed')
    seed = option.seed;
else
    seed = fix(mod(cputime,100));
end

%%%%%%%%%%% Load dataset
%% Load training dataset
P=X_train;
T=Y_train;

P=double(P);
T=double(T);

nInputNeurons=size(P,2);

nQueries=length(Q_train); 

t1 = clock;

%%%%%%%%%%%% Preprocessing T in the case of CLASSIFICATION 

clear temp_T temp_TV_T sorted_target

start_time_train=cputime;

%%%%%%%%%%% step 1 Initialization Phase
idx = [];
sub_qids = {};
if N0>nQueries, N0=nQueries; end
for i=1:N0
    idx = [idx Q_train{i}];
    n_all = length(idx);
    n_last = length(Q_train{i});
    sub_qids{i} = n_all-n_last+1:n_all;
end
P0=P(idx,:); 
T0=T(idx,:);

[H, IW, Bias] = elm_Hiddenoutput_gen(P0, nHiddenNeurons, ActivationFunction, seed);

[H, T0] = convert2laplacianData(H, T0, sub_qids);

% M = pinv(H' * H);
% beta = pinv(H) * T0;
woodbury_choose = 0.5;
if size(H,1) < size(H,2) * woodbury_choose;
    compute_type = 2;
else
    compute_type = 1;
end;

K = H'*H+eye(size(H,2))/C;
if compute_type == 1
    M = inv(K);
elseif compute_type == 2
    M = C*eye(size(H,2))-C*C*H'*((eye(size(H,1))+C*H*H')^-1)*H;
end
beta = M * H'*T0;

clear P0 H T0;

fcount = 0;
tic;
%%%%%%%%%%%%% step 2 Sequential Learning Phase
for n = N0 : Block : nQueries
    if mod(n, Block*5)==0
        model.InputWeight = IW;
        model.BiasofHiddenNeurons = Bias';
        model.ActivationFunction = ActivationFunction;
        model.metric_type = metric_type;
        model.OutputWeight = beta;
        for i=1:length(Q_vali)
            query_lens_t(i) = length(Q_vali{i});
        end
        xt_query_idx = cumsum(query_lens_t);
        idx = []; sub_qids = {};
        l = N0;
        if N0>length(Q_vali), l=length(Q_vali); end
        for i=1:l
            idx = [idx Q_vali{i}];
            n_all = length(idx);
            n_last = length(Q_vali{i});
            sub_qids{i} = n_all-n_last+1:n_all;
        end
%         [~, ~, ValidMAP, ValidNDCG] = os_rank_elm_pairwise_predict(model, X_vali(idx,:), Y_vali(idx), sub_qids, Block);
        [~, ValidTime, ValidMAP, ValidNDCG] = os_rank_elm_pairwise_predict(model, X_vali, Y_vali, Q_vali, Block);
        consumed_time = toc;
        if verbose==0, fprintf(1, repmat('\b',1,fcount)); end %delete line before
        fcount=fprintf('[%d / %d]  Elapsed time is %.2f s. MAP=%.4f NDCG=%.4f', n, nQueries, consumed_time,ValidMAP, ValidNDCG);
        if verbose==1, fprintf('\n'); end %delete line before
        tic;
    end;
    if (n+Block-1) > nQueries
        n_len = nQueries - n;          %%%% correct the block size
    else
        n_len = Block;
    end
    if n_len==0, break; end
    idx = [];
    for i=n:(n+n_len-1)
        idx = [idx Q_train{i}];
    end
    Pn=P(idx,:); Tn=T(idx,:);
    sub_qids = {}; idx = [];
    for i=n:(n+n_len-1)
        idx = [idx Q_train{i}];
        n_all = length(idx);
        n_last = length(Q_train{i});
        sub_qids{i-n+1} = n_all-n_last+1:n_all;
    end

    H = elm_Hiddenoutput_apply(Pn, IW, Bias, ActivationFunction);

    [H, Tn] = convert2laplacianData(H, Tn, sub_qids);
    % M = M - M * H' * (eye(size(H,1)) + H * M * H')^(-1) * H * M; 
    % beta = beta + M * H' * (Tn - H * beta);

    if size(H,1) < size(H,2) * woodbury_choose
        compute_type = 2;
    else
        compute_type = 1;
    end;
    
    K = K + H'*H;
    if compute_type == 1
        M = inv(K);
    elseif compute_type == 2
        M = M - M * H' * (eye(size(H,1)) + H * M * H')^(-1) * H * M; 
    end
    beta = beta + M * H' * (Tn - H * beta);
end

if verbose==0, fprintf(1, repmat('\b',1,fcount)); end %delete line before

t2 = clock;
TrainingTime = etime(t2,t1);

clear Pn Tn H M;

model.TrainTime = TrainingTime;
model.InputWeight = IW;
model.BiasofHiddenNeurons = Bias';
model.OutputWeight = beta;
model.ActivationFunction = ActivationFunction;
model.metric_type = metric_type;
model.Block = Block;
model.C = option.C;

[pred, ~, TrainMAP, TrainNDCG] = os_rank_elm_pairwise_predict(model, X_train, Y_train, Q_train, Block);

model.TrainMAP = TrainMAP;
model.TrainNDCG = TrainNDCG;
