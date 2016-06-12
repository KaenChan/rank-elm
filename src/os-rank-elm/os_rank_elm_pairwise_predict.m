function [pred, TestingTime, TestMAP, TestNDCG] = os_rank_elm_pairwise_predict(model, X, Y, qids, Block)

    P=X;

    % NumberofTrainingData=size(P,2);
%     nInputNeurons=size(P,2);

    T=Y;

    for i=1:length(qids)
        query_lens(i) = length(qids{i});
    end
    query_idx = cumsum(query_lens);

    % P = single(P);
    % T = single(T);
    P = double(P);
    T = double(T);

    t_start=clock;

%     load rank_elm_model.mat;
    IW = model.InputWeight;
    Bias = model.BiasofHiddenNeurons';
    Wout = model.OutputWeight;
    ActivationFunction = model.ActivationFunction;

    %%%%%%%%%%% Calculate the output of testing input
    pred = zeros(length(T),1);

    N = 0;  N_query=length(qids);
    while N < N_query
        if N == 0 && Block > N_query
            idx1 = 1;
            idx2 = query_idx(N_query);
        elseif N == 0 && Block <= N_query
            idx1 = 1;
            idx2 = query_idx(Block);
        elseif (N+Block) > N_query
            idx1 = query_idx(N)+1;
            idx2 = query_idx(N_query);
            Block = N_query-N;             %%%% correct the block size
        else
            idx1 = query_idx(N)+1;
            idx2 = query_idx(N+Block);
        end
        Pn = P(idx1:idx2,:);
        H_test = elm_Hiddenoutput_apply(Pn, IW, Bias, ActivationFunction);
        predn=(H_test * Wout);
        pred(idx1:idx2) = predn;
        N = N + Block;
    end

    TestMAP=compute_map(pred, Y, qids);
    TestNDCG=compute_ndcg(pred, Y, qids, model.metric_type.k_ndcg);
    
    t_end = clock;
    TestingTime = etime(t_end,t_start);

