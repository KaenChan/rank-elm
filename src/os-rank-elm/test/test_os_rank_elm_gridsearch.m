function test_os_rank_elm_gridsearch
    
load example_data;

metric_type.name = 'MAP';
metric_type.k_ndcg = 5;

Block = 2;
N0 = Block*2;

% NHiddenNeurons  = [400 600 800];
% Cs = -7:1:4;

NHiddenNeurons  = 100;
Cs = -7;

info = 'Online Sequential Rank elm\n';
info = [info sprintf('metric_type = %s\n', metric_type.name)];
info = [info sprintf('k_ndcg      = %d\n', metric_type.k_ndcg)];
info = [info sprintf('N-hidden    = %s\n', mat2str(NHiddenNeurons))];
info = [info sprintf('C           = %s\n', mat2str(Cs))];
info = [info sprintf('Blocks      = %s\n', mat2str(Block))];
info = [info '\n'];

fprintf(info);

% Read the training and validation data
bestvalid = 0;
     
% seed = rng;
% seed = 0;

for N     = [NHiddenNeurons] % 400 600 800]
for C     = [Cs]
    fprintf('Fold%d N=%d C=2^%-2d B=%-2d ', i, N, C, Block);
    option.NumberofHiddenNeurons = N;
    option.ActivationFunction    = 'sig';
    option.C                     = C;
    option.N0                    = N0;
    option.Block                 = Block;
    option.metric_type           = metric_type;
    option.verbose = 0;

    model = os_rank_elm_pairwise_lh_train(X_train, Y_train, Q_train, X_vali, Y_vali, Q_vali, option);

    TrainMAP = model.TrainMAP;
    TrainNDCG = model.TrainNDCG;

    [~, ValidTime, ValidNDCG, ValidMAP] = os_rank_elm_pairwise_predict(model, X_vali, Y_vali, Q_vali, model.Block);

    fprintf('Time=(%.4f %.4f) MAP=(%.4f %.4f) NDCG=(%.4f %.4f)', model.TrainTime, ValidTime, TrainMAP, ValidMAP, TrainNDCG, ValidNDCG);

    if strcmp(metric_type.name, 'MAP')
        validScore = ValidMAP;
    elseif strcmp(metric_type.name, 'NDCG')
        validScore = ValidNDCG;
    end;

    if bestvalid < validScore,
        bestvalid = validScore;
        bestmodel = model;
        bestmodel.N = N;
        bestmodel.C = C;
        bestmodel.Block = Block;
        bestmodel.Time = model.TrainTime;
        bestmodel.MAP = [TrainMAP ValidMAP];
        bestmodel.NDCG = [TrainNDCG ValidNDCG];
    end;
    fprintf(' | Best Valid N=%d C=2^%d B=%-2d MAP=%.4f NDCG=%.4f\n', bestmodel.N, bestmodel.C, bestmodel.Block, bestmodel.MAP(2), bestmodel.NDCG(2));
end;
end;
% Print predictions and compute the metrics.
model = bestmodel;
[pred, TestTime, TestMAP, TestNDCG] = os_rank_elm_pairwise_predict(model, X_test, Y_test, Q_test, Block);
bestmodel.MAP(3)  = TestMAP;
bestmodel.NDCG(3)  = TestNDCG;
fprintf('\nFold%d The best model: N=%d C=10^%-2d B=%-2d ', i, bestmodel.N, bestmodel.C, bestmodel.Block);
fprintf('TrainTime=%4f; MAP=(%.4f %.4f %.4f) NDCG=(%.4f %.4f %.4f)\n\n', ...
    bestmodel.Time, ...
    bestmodel.MAP(1), bestmodel.MAP(2), bestmodel.MAP(3), ...
    bestmodel.NDCG(1), bestmodel.NDCG(2), bestmodel.NDCG(3));
