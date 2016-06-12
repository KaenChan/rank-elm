function test_os_rank_elm_eig_gridsearch
    
load example_data;

metric_type.name = 'MAP';
metric_type.k_ndcg = 5;

Block = 2;
N0 = Block*2;

% NHiddenNeurons  = [400 600 800];% 1200 1600 1800];

NHiddenNeurons  = [200];

info = 'Online Sequential Rank ELM eig\n';
info = [info sprintf('metric_type = %s\n', metric_type.name)];
info = [info sprintf('k_ndcg      = %d\n', metric_type.k_ndcg)];
info = [info sprintf('N-hidden    = %s\n', mat2str(NHiddenNeurons))];
info = [info sprintf('Block       = %d\n', Block)];
info = [info sprintf('Train size  = %s\n', mat2str(size(X_train)))];
info = [info sprintf('Valid size  = %s\n', mat2str(size(X_vali)))];
info = [info sprintf('Test size   = %s\n', mat2str(size(X_test)))];
info = [info '\n'];

fprintf(info);

seed = rng;

bestmodels = [];
fprintf('------------------------------------------------------------------------------------------------------------------------------------\n');
fprintf('      | N    | C     | Train Time | MAP                    | NDCG                   || Best | N     | C     | ValidMAP | ValidNDCG |\n');
fprintf('------------------------------------------------------------------------------------------------------------------------------------\n');
% Read the training and validation data
i = 0;
bestvalid = 0;
for N = NHiddenNeurons
    option.NumberofHiddenNeurons = N;
    option.ActivationFunction    = 'sig';
    option.N0                    = N0;
    option.Block                 = Block;
    option.metric_type           = metric_type;
    option.verbose               = 1;

    model = os_rank_elm_pairwise_lh_eig_train(X_train, Y_train, Q_train, X_vali, Y_vali, Q_vali, option);

    [~, ~, TestMAP, TestNDCG] = os_rank_elm_pairwise_predict(model, X_test, Y_test, Q_test, Block);

    fprintf('Fold%d | %-4d | 2^%-3d | %7.4f s  | (%.4f %.4f %.4f) | (%.4f %.4f %.4f) ', ...
        i, N, model.C, model.TrainTime, model.TrainMAP, model.ValidMAP, TestMAP, model.TrainNDCG, model.ValidNDCG, TestNDCG);

    if strcmp(metric_type.name, 'MAP')
        validScore = model.ValidMAP;
    elseif strcmp(metric_type.name, 'NDCG')
        validScore = model.ValidNDCG;
    end;

    if bestvalid < validScore,
        bestvalid = validScore;
        bestmodel = model;
        bestmodel.N = N;
        bestmodel.Block = Block;
        bestmodel.C = model.C;
        bestmodel.Time = model.TrainTime;
        bestmodel.MAP = [model.TrainMAP model.ValidMAP TestMAP];
        bestmodel.NDCG = [model.TrainNDCG model.ValidNDCG TestNDCG];
    end;
    fprintf('|| Best | %-4d  | 2^%-3d | %.4f   | %.4f    |\n', bestmodel.N, bestmodel.C, bestmodel.MAP(2), bestmodel.NDCG(2));
end;
% Print predictions and compute the metrics.

fprintf('Best%d | %-4d | 2^%-3d | %7.4f s  | (%.4f %.4f %.4f) | (%.4f %.4f %.4f) ||\n', ...
    i, bestmodel.N, bestmodel.C, ...
    bestmodel.Time, ...
    bestmodel.MAP(1), bestmodel.MAP(2), bestmodel.MAP(3), ...
    bestmodel.NDCG(1), bestmodel.NDCG(2), bestmodel.NDCG(3));
fprintf('\n');
