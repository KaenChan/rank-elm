function test_rank_elm_eig_gridsearch

load example_data;

% rank_type = 'pointwise';
rank_type = 'pairwise';
% rank_type = 'odinal_regression';

metric_type.name = 'MAP';
% metric_type.name = 'MSE';
% metric_type.name = 'NDCG';
metric_type.k_ndcg = 5;

NHiddenNeurons  = [400:200:600];
% NHiddenNeurons  = [200:200:2000];
% NHiddenNeurons  = [400:400:5200];
% NHiddenNeurons  = [1000:1000:6000];
% NHiddenNeurons  = [200 400 600 1000 1500 2000 3000 5000 7000];
% NHiddenNeurons  = [600:200:1000];
% NHiddenNeurons  = [400:200:600];
% NHiddenNeurons  = [1000];
elm_type = 'rankelm';

datasize = [size(X_train,1)+size(X_vali,1)+size(X_test,1), size(X_train,2)];

info = '';
info = [info sprintf('datasize    = %s\n', mat2str(datasize))];
info = [info sprintf('elm_type    = %s\n', elm_type)];
info = [info sprintf('rank_type   = %s\n', rank_type)];
info = [info sprintf('metric_type = %s\n', metric_type.name)];
info = [info sprintf('k_ndcg      = %d\n', metric_type.k_ndcg)];
info = [info sprintf('N-hidden    = %s\n', mat2str(NHiddenNeurons))];
info = [info sprintf('\n')];

fprintf(info);

bestmodels = [];
fprintf('-----------------------------------------------------------------------------------------------\n');
fprintf('      | N    | C     | Train Time | %-6s                 || Best | N     | C     | %-6s   |\n', metric_type.name, metric_type.name);
fprintf('-----------------------------------------------------------------------------------------------\n');

i = 1;
bestvalid = 0;
     
% norm_type = 'zscore';
% X_train = query_level_normalize(X_train, Q_train, norm_type);
% X_vali = query_level_normalize(X_vali, Q_train, norm_type);
% X_test = query_level_normalize(X_test, Q_train, norm_type);

for N = NHiddenNeurons
    option.NumberofHiddenNeurons = N;
    option.ActivationFunction    = 'sig';
    option.metric_type           = metric_type;

    if strcmp(rank_type, 'pointwise')
        model = rank_elm_pointwise_eig_train(X_train, Y_train, Q_train, X_vali, Y_vali, Q_vali, option);
    elseif strcmp(rank_type, 'pairwise')
        model = rank_elm_pairwise_lh_eig_train(X_train, Y_train, Q_train, X_vali, Y_vali, Q_vali, option);
    end

    [pred, ~, TestEVAL] = rank_elm_predict(model, X_test, Y_test, Q_test);

    fprintf('Fold%d | %-4d | 2^%-3d | %7.4f s  | (%.4f %.4f %.4f) ', ...
        i, model.N, model.C, model.TrainTime, model.TrainEVAL, model.ValidEVAL, TestEVAL);

    validScore = model.ValidEVAL;

    if bestvalid < validScore,
        bestvalid = validScore;
        bestmodel = model;
        bestmodel.N = model.N;
        bestmodel.C = model.C;
        bestmodel.Time = model.TrainTime;
        bestmodel.EVAL = [model.TrainEVAL, model.ValidEVAL, TestEVAL];
    end;
    fprintf('|| Best | %-4d  | 2^%-3d | %.4f   |\n', bestmodel.N, bestmodel.C, bestvalid);
end;
% Print predictions and compute the metrics.

fprintf('Best%d | %-4d | 2^%-3d | %7.4f s  | (%.4f %.4f %.4f) ||\n', ...
    i, bestmodel.N, bestmodel.C, ...
    bestmodel.Time, ...
    bestmodel.EVAL(1), bestmodel.EVAL(2), bestmodel.EVAL(3));
fprintf('\n');

