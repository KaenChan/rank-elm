% clear;

load example_data;

% rank_type = 'pointwise';
rank_type = 'pairwise';
metric_type.name = 'NDCG';
% metric_type.name = 'MAP';
metric_type.k_ndcg = 0;

NHiddenNeurons  = [400 600 800];
% NHiddenNeurons  = [200:200:3000];
% NHiddenNeurons  = [1000:1000:8000];
% NHiddenNeurons  = 400;

% Cs = -7:1:4;
Cs = -5;

info = 'Rank-ELM\n';
info = [info sprintf('rank_type    = %s\n', rank_type)];
info = [info sprintf('metric_type = %s\n', metric_type.name)];
info = [info sprintf('k_ndcg      = %d\n', metric_type.k_ndcg)];
info = [info sprintf('N-hidden    = %s\n', mat2str(NHiddenNeurons))];
info = [info sprintf('C           = %s\n', mat2str(Cs))];
info = [info '\n'];

fprintf(info);

bestmodels = [];

bestvalid = 0;
i = 1;

for N = [NHiddenNeurons]
for C = [Cs]
        option.NumberofHiddenNeurons = N;
        option.ActivationFunction    = 'sig';
        option.C                     = C;
        option.metric_type = metric_type;
        option.usegpu = 0;

        strC = sprintf('2^%d', C);
        fprintf('Fold%d N=%d C=%-8s ', i, N, strC);
        if strcmp(rank_type, 'pointwise')
            model = rank_elm_pointwise_train(X_train, Y_train, Q_train, option);
        elseif strcmp(rank_type, 'pairwise')
            model = rank_elm_pairwise_lh_train(X_train, Y_train, Q_train, option);
        end
        TrainMAP = model.TrainMAP;
        TrainNDCG = model.TrainNDCG;
        [pred, ValidTime] = rank_elm_predict(model, X_vali, Y_vali, Q_vali);
        ValidMAP = compute_map(pred, Y_vali, Q_vali);
        ValidNDCG = compute_ndcg(pred, Y_vali, Q_vali,metric_type.k_ndcg);
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
            bestmodel.Time = model.TrainTime;
            bestmodel.MAP = [TrainMAP ValidMAP];
            bestmodel.NDCG = [TrainNDCG ValidNDCG];
        end;
        fprintf('\t|\tBest model N=%d C=2^%d ValidMAP=%.4f ValidNDCG=%.4f\n', bestmodel.N, bestmodel.C, bestmodel.MAP(2), bestmodel.NDCG(2));
end;
end;
% Print predictions and compute the metrics.
model = bestmodel;
pred = rank_elm_predict(model, X_test, Y_test, Q_test);
TestMAP = compute_map(pred, Y_test, Q_test);
TestNDCG = compute_ndcg(pred, Y_test, Q_test,metric_type.k_ndcg);
bestmodel.MAP(3)  = TestMAP;
bestmodel.NDCG(3)  = TestNDCG;
fprintf('\nThe best model: N=%d C=2^%d ', bestmodel.N, bestmodel.C);
fprintf('TrainTime=%7.4f; MAP=(%.4f %.4f %.4f) NDCG=(%.4f %.4f %.4f)\n\n', ...
    bestmodel.Time, ...
    bestmodel.MAP(1), bestmodel.MAP(2), bestmodel.MAP(3), ...
    bestmodel.NDCG(1), bestmodel.NDCG(2), bestmodel.NDCG(3));
