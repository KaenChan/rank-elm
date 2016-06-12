function test_incremental_rank_elm
    
load example_data;

elm_type = 'i-rankelm';

% rank_type = 'pointwise';
rank_type = 'pairwise';

metric_type.name = 'MAP';
% metric_type.name = 'NDCG';
metric_type.k_ndcg = 0;

NHiddenNeuron  = 200;
% NHiddenNeuron  = 500;
% NHiddenNeuron  = 3000;

valid_interval = 10;

n_candidates = 1;

stop_delta = 0.1;

% option.seed = 0;

datasize = [size(X_train,1)+size(X_vali,1)+size(X_test,1), size(X_train,2)];

info = '';
info = [info sprintf('datasize     = %s\n', mat2str(datasize))];
info = [info sprintf('elm_type     = %s\n', elm_type)];
info = [info sprintf('rank_type    = %s\n', rank_type)];
info = [info sprintf('metric_type  = %s\n', metric_type.name)];
info = [info sprintf('k_ndcg       = %d\n', metric_type.k_ndcg)];
info = [info sprintf('N-hidden     = %s\n', mat2str(NHiddenNeuron))];
info = [info sprintf('n_candidates = %d\n', n_candidates)];
info = [info sprintf('Valid_interval = %d\n', valid_interval)];
info = [info sprintf('\n')];

fprintf(info);

i = 1;
     
option.NumberofHiddenNeurons = NHiddenNeuron;
option.ActivationFunction    = 'sig';
option.metric_type           = metric_type;
option.n_candidate_nodes = n_candidates;
option.rank_type = rank_type;
option.valid_interval = valid_interval;
option.X_test = X_test;
option.Y_test = Y_test;
option.Q_test = Q_test;
option.stop_delta = stop_delta;
option.fold = i;
option.plot = 0;

model = i_rank_elm_light_train(X_train, Y_train, Q_train, X_vali, Y_vali, Q_vali, option);

fprintf('\n');
fprintf('flod%d best | n=%-4d | traintime=%.4f s | %s (%.4f %.4f %.4f) ||\n', ...
    i, model.N, model.TrainTime, ...
    metric_type.name, model.EVAL(1), model.EVAL(2), model.EVAL(3));
fprintf('\n');
