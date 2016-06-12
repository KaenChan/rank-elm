function model = rank_elm_pointwise_train(X_train, Y_train, Q_train, option)

%% input option
NumberofHiddenNeurons = option.NumberofHiddenNeurons;
ActivationFunction    = option.ActivationFunction;
metric_type                = option.metric_type;
C = 2^option.C;

if isfield(option, 'seed')
    seed = option.seed;
else
    seed = fix(mod(cputime,100));
end

if isfield(option, 'dataWeights')
    dataWeights = option.dataWeights;
else
    dataWeights = ones(length(Q_train),1);
    % dataWeights = ones(length(Q_train),1);
    % for i=1:length(Q_train)
    %     dataWeights(i) = dataWeights(i)/length(Q_train{i});
    % end
    % dataWeights = dataWeights/mean(dataWeights);
end

%% Woad training dataset

T=Y_train;

NumberofTrainingData=size(X_train,1);
NumberofInputNeurons=size(X_train,2);

T=double(T);

t1=clock;

nt = 0;
I=zeros(1e7,1);
for i=1:length(Q_train)
    qids = Q_train{i};
    n = length(qids);
    I(nt+1:nt+n) = ones(n,1)*dataWeights(i);
    nt = nt+n;
end;
W = I(1:nt);
W = diag(W);

%% Calculate weights & biases
[H, InputWeight, BiasofHiddenNeurons] = elm_Hiddenoutput_gen(X_train, NumberofHiddenNeurons, ActivationFunction, seed);

%% Calculate output weights OutputWeight (beta_i)
n = NumberofHiddenNeurons;

OutputWeight=((H'*W*H+(eye(n)/C))\(H'*W*T)); 

t2 = clock;
TrainingTime = etime(t2,t1);

% TrainingTime=toc;
%%%%%%%%%%% Calculate the training accuracy
pred=(H * OutputWeight);                       %   the actual output of the training data

TrainMAP  = compute_map(pred, Y_train, Q_train);
TrainNDCG = compute_ndcg(pred, Y_train, Q_train, metric_type.k_ndcg);

clear H;

model.InputWeight = InputWeight;
model.BiasofHiddenNeurons = BiasofHiddenNeurons;
model.OutputWeight = OutputWeight;
model.ActivationFunction = ActivationFunction;
model.metric_type = metric_type;
model.C = option.C;
model.TrainTime = TrainingTime;
model.TrainMAP = TrainMAP;
model.TrainNDCG = TrainNDCG;
