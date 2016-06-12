function model = rank_elm_pairwise_lh_train(X_train, Y_train, Q_train, option)

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


%% Woad training dataset

T=Y_train;

NumberofTrainingData=size(X_train,1);
NumberofInputNeurons=size(X_train,2);

T=double(T);

t1=clock;
t1_cpu=cputime;

%% Calculate weights & biases

[H, InputWeight, BiasofHiddenNeurons] = elm_Hiddenoutput_gen(X_train, NumberofHiddenNeurons, ActivationFunction, seed);

[H, T] = convert2laplacianData(H, T, Q_train);

%% Calculate output weights OutputWeight (beta_i)
n = NumberofHiddenNeurons;

% OutputWeight=((H'*H+(eye(n)/C))\(H'*T)); 

HH = H'*H;
HT = H'*T;

OutputWeight=((HH+(eye(n)/C))\(HT)); 

t2 = clock;
TrainingTime = etime(t2,t1);

t2_cpu=cputime;
TrainingCPUTime = t2_cpu-t1_cpu;

% TrainingTime=toc;
%%%%%%%%%%% Calculate the training accuracy
pred=(H * OutputWeight);

TrainMAP  = compute_map(pred, Y_train, Q_train);
TrainNDCG = compute_ndcg(pred, Y_train, Q_train, metric_type.k_ndcg);

clear H;

model.elm_type = 'rankelm';
model.rank_type = 'pairwise';
model.InputWeight = InputWeight;
model.BiasofHiddenNeurons = BiasofHiddenNeurons;
model.OutputWeight = OutputWeight;
model.ActivationFunction = ActivationFunction;
model.metric_type = metric_type;
model.C = option.C;
model.TrainTime = TrainingTime;
model.TrainCPUTime = TrainingCPUTime;
model.TrainMAP = TrainMAP;
model.TrainNDCG = TrainNDCG;
