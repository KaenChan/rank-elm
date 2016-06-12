function model = rank_elm_pairwise_lh_eig_train(X_train, Y_train, Q_train, X_vali, Y_vali, Q_vali, option)

%% input option
NumberofHiddenNeurons = option.NumberofHiddenNeurons;
ActivationFunction    = option.ActivationFunction;
metric_type           = option.metric_type;

%% Woad training dataset
% rng(0);

if isfield(option, 'seed')
    seed = option.seed;
else
    seed = fix(mod(cputime,100));
end

% start_time_train=cputime;
t1 = clock;

T=Y_train;
T=double(T);

NumberofTrainingData=size(X_train,1);
NumberofInputNeurons=size(X_train,2);

%% Calculate weights & biases
%% Calculate weights & biases
[H, InputWeight, BiasofHiddenNeurons] = elm_Hiddenoutput_gen(X_train, NumberofHiddenNeurons, ActivationFunction, seed);

[H, T] = convert2laplacianData(H, T, Q_train);

%% Calculate the output of validation input
H_valid = elm_Hiddenoutput_apply(X_vali, InputWeight, BiasofHiddenNeurons, ActivationFunction);
    
%% Calculate output weights OutputWeight (beta_i)
n = NumberofHiddenNeurons;

tic;
HH = H'*H;
HT = H'*T;

[V D] = eig(HH);
D = diag(D);
Q = V'*(HT);
eigtime = toc;

bestvalid = 0;
tic;

CC = -30:1:30;
% CC = randi([-30,20],1,10);

for C = CC
    c = 2^C;
    delta = (D+1/c).^(-1);
    OutputWeight=V*(diag(delta)*Q);
    OutputWeight = real(OutputWeight);

    %%%%%%%%%%% Calculate the output of valid input
    pred = (H_valid * OutputWeight);

    validScore = compute_metric(pred, Y_vali, Q_vali, metric_type);
%     fprintf('%.4f %f\n', ValidNDCG, c);
    updatebest = 0;
    if strcmp(metric_type, 'MSE')
        if bestvalid > validScore || bestvalid==0
            updatebest = 1;
        end
    elseif bestvalid < validScore
        updatebest = 1;
    end
    
    if updatebest == 1
        bestvalid = validScore;
        bestmodel.c = C;
        bestmodel.OutputWeight = OutputWeight;
    end;
end
valid_time=toc;

t2 = clock;
TrainTime = etime(t2,t1);
% TrainTime=cputime - start_time_train;
%% Calculate the training accuracy
pred_train=(H * bestmodel.OutputWeight);
pred_valid=(H_valid * bestmodel.OutputWeight);

TrainEVAL = compute_metric(pred_train, Y_train, Q_train, metric_type);
ValidEVAL = compute_metric(pred_valid, Y_vali, Q_vali, metric_type);

clear H;

model.elm_type = 'rankelm';
model.rank_type = 'pairwise';
model.TrainTime           = TrainTime;
model.InputWeight         = InputWeight;
model.BiasofHiddenNeurons = BiasofHiddenNeurons;
model.OutputWeight        = bestmodel.OutputWeight;
model.C                   = bestmodel.c;
model.N                   = NumberofHiddenNeurons;
model.ActivationFunction  = ActivationFunction;
model.TrainEVAL           = TrainEVAL;
model.ValidEVAL           = ValidEVAL;
model.seed                = seed;
model.metric_type         = metric_type;
