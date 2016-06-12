function model = rank_elm_pointwise_eig_train(X_train, Y_train, Q_train, X_vali, Y_vali, Q_vali, option)

%% input option
NumberofHiddenNeurons = option.NumberofHiddenNeurons;
ActivationFunction    = option.ActivationFunction;
metric_type           = option.metric_type;

if isfield(option, 'dataWeights')
    dataWeights = option.dataWeights;
else
    dataWeights = ones(length(Q_train),1);
    % dataWeights = ones(length(Q_train),1);
    % for i=1:length(Q_train)
    %     dataWeights(i) = dataWeights(i);
    % end
    % dataWeights = dataWeights/mean(dataWeights);
end

%% Woad training dataset
if isfield(option, 'seed')
    seed = option.seed;
else
    seed = fix(mod(cputime,100));
end

% start_time_train=cputime;
t1 = clock;

T=Y_train;

NumberofTrainingData=size(X_train,1);
NumberofInputNeurons=size(X_train,2);

TV.T=Y_vali;

T=double(T);

nt = 0;
I=zeros(1e7,1);
for i=1:length(Q_train)
    qids = Q_train{i};
    n = length(qids);
    I(nt+1:nt+n) = ones(n,1)*dataWeights(i);
    nt = nt+n;
end;
W = I(1:nt);

% ts = unique(Y_train);
% for i=1:length(ts)
%     num_t(i) = sum(Y_train==ts(i));
% end
% 
% num_t = num_t/mean(num_t);
% num_w = 1./(num_t);
% % num_w = 1.1.^num_w;
% % num_w = 1.5.^num_w;
% num_w = 2.^num_w;
% 
% for i=1:length(W)
%     idx = fix(Y_train(i))+1;
%     W(i)=W(i)*num_w(idx);
% end

W = sparse(1:nt,1:nt,W);  
% W = diag(W);

%% Calculate weights & biases
[H, InputWeight, BiasofHiddenNeurons] = elm_Hiddenoutput_gen(X_train, NumberofHiddenNeurons, ActivationFunction, seed);

%% Calculate the output of validation input
H_valid = elm_Hiddenoutput_apply(X_vali, InputWeight, BiasofHiddenNeurons, ActivationFunction);
    
%% Calculate output weights OutputWeight (beta_i)
n = NumberofHiddenNeurons;

tic;
[V D] = eig(H'*W*H);
D = diag(D);
Q = V'*(H'*W*T);
eigtime = toc;

bestvalid = 0;
tic;

CC = -50:1:50;
% CC = randi([-30,20],1,10);

for C = CC
    c = 2^C;
    delta = (D+1/c).^(-1);
    OutputWeight=V*(diag(delta)*Q);
    OutputWeight = real(OutputWeight);

    %%%%%%%%%%% Calculate the output of valid input
    pred=(H_valid * OutputWeight);                       %   TY: the actual output of the validing data

    validScore = compute_metric(pred, Y_vali, Q_vali, metric_type);
%     fprintf('%.4f %f\n', ValidNDCG, c);
    updatebest = 0;
    if strcmp(metric_type.name, 'MSE')
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

model.rank_type = 'pointwise';
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
