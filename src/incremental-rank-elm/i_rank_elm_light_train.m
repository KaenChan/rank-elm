function model = i_rank_elm_light_train(X_train, Y_train, Q_train, X_vali, Y_vali, Q_vali, option)
% incremental rankELM

%% input option
maxNhidden = option.NumberofHiddenNeurons;
n_candidate_nodes = option.n_candidate_nodes;
ActivationFunction = option.ActivationFunction;
metric_type = option.metric_type;
valid_interval = option.valid_interval;
stop_delta = option.stop_delta;

if ~isfield(option, 'rank_type')
    option.rank_type = 'pairwise';
end

if isfield(option, 'Block')
    Block = option.Block;
else
    Block = 0;
end

if isfield(option, 'seed')
    seed = option.seed;
else
    seed = fix(mod(cputime,100));
end
rng(seed);

loss.train.e               = [];
loss.vali.e                = [];
loss.test.e                = [];

T=Y_train;
T=double(T);

NumberofTrainingData=size(X_train,1);
NumberofInputNeurons=size(X_train,2);

t1=clock;
t1_cpu=cputime;

%% 
start_time_train=cputime;

if strcmp(option.rank_type, 'pairwise')
    T = convert2laplacianDataH(T, Q_train);
end

E=T;
bestvalid = 0;
tic;
InputWeight=zeros(maxNhidden, size(X_train,2));
BiasofHiddenNeurons=zeros(maxNhidden, 1);
OutputWeight=zeros(maxNhidden, 1);
I_InputWeight = zeros(1, size(X_train,2), n_candidate_nodes);
I_BiasofHiddenNeurons=zeros(n_candidate_nodes);
I_Hs=zeros(size(X_train,1), n_candidate_nodes);
I_LHs=zeros(size(X_train,1), n_candidate_nodes);
I_OutputWeightsave=zeros(n_candidate_nodes);
pred_mse = zeros(size(Y_train,1),1);
pred_train = zeros(size(Y_train,1),1);
pred_valid = zeros(size(Y_vali,1),1);
pred_test = zeros(size(option.Y_test,1),1);

for k=1:maxNhidden%
	%%
    for j=1:n_candidate_nodes
		[I_H, I_iw, I_Bias] = elm_Hiddenoutput_gen(X_train, 1, ActivationFunction);
        
        I_InputWeight(:,:,j)=I_iw;
        I_BiasofHiddenNeurons(j)=I_Bias;
        I_Hs(:,j)=I_H;

        if strcmp(option.rank_type, 'pairwise')
            I_H = convert2laplacianDataH(I_H, Q_train);
            I_LHs(:,j)=I_H;
        end
        
        I_OutputWeight= (I_H'*E) / (I_H'*I_H);
        I_OutputWeightsave(j)=I_OutputWeight;
        
        Y=(I_H * I_OutputWeight);
        
        yym1(:,j)=E-Y;
        TT1(j)=sqrt(mse(E-Y)); 
    end
    [fe,i]=min(TT1);
    % disp(num2str(fe));
    E=yym1(:,i);
    OutputWeight(k,1)=I_OutputWeightsave(i);
    InputWeight(k,:)=I_InputWeight(:,:,i);
    BiasofHiddenNeurons(k,1)=I_BiasofHiddenNeurons(i);
    % LH_train(:,k) = I_LHs(:,i);

    I_H_vali = elm_Hiddenoutput_apply(X_vali, I_InputWeight(:,:,i), I_BiasofHiddenNeurons(i), ActivationFunction);
    I_H_test = elm_Hiddenoutput_apply(option.X_test, I_InputWeight(:,:,i), I_BiasofHiddenNeurons(i), ActivationFunction);
    
    pred_mse = pred_mse + I_LHs(:,i) * OutputWeight(k);   
    pred_train = pred_train + I_Hs(:,i) * OutputWeight(k);
    pred_valid = pred_valid + I_H_vali * OutputWeight(k);
    pred_test = pred_test + I_H_test * OutputWeight(k);

    %% validation select
    if mod(k,valid_interval)==0
        consumed_time = toc;

        train_mse = mse(T-pred_mse);
%         train_mse = 0;

        TrainEVAL = compute_metric(pred_train, Y_train, Q_train, metric_type);
        ValidEVAL = compute_metric(pred_valid, Y_vali, Q_vali, metric_type);
        TestEVAL = compute_metric(pred_test, option.Y_test, option.Q_test, metric_type);

        validScore = ValidEVAL;
        if bestvalid < validScore,
            bestvalid = validScore;
            bestmodel.NumberofHiddenNeurons = k;
            bestmodel.N = k;
            bestmodel.EVAL = [TrainEVAL, ValidEVAL, TestEVAL];
        end;

        t2=clock;
        TrainingTime=etime(t2,t1);
        fprintf('%.2f s (%.2f s) | N: %d / %d | Train mse: %.4f | %s %.4f - %.4f - %.4f ', ...
            TrainingTime, consumed_time, k, maxNhidden, train_mse, metric_type.name, TrainEVAL, ValidEVAL, TestEVAL);
        fprintf('|| Valid N:%d  %s %.4f %.4f %.4f', ...
            bestmodel.N, metric_type.name, bestmodel.EVAL(1), bestmodel.EVAL(2), bestmodel.EVAL(3));
        fprintf(' |\n');

        loss.train.e(end+1) = TrainEVAL;
        loss.vali.e(end+1) = ValidEVAL;
        loss.test.e(end+1) = TestEVAL;
        if option.plot
            figure(option.fold) ; clf ;
            ki = k/valid_interval;
            plot((1:ki)*valid_interval, loss.train.e, 'k') ; hold on ;
            plot((1:ki)*valid_interval, loss.vali.e, 'b') ;
            plot((1:ki)*valid_interval, loss.test.e, 'r') ;
            h=legend('train','vali','test') ;
            grid on ;
            xlabel('Num of Hidden Neurons') ; ylabel(metric_type.name) ;
            set(h,'color','none') ;
            title(metric_type.name) ;
            drawnow;
        end

        if k >= 100 && validScore < bestvalid-stop_delta, break; end
        tic;
    end
end

bestmodel.InputWeight = InputWeight(1:bestmodel.N,:);
bestmodel.BiasofHiddenNeurons = BiasofHiddenNeurons(1:bestmodel.N);
bestmodel.OutputWeight = OutputWeight(1:bestmodel.N);

t2 = clock;
TrainingTime = etime(t2,t1);

t2_cpu=cputime;
TrainingCPUTime=t2_cpu-t1_cpu;

model = bestmodel;
model.elm_type = 'i-rankelm';
model.n_candidate_nodes = option.n_candidate_nodes;
model.rank_type = option.rank_type;
model.ActivationFunction = ActivationFunction;
model.metric_type = metric_type;
model.TrainTime = TrainingTime;
model.TrainCPUTime = TrainingCPUTime;
model.Block=Block;
model.loss = loss;
