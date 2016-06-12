function [pred, TestTime, TestEVA] = rank_elm_predict(model, Xt, Yt, qids)

%     load rank_elm_model.mat;
    InputWeight = model.InputWeight;
    BiasofHiddenNeurons = model.BiasofHiddenNeurons;
    OutputWeight = model.OutputWeight;
    ActivationFunction = model.ActivationFunction;

    %%%%%%%%%%% Calculate the output of testing input
    t1=clock;

    H_test = elm_Hiddenoutput_apply(Xt, InputWeight, BiasofHiddenNeurons, ActivationFunction);

    pred=(H_test * OutputWeight);

    if nargin >= 3
        if ~isfield(model, 'metric_type')
            model.metric_type.name = 'MAP';
        end
        TestEVA = compute_metric(pred, Yt, qids, model.metric_type);
    else
        TestEVA=0;
    end

    t2 = clock;
    TestTime = etime(t2,t1);
