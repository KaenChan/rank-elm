function [H, InputWeight, BiasofHiddenNeurons] = elm_Hiddenoutput_gen(X, NumberofHiddenNeurons, ActivationFunction, seed, Block)
% Random generate input weights InputWeight (w_i) and biases BiasofHiddenNeurons (b_i) of hidden neurons

if nargin == 4
    rng(seed);
end
if nargin < 5
    Block=0;
end

NumberofInputNeurons=size(X,2);
NumberofTrainingData=size(X,1);
InputWeight=rand(NumberofHiddenNeurons,NumberofInputNeurons)*2-1;
BiasofHiddenNeurons=rand(NumberofHiddenNeurons,1);

if Block==0
    H=InputWeight*X';
else
    H = MatrixMul_Block(InputWeight, X', Block);
end

ind=ones(1,NumberofTrainingData);
BiasMatrix=BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
H=H+BiasMatrix;

%% Calculate hidden neuron output matrix H
switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        H = 1 ./ (1 + exp(-H));
    case {'tanh'}
        %%%%%%%% tanh
        H = tanh(H);
    case {'sin','sine'}
        %%%%%%%% Sine
        H = sin(H);    
    case {'hardlim'}
        %%%%%%%% Hard Limit
        H = double(hardlim(H));
    case {'tribas'}
        %%%%%%%% Triangular basis function
        H = tribas(H);
    case {'rbf','radbas'}
        %%%%%%%% Radial basis function
        H = radbas(H);
        %%%%%%%% More activation functions can be added here                
end
% clear tempH;

H = H';