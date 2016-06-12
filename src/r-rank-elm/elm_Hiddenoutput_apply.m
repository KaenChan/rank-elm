function H = elm_Hiddenoutput_apply(X, InputWeight, BiasofHiddenNeurons, ActivationFunction, Block)
% Random generate input weights InputWeight (w_i) and biases BiasofHiddenNeurons (b_i) of hidden neurons

if nargin < 5
    Block=0;
end

NumberofTestingData=size(X,1);


if Block==0
    H=InputWeight*X';
else
    H = MatrixMul_Block(InputWeight, X', Block);
end          
ind=ones(1,NumberofTestingData);
BiasMatrix=BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
H=H + BiasMatrix;
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
        H = hardlim(H);        
    case {'tribas'}
        %%%%%%%% Triangular basis function
        H = tribas(H);        
    case {'rbf','radbas'}
        %%%%%%%% Radial basis function
        H = radbas(H);        
        %%%%%%%% More activation functions can be added here        
end
H = H';
