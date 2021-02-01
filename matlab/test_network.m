%% Network defintion
layers = get_lenet();

%% Loading data
fullset = false;
[xtrain, ytrain, xvalidate, yvalidate, xtest, ytest] = load_mnist(fullset);

% load the trained weights
load lenet.mat

%% Testing the network
predit = zeros(1, size(xtest, 2));

for i = 1 : 100 : size(xtest, 2)
    [output, P] = convnet_forward(params, layers, xtest(:, i:i+99));
    [M, predit(i:i+99)] = max(P);
end

confusionchart(confusionmat(ytest, predit))
disp(C)