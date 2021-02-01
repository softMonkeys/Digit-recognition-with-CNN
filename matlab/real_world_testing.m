layers = get_lenet();
load lenet.mat

path = '../images/q3_3/';
images = dir([path '*.png']);

trueValue = [1, 2, 3, 7, 8];

layers{1}.batch_size = 1;
for i = 1 : size(images, 1)
    img = imread([path images(i).name]);
    
    if (size(img, 3) == 3)
        img = rgb2gray(img);
    end
    
    img = im2double(imresize(img, [28 28]));
    img = reshape(img', [28 * 28 1]);
    
    [output, P] = convnet_forward(params, layers, img);
    [M, ypred] = max(P);
    
    disp('True Class:');
    disp(trueValue(i));
    disp('Predicted Class:');
    disp(ypred);
    disp('----------------');
    
end
