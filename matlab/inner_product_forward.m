function [output] = inner_product_forward(input, layer, param)

d = size(input.data, 1);
k = size(input.data, 2); % batch size
n = size(param.w, 2);

output.height = input.height;   % height
output.width = input.width;     % width
output.channel = input.channel; % channel
output.batch_size = k;          % batch size
output.data = param.w' * input.data + param.b';

end
