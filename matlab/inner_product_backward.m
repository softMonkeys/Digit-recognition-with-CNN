function [param_grad, input_od] = inner_product_backward(output, input, layer, param)

param_grad.w = (output.diff * input.data')';
param_grad.b = (output.diff * ones(1, input.batch_size)')';
input_od = (output.diff' * param.w')';

end
