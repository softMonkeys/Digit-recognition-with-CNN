function [output] = pooling_layer_forward(input, layer)

    h_in = input.height;
    w_in = input.width;
    c = input.channel;
    batch_size = input.batch_size;
    k = layer.k;
    pad = layer.pad;
    stride = layer.stride;
    
    h_out = (h_in + 2*pad - k) / stride + 1;
    w_out = (w_in + 2*pad - k) / stride + 1;
    
    
    output.height = h_out;
    output.width = w_out;
    output.channel = c;
    output.batch_size = batch_size;

    output.data = zeros([h_out, w_out, c, batch_size]);
    
    for b = 1 : batch_size
        data = reshape(input.data(:, b), [h_in, w_in, c]);
        data = padarray(data, [pad, pad]);
        
        for h = 1 : h_out
            for w = 1 : w_out
                kernel = data((h - 1) * stride + 1: (h - 1) * stride + k, (w - 1) * stride + 1: (w - 1) * stride + k, :);
                output.data(h, w, :, b) = max(max(kernel));
            end
        end
    end
    
    output.data = reshape(output.data, [h_out * w_out * c, batch_size]);

end

