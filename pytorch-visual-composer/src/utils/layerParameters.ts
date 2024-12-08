export interface LayerParameter {
    name: string;
    type: 'number' | 'text' | 'boolean' | 'select' | 'class';
    default?: any;
    options?: string[];
}

// Comprehensive parameter definitions for all layers
export function layerParameters(): { [key: string]: LayerParameter[] } {
    return {
        // Input/Output
        "Input": [],
        "Output": [],

        "Config": [
            { name: 'param_name', type: 'text', default: 'parameter' },
            { name: 'value', type: 'text', default: '0' }, // Default value as text to allow complex types like tuples
        ],

        "MLP": [
            { name: 'hidden_dim', type: 'number', default: 128 },
            { name: 'layers', type: 'number', default: 3 },
            { name: 'sigmoid_output', type: 'boolean', default: false },
            { name: 'activation', type: 'class', default: "nn.GELU" },
            { name: 'output_dim', type: 'number', default: 1 },
            { name: 'use_residual', type: 'boolean', default: false },
        ],

        // torch.cat
        "Cat": [
            { name: 'dim', type: 'number', default: 1 },
            { name: 'comment', type: 'text', default: '' }, // Optional comment
        ],
        // nn.ModuleList
        "ModuleList": [
            { name: 'modelType', type: 'class', default: '' },
            { name: 'num_layers', type: 'number', default: 3 },
        ],

        "Math": [
            { name: 'a', type: 'number', default: '0' },
            { name: 'b', type: 'number', default: '0' },
            { name: 'operation', type: 'select', default: 'multiply', options: ['multiply', 'divide', 'add', 'subtract'] },
        ],

        // nn.Sequential
        "Sequential": [
            { name: 'name', type: 'text', default: 'Sequential' },
            { name: 'comment', type: 'text', default: '' }, // Optional comment
        ],

        "Squeeze": [
            { name: 'a', type: 'number', default: '0' }
        ],

        // Convolutional
        "Conv1d": [
            { name: 'in_channels', type: 'number', default: 1 },
            { name: 'out_channels', type: 'number', default: 1 },
            { name: 'kernel_size', type: 'text', default: '3' },
            { name: 'stride', type: 'text', default: '1' },
            { name: 'padding', type: 'text', default: '0' },
            { name: 'dilation', type: 'text', default: '1' },
            { name: 'groups', type: 'number', default: 1 },
            { name: 'bias', type: 'boolean', default: true },
            { name: 'padding_mode', type: 'select', default: 'zeros', options: ['zeros', 'reflect', 'replicate', 'circular'] },
        ],
        "Conv2d": [
            { name: 'in_channels', type: 'number', default: 1 },
            { name: 'out_channels', type: 'number', default: 1 },
            { name: 'kernel_size', type: 'text', default: '3' },
            { name: 'stride', type: 'text', default: '1' },
            { name: 'padding', type: 'text', default: '0' },
            { name: 'dilation', type: 'text', default: '1' },
            { name: 'groups', type: 'number', default: 1 },
            { name: 'bias', type: 'boolean', default: true },
            { name: 'padding_mode', type: 'select', default: 'zeros', options: ['zeros', 'reflect', 'replicate', 'circular'] },
        ],
        "Conv3d": [
            { name: 'in_channels', type: 'number', default: 1 },
            { name: 'out_channels', type: 'number', default: 1 },
            { name: 'kernel_size', type: 'text', default: '3' },
            { name: 'stride', type: 'text', default: '1' },
            { name: 'padding', type: 'text', default: '0' },
            { name: 'dilation', type: 'text', default: '1' },
            { name: 'groups', type: 'number', default: 1 },
            { name: 'bias', type: 'boolean', default: true },
            { name: 'padding_mode', type: 'select', default: 'zeros', options: ['zeros', 'reflect', 'replicate', 'circular'] },
        ],
        "ConvTranspose1d": [
            { name: 'in_channels', type: 'number', default: 1 },
            { name: 'out_channels', type: 'number', default: 1 },
            { name: 'kernel_size', type: 'text', default: '3' },
            { name: 'stride', type: 'text', default: '1' },
            { name: 'padding', type: 'text', default: '0' },
            { name: 'output_padding', type: 'text', default: '0' },
            { name: 'groups', type: 'number', default: 1 },
            { name: 'bias', type: 'boolean', default: true },
            { name: 'dilation', type: 'text', default: '1' },
            { name: 'padding_mode', type: 'select', default: 'zeros', options: ['zeros'] }
        ],
        "ConvTranspose2d": [
            { name: 'in_channels', type: 'number', default: 1 },
            { name: 'out_channels', type: 'number', default: 1 },
            { name: 'kernel_size', type: 'text', default: '3' },
            { name: 'stride', type: 'text', default: '1' },
            { name: 'padding', type: 'text', default: '0' },
            { name: 'output_padding', type: 'text', default: '0' },
            { name: 'groups', type: 'number', default: 1 },
            { name: 'bias', type: 'boolean', default: true },
            { name: 'dilation', type: 'text', default: '1' },
            { name: 'padding_mode', type: 'select', default: 'zeros', options: ['zeros'] }
        ],
        "ConvTranspose3d": [
            { name: 'in_channels', type: 'number', default: 1 },
            { name: 'out_channels', type: 'number', default: 1 },
            { name: 'kernel_size', type: 'text', default: '3' },
            { name: 'stride', type: 'text', default: '1' },
            { name: 'padding', type: 'text', default: '0' },
            { name: 'output_padding', type: 'text', default: '0' },
            { name: 'groups', type: 'number', default: 1 },
            { name: 'bias', type: 'boolean', default: true },
            { name: 'dilation', type: 'text', default: '1' },
            { name: 'padding_mode', type: 'select', default: 'zeros', options: ['zeros'] }
        ],

        // Linear & Embedding
        "Linear": [
            { name: 'in_features', type: 'number', default: 128 },
            { name: 'out_features', type: 'number', default: 64 },
            { name: 'bias', type: 'boolean', default: true },
        ],
        "Bilinear": [
            { name: 'in1_features', type: 'number', default: 64 },
            { name: 'in2_features', type: 'number', default: 64 },
            { name: 'out_features', type: 'number', default: 64 },
            { name: 'bias', type: 'boolean', default: true },
        ],
        "Embedding": [
            { name: 'num_embeddings', type: 'number', default: 1000 },
            { name: 'embedding_dim', type: 'number', default: 128 },
            { name: 'padding_idx', type: 'text', default: '' },
            { name: 'max_norm', type: 'text', default: '' },
            { name: 'norm_type', type: 'number', default: 2.0 },
            { name: 'scale_grad_by_freq', type: 'boolean', default: false },
            { name: 'sparse', type: 'boolean', default: false },
        ],
        "EmbeddingBag": [
            { name: 'num_embeddings', type: 'number', default: 1000 },
            { name: 'embedding_dim', type: 'number', default: 128 },
            { name: 'max_norm', type: 'text', default: '' },
            { name: 'norm_type', type: 'number', default: 2.0 },
            { name: 'scale_grad_by_freq', type: 'boolean', default: false },
            { name: 'mode', type: 'select', default: 'mean', options: ['mean', 'sum', 'max'] },
            { name: 'sparse', type: 'boolean', default: false },
            { name: 'include_last_offset', type: 'boolean', default: false },
        ],

        // Recurrent
        "RNN": [
            { name: 'input_size', type: 'number', default: 128 },
            { name: 'hidden_size', type: 'number', default: 64 },
            { name: 'num_layers', type: 'number', default: 1 },
            { name: 'nonlinearity', type: 'select', default: 'tanh', options: ['tanh', 'relu'] },
            { name: 'bias', type: 'boolean', default: true },
            { name: 'batch_first', type: 'boolean', default: false },
            { name: 'dropout', type: 'number', default: 0.0 },
            { name: 'bidirectional', type: 'boolean', default: false },
        ],
        "LSTM": [
            { name: 'input_size', type: 'number', default: 128 },
            { name: 'hidden_size', type: 'number', default: 64 },
            { name: 'num_layers', type: 'number', default: 1 },
            { name: 'bias', type: 'boolean', default: true },
            { name: 'batch_first', type: 'boolean', default: false },
            { name: 'dropout', type: 'number', default: 0.0 },
            { name: 'bidirectional', type: 'boolean', default: false },
            { name: 'proj_size', type: 'number', default: 0 },
        ],
        "GRU": [
            { name: 'input_size', type: 'number', default: 128 },
            { name: 'hidden_size', type: 'number', default: 64 },
            { name: 'num_layers', type: 'number', default: 1 },
            { name: 'bias', type: 'boolean', default: true },
            { name: 'batch_first', type: 'boolean', default: false },
            { name: 'dropout', type: 'number', default: 0.0 },
            { name: 'bidirectional', type: 'boolean', default: false },
        ],

        // Normalization
        "BatchNorm1d": [
            { name: 'num_features', type: 'number', default: 64 },
            { name: 'eps', type: 'number', default: 1e-5 },
            { name: 'momentum', type: 'number', default: 0.1 },
            { name: 'affine', type: 'boolean', default: true },
            { name: 'track_running_stats', type: 'boolean', default: true },
        ],
        "BatchNorm2d": [
            { name: 'num_features', type: 'number', default: 64 },
            { name: 'eps', type: 'number', default: 1e-5 },
            { name: 'momentum', type: 'number', default: 0.1 },
            { name: 'affine', type: 'boolean', default: true },
            { name: 'track_running_stats', type: 'boolean', default: true },
        ],
        "BatchNorm3d": [
            { name: 'num_features', type: 'number', default: 64 },
            { name: 'eps', type: 'number', default: 1e-5 },
            { name: 'momentum', type: 'number', default: 0.1 },
            { name: 'affine', type: 'boolean', default: true },
            { name: 'track_running_stats', type: 'boolean', default: true },
        ],
        "GroupNorm": [
            { name: 'num_groups', type: 'number', default: 32 },
            { name: 'num_channels', type: 'number', default: 64 },
            { name: 'eps', type: 'number', default: 1e-5 },
            { name: 'affine', type: 'boolean', default: true },
        ],
        "LayerNorm": [
            { name: 'normalized_shape', type: 'text', default: '64' },
            { name: 'eps', type: 'number', default: 1e-5 },
            { name: 'elementwise_affine', type: 'boolean', default: true },
        ],
        "InstanceNorm1d": [
            { name: 'num_features', type: 'number', default: 64 },
            { name: 'eps', type: 'number', default: 1e-5 },
            { name: 'momentum', type: 'number', default: 0.1 },
            { name: 'affine', type: 'boolean', default: false },
            { name: 'track_running_stats', type: 'boolean', default: false },
        ],
        "InstanceNorm2d": [
            { name: 'num_features', type: 'number', default: 64 },
            { name: 'eps', type: 'number', default: 1e-5 },
            { name: 'momentum', type: 'number', default: 0.1 },
            { name: 'affine', type: 'boolean', default: false },
            { name: 'track_running_stats', type: 'boolean', default: false },
        ],
        "InstanceNorm3d": [
            { name: 'num_features', type: 'number', default: 64 },
            { name: 'eps', type: 'number', default: 1e-5 },
            { name: 'momentum', type: 'number', default: 0.1 },
            { name: 'affine', type: 'boolean', default: false },
            { name: 'track_running_stats', type: 'boolean', default: false },
        ],

        // Activation
        "ReLU": [
            { name: 'inplace', type: 'boolean', default: false },
        ],
        "LeakyReLU": [
            { name: 'negative_slope', type: 'number', default: 0.01 },
            { name: 'inplace', type: 'boolean', default: false },
        ],
        "ELU": [
            { name: 'alpha', type: 'number', default: 1.0 },
            { name: 'inplace', type: 'boolean', default: false },
        ],
        "SELU": [
            { name: 'inplace', type: 'boolean', default: false },
        ],
        "Sigmoid": [],
        "Tanh": [],
        "Softmax": [
            { name: 'dim', type: 'number', default: 1 }
        ],
        "LogSoftmax": [
            { name: 'dim', type: 'number', default: 1 }
        ],
        "Hardtanh": [
            { name: 'min_val', type: 'number', default: -1 },
            { name: 'max_val', type: 'number', default: 1 },
            { name: 'inplace', type: 'boolean', default: false },
        ],
        "Hardshrink": [
            { name: 'lambda', type: 'number', default: 0.5 }
        ],
        "Hardsigmoid": [
            { name: 'inplace', type: 'boolean', default: false },
        ],
        "Hardswish": [
            { name: 'inplace', type: 'boolean', default: false },
        ],
        "Mish": [
            { name: 'inplace', type: 'boolean', default: false },
        ],
        "GELU": [
            { name: 'approximate', type: 'select', default: 'none', options: ['none', 'tanh'] }
        ],

        // Pooling
        "MaxPool1d": [
            { name: 'kernel_size', type: 'text', default: '2' },
            { name: 'stride', type: 'text', default: '' },
            { name: 'padding', type: 'text', default: '0' },
            { name: 'dilation', type: 'text', default: '1' },
            { name: 'return_indices', type: 'boolean', default: false },
            { name: 'ceil_mode', type: 'boolean', default: false },
        ],
        "MaxPool2d": [
            { name: 'kernel_size', type: 'text', default: '2' },
            { name: 'stride', type: 'text', default: '' },
            { name: 'padding', type: 'text', default: '0' },
            { name: 'dilation', type: 'text', default: '1' },
            { name: 'return_indices', type: 'boolean', default: false },
            { name: 'ceil_mode', type: 'boolean', default: false },
        ],
        "MaxPool3d": [
            { name: 'kernel_size', type: 'text', default: '2' },
            { name: 'stride', type: 'text', default: '' },
            { name: 'padding', type: 'text', default: '0' },
            { name: 'dilation', type: 'text', default: '1' },
            { name: 'return_indices', type: 'boolean', default: false },
            { name: 'ceil_mode', type: 'boolean', default: false },
        ],
        "AvgPool1d": [
            { name: 'kernel_size', type: 'text', default: '2' },
            { name: 'stride', type: 'text', default: '' },
            { name: 'padding', type: 'text', default: '0' },
            { name: 'ceil_mode', type: 'boolean', default: false },
            { name: 'count_include_pad', type: 'boolean', default: true },
        ],
        "AvgPool2d": [
            { name: 'kernel_size', type: 'text', default: '2' },
            { name: 'stride', type: 'text', default: '' },
            { name: 'padding', type: 'text', default: '0' },
            { name: 'ceil_mode', type: 'boolean', default: false },
            { name: 'count_include_pad', type: 'boolean', default: true },
        ],
        "AvgPool3d": [
            { name: 'kernel_size', type: 'text', default: '2' },
            { name: 'stride', type: 'text', default: '' },
            { name: 'padding', type: 'text', default: '0' },
            { name: 'ceil_mode', type: 'boolean', default: false },
            { name: 'count_include_pad', type: 'boolean', default: true },
        ],
        "AdaptiveMaxPool1d": [
            { name: 'output_size', type: 'text', default: '1' },
            { name: 'return_indices', type: 'boolean', default: false },
        ],
        "AdaptiveMaxPool2d": [
            { name: 'output_size', type: 'text', default: '(1,1)' },
            { name: 'return_indices', type: 'boolean', default: false },
        ],
        "AdaptiveMaxPool3d": [
            { name: 'output_size', type: 'text', default: '(1,1,1)' },
            { name: 'return_indices', type: 'boolean', default: false },
        ],
        "AdaptiveAvgPool1d": [
            { name: 'output_size', type: 'text', default: '1' }
        ],
        "AdaptiveAvgPool2d": [
            { name: 'output_size', type: 'text', default: '(1,1)' }
        ],
        "AdaptiveAvgPool3d": [
            { name: 'output_size', type: 'text', default: '(1,1,1)' }
        ],

        // Dropout & Regularization
        "Dropout": [
            { name: 'p', type: 'number', default: 0.5 },
            { name: 'inplace', type: 'boolean', default: false },
        ],
        "Dropout2d": [
            { name: 'p', type: 'number', default: 0.5 },
            { name: 'inplace', type: 'boolean', default: false },
        ],
        "Dropout3d": [
            { name: 'p', type: 'number', default: 0.5 },
            { name: 'inplace', type: 'boolean', default: false },
        ],
        "AlphaDropout": [
            { name: 'p', type: 'number', default: 0.5 },
            { name: 'inplace', type: 'boolean', default: false },
        ],

        // Padding
        "ReflectionPad1d": [
            { name: 'padding', type: 'text', default: '0' }
        ],
        "ReflectionPad2d": [
            { name: 'padding', type: 'text', default: '0' }
        ],
        "ReflectionPad3d": [
            { name: 'padding', type: 'text', default: '0' }
        ],
        "ReplicationPad1d": [
            { name: 'padding', type: 'text', default: '0' }
        ],
        "ReplicationPad2d": [
            { name: 'padding', type: 'text', default: '0' }
        ],
        "ReplicationPad3d": [
            { name: 'padding', type: 'text', default: '0' }
        ],
        "ZeroPad2d": [
            { name: 'padding', type: 'text', default: '0' }
        ],

        // Upsampling & Resizing
        "Upsample": [
            { name: 'size', type: 'text', default: '' },
            { name: 'scale_factor', type: 'text', default: '' },
            { name: 'mode', type: 'select', default: 'nearest', options: ['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear'] },
            { name: 'align_corners', type: 'boolean', default: false },
        ],
        "UpsamplingNearest2d": [
            { name: 'size', type: 'text', default: '' },
            { name: 'scale_factor', type: 'text', default: '' },
        ],
        "UpsamplingBilinear2d": [
            { name: 'size', type: 'text', default: '' },
            { name: 'scale_factor', type: 'text', default: '' },
            { name: 'align_corners', type: 'boolean', default: false },
        ],

        // Transformers & Attention
        "Transformer": [
            { name: 'd_model', type: 'number', default: 512 },
            { name: 'nhead', type: 'number', default: 8 },
            { name: 'num_encoder_layers', type: 'number', default: 6 },
            { name: 'num_decoder_layers', type: 'number', default: 6 },
            { name: 'dim_feedforward', type: 'number', default: 2048 },
            { name: 'dropout', type: 'number', default: 0.1 },
            { name: 'activation', type: 'select', default: 'relu', options: ['relu', 'gelu'] },
        ],
        "TransformerEncoder": [
            { name: 'num_layers', type: 'number', default: 6 },
            // custom_encoder and norm are modules, skip them for now
        ],
        "TransformerDecoder": [
            { name: 'num_layers', type: 'number', default: 6 },
            // custom_decoder and norm omitted
        ],
        "TransformerEncoderLayer": [
            { name: 'd_model', type: 'number', default: 512 },
            { name: 'nhead', type: 'number', default: 8 },
            { name: 'dim_feedforward', type: 'number', default: 2048 },
            { name: 'dropout', type: 'number', default: 0.1 },
            { name: 'activation', type: 'select', default: 'relu', options: ['relu', 'gelu'] },
            { name: 'batch_first', type: 'boolean', default: false },
            { name: 'norm_first', type: 'boolean', default: false },
        ],
        "TransformerDecoderLayer": [
            { name: 'd_model', type: 'number', default: 512 },
            { name: 'nhead', type: 'number', default: 8 },
            { name: 'dim_feedforward', type: 'number', default: 2048 },
            { name: 'dropout', type: 'number', default: 0.1 },
            { name: 'activation', type: 'select', default: 'relu', options: ['relu', 'gelu'] },
            { name: 'batch_first', type: 'boolean', default: false },
            { name: 'norm_first', type: 'boolean', default: false },
        ],
        "MultiheadAttention": [
            { name: 'embed_dim', type: 'number', default: 512 },
            { name: 'num_heads', type: 'number', default: 8 },
            { name: 'dropout', type: 'number', default: 0.0 },
            { name: 'bias', type: 'boolean', default: true },
            { name: 'add_bias_kv', type: 'boolean', default: false },
            { name: 'add_zero_attn', type: 'boolean', default: false },
            { name: 'kdim', type: 'text', default: '' },
            { name: 'vdim', type: 'text', default: '' },
            { name: 'batch_first', type: 'boolean', default: false },
        ],

        // Reshaping & Folding
        "Flatten": [
            { name: 'start_dim', type: 'number', default: 1 },
            { name: 'end_dim', type: 'number', default: -1 },
        ],
        "Unfold": [
            { name: 'kernel_size', type: 'text', default: '3' },
            { name: 'dilation', type: 'text', default: '1' },
            { name: 'padding', type: 'text', default: '0' },
            { name: 'stride', type: 'text', default: '1' },
        ],
        "Fold": [
            { name: 'output_size', type: 'text', default: '(4,4)' },
            { name: 'kernel_size', type: 'text', default: '3' },
            { name: 'dilation', type: 'text', default: '1' },
            { name: 'padding', type: 'text', default: '0' },
            { name: 'stride', type: 'text', default: '1' },
        ],

        // Other
        "PixelShuffle": [
            { name: 'upscale_factor', type: 'number', default: 2 }
        ],
        "ChannelShuffle": [
            { name: 'channels_per_group', type: 'number', default: 2 }
        ],
        "Softmax2d": []
    };
}