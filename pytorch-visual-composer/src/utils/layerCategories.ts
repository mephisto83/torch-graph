
// Categories
export const defaultLayerCategories: { [category: string]: string[] } = {
    "Input/Output": ["Input", "Output"],
    "Convolutional": ["Conv1d", "Conv2d", "Conv3d", "ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d"],
    "Linear & Embedding": ["Linear", "Bilinear", "Embedding", "EmbeddingBag"],
    "Recurrent": ["RNN", "LSTM", "GRU"],
    "Normalization": ["BatchNorm1d", "BatchNorm2d", "BatchNorm3d", "GroupNorm", "LayerNorm", "InstanceNorm1d", "InstanceNorm2d", "InstanceNorm3d"],
    "Activation": ["ReLU", "LeakyReLU", "ELU", "SELU", "Sigmoid", "Tanh", "Softmax", "LogSoftmax", "Hardtanh", "Hardshrink", "Hardsigmoid", "Hardswish", "Mish", "GELU"],
    "Pooling": ["MaxPool1d", "MaxPool2d", "MaxPool3d", "AvgPool1d", "AvgPool2d", "AvgPool3d", "AdaptiveMaxPool1d", "AdaptiveMaxPool2d", "AdaptiveMaxPool3d", "AdaptiveAvgPool1d", "AdaptiveAvgPool2d", "AdaptiveAvgPool3d"],
    "Dropout & Regularization": ["Dropout", "Dropout2d", "Dropout3d", "AlphaDropout"],
    "Padding": ["ReflectionPad1d", "ReflectionPad2d", "ReflectionPad3d", "ReplicationPad1d", "ReplicationPad2d", "ReplicationPad3d", "ZeroPad2d"],
    "Upsampling & Resizing": ["Upsample", "UpsamplingNearest2d", "UpsamplingBilinear2d"],
    "Transformers & Attention": ["Transformer", "TransformerEncoder", "TransformerDecoder", "TransformerEncoderLayer", "TransformerDecoderLayer", "MultiheadAttention"],
    "Reshaping & Folding": ["Flatten", "Unfold", "Fold"],
    "Other": ["PixelShuffle", "ChannelShuffle", "Softmax2d"],
    //custom
    "rapstar": ["MLP"],
    "Math": ["Math"],
    "Config": ["Config"], // Added ConfigNode
    "Utilities": ["Cat", "Sequential", "ModuleList", "Squeeze"], // New category for utility operations
};