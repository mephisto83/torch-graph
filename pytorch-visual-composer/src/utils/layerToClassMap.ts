 
// Map each node type to its corresponding PyTorch nn class name.
// If a layer is not in this map, we will just print a comment.
export const defaultLayerToClassMap: { [key: string]: string } = {

    // Configuration Layer
    "Config": "Config", // Handled separately
    "Math": "Math",

    // Custom
    "MLP": "python.syllable.MLP",

    // New Utility Layers
    "Cat": "", // Handled as a functional operation
    "ModuleList": "nn.ModuleList",
    "Sequential": "nn.Sequential",

    // Input/Output
    "Input": "",    // special handling in forward
    "Output": "",   // handled in forward (final output)

    // Convolutional
    "Conv1d": "nn.Conv1d",
    "Conv2d": "nn.Conv2d",
    "Conv3d": "nn.Conv3d",
    "ConvTranspose1d": "nn.ConvTranspose1d",
    "ConvTranspose2d": "nn.ConvTranspose2d",
    "ConvTranspose3d": "nn.ConvTranspose3d",

    // Linear & Embedding
    "Linear": "nn.Linear",
    "Bilinear": "nn.Bilinear",
    "Embedding": "nn.Embedding",
    "EmbeddingBag": "nn.EmbeddingBag",


    // Recurrent
    "RNN": "nn.RNN",
    "LSTM": "nn.LSTM",
    "GRU": "nn.GRU",

    // Normalization
    "BatchNorm1d": "nn.BatchNorm1d",
    "BatchNorm2d": "nn.BatchNorm2d",
    "BatchNorm3d": "nn.BatchNorm3d",
    "GroupNorm": "nn.GroupNorm",
    "LayerNorm": "nn.LayerNorm",
    "InstanceNorm1d": "nn.InstanceNorm1d",
    "InstanceNorm2d": "nn.InstanceNorm2d",
    "InstanceNorm3d": "nn.InstanceNorm3d",

    // Activation
    "ReLU": "nn.ReLU",
    "LeakyReLU": "nn.LeakyReLU",
    "ELU": "nn.ELU",
    "SELU": "nn.SELU",
    "Sigmoid": "nn.Sigmoid",
    "Tanh": "nn.Tanh",
    "Softmax": "nn.Softmax",
    "LogSoftmax": "nn.LogSoftmax",
    "Hardtanh": "nn.Hardtanh",
    "Hardshrink": "nn.Hardshrink",
    "Hardsigmoid": "nn.Hardsigmoid",
    "Hardswish": "nn.Hardswish",
    "Mish": "nn.Mish",
    "GELU": "nn.GELU",

    // Pooling
    "MaxPool1d": "nn.MaxPool1d",
    "MaxPool2d": "nn.MaxPool2d",
    "MaxPool3d": "nn.MaxPool3d",
    "AvgPool1d": "nn.AvgPool1d",
    "AvgPool2d": "nn.AvgPool2d",
    "AvgPool3d": "nn.AvgPool3d",
    "AdaptiveMaxPool1d": "nn.AdaptiveMaxPool1d",
    "AdaptiveMaxPool2d": "nn.AdaptiveMaxPool2d",
    "AdaptiveMaxPool3d": "nn.AdaptiveMaxPool3d",
    "AdaptiveAvgPool1d": "nn.AdaptiveAvgPool1d",
    "AdaptiveAvgPool2d": "nn.AdaptiveAvgPool2d",
    "AdaptiveAvgPool3d": "nn.AdaptiveAvgPool3d",

    // Dropout & Regularization
    "Dropout": "nn.Dropout",
    "Dropout2d": "nn.Dropout2d",
    "Dropout3d": "nn.Dropout3d",
    "AlphaDropout": "nn.AlphaDropout",

    // Padding
    "ReflectionPad1d": "nn.ReflectionPad1d",
    "ReflectionPad2d": "nn.ReflectionPad2d",
    "ReflectionPad3d": "nn.ReflectionPad3d",
    "ReplicationPad1d": "nn.ReplicationPad1d",
    "ReplicationPad2d": "nn.ReplicationPad2d",
    "ReplicationPad3d": "nn.ReplicationPad3d",
    "ZeroPad2d": "nn.ZeroPad2d",

    // Upsampling & Resizing
    "Upsample": "nn.Upsample",
    "UpsamplingNearest2d": "nn.UpsamplingNearest2d",
    "UpsamplingBilinear2d": "nn.UpsamplingBilinear2d",

    // Transformers & Attention
    "Transformer": "nn.Transformer",
    "TransformerEncoder": "nn.TransformerEncoder", // partial - may need custom modules
    "TransformerDecoder": "nn.TransformerDecoder",
    "TransformerEncoderLayer": "nn.TransformerEncoderLayer",
    "TransformerDecoderLayer": "nn.TransformerDecoderLayer",
    "MultiheadAttention": "nn.MultiheadAttention",

    // Reshaping & Folding
    "Flatten": "nn.Flatten",
    "Unfold": "nn.Unfold",
    "Fold": "nn.Fold",

    // Other
    "PixelShuffle": "nn.PixelShuffle",
    "ChannelShuffle": "", // Not directly in PyTorch nn, may need custom. We'll leave blank or comment.
    "Softmax2d": "nn.Softmax2d"
};
