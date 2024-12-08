import ConfigNode from "../components/ConfigNode";
import CustomNode from "../components/CustomNode";

export function nodeTypes() {
    return {
        Config: ConfigNode,
        // Basic I/O and structural nodes
        Input: CustomNode,
        Output: CustomNode,

        // Convolutional layers
        Conv1d: CustomNode,
        Conv2d: CustomNode,
        Conv3d: CustomNode,
        ConvTranspose1d: CustomNode,
        ConvTranspose2d: CustomNode,
        ConvTranspose3d: CustomNode,

        // Linear and embedding layers
        Linear: CustomNode,
        Bilinear: CustomNode,
        Embedding: CustomNode,
        EmbeddingBag: CustomNode,

        // Recurrent layers
        RNN: CustomNode,
        LSTM: CustomNode,
        GRU: CustomNode,

        // Normalization layers
        BatchNorm1d: CustomNode,
        BatchNorm2d: CustomNode,
        BatchNorm3d: CustomNode,
        GroupNorm: CustomNode,
        LayerNorm: CustomNode,
        InstanceNorm1d: CustomNode,
        InstanceNorm2d: CustomNode,
        InstanceNorm3d: CustomNode,

        // Activation layers
        ReLU: CustomNode,
        LeakyReLU: CustomNode,
        ELU: CustomNode,
        SELU: CustomNode,
        Sigmoid: CustomNode,
        Tanh: CustomNode,
        Softmax: CustomNode,
        LogSoftmax: CustomNode,
        Hardtanh: CustomNode,
        Hardshrink: CustomNode,
        Hardsigmoid: CustomNode,
        Hardswish: CustomNode,
        Mish: CustomNode,
        GELU: CustomNode,

        // Pooling layers
        MaxPool1d: CustomNode,
        MaxPool2d: CustomNode,
        MaxPool3d: CustomNode,
        AvgPool1d: CustomNode,
        AvgPool2d: CustomNode,
        AvgPool3d: CustomNode,
        AdaptiveMaxPool1d: CustomNode,
        AdaptiveMaxPool2d: CustomNode,
        AdaptiveMaxPool3d: CustomNode,
        AdaptiveAvgPool1d: CustomNode,
        AdaptiveAvgPool2d: CustomNode,
        AdaptiveAvgPool3d: CustomNode,

        // Dropout and other regularization layers
        Dropout: CustomNode,
        Dropout2d: CustomNode,
        Dropout3d: CustomNode,
        AlphaDropout: CustomNode,

        // Padding layers
        ReflectionPad1d: CustomNode,
        ReflectionPad2d: CustomNode,
        ReflectionPad3d: CustomNode,
        ReplicationPad1d: CustomNode,
        ReplicationPad2d: CustomNode,
        ReplicationPad3d: CustomNode,
        ZeroPad2d: CustomNode,

        // Upsampling and resizing layers
        Upsample: CustomNode,
        UpsamplingNearest2d: CustomNode,
        UpsamplingBilinear2d: CustomNode,

        // Transformer and attention layers
        Transformer: CustomNode,
        TransformerEncoder: CustomNode,
        TransformerDecoder: CustomNode,
        TransformerEncoderLayer: CustomNode,
        TransformerDecoderLayer: CustomNode,
        MultiheadAttention: CustomNode,

        // Utility layers for reshaping and folding
        Flatten: CustomNode,
        Unfold: CustomNode,
        Fold: CustomNode,

        // Other miscellaneous layers
        PixelShuffle: CustomNode,
        ChannelShuffle: CustomNode, // Often implemented manually, but can be included as a conceptual node
        Softmax2d: CustomNode,
        // ... Add more if required

        MLP: CustomNode,
        Cat: CustomNode,
        ModuleList: CustomNode,
        Sequential: CustomNode,

        Math: CustomNode,
        Squeeze: CustomNode
    }
};