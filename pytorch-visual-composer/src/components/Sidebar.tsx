// components/Sidebar.tsx
import React from 'react';
import { Node } from 'reactflow';

import Accordion from '@mui/material/Accordion';
import AccordionSummary from '@mui/material/AccordionSummary';
import AccordionDetails from '@mui/material/AccordionDetails';
import Typography from '@mui/material/Typography';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';

interface LayerParameter {
    name: string;
    type: 'number' | 'text' | 'boolean' | 'select';
    default?: any;
    options?: string[];
}

// Comprehensive parameter definitions for all layers
const layerParameters: { [key: string]: LayerParameter[] } = {
    // Input/Output
    "Input": [],
    "Output": [],

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

// Categories
const layerCategories: { [category: string]: string[] } = {
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
    "Other": ["PixelShuffle", "ChannelShuffle", "Softmax2d"]
};

interface SidebarProps {
    nodes: Node[];
    setNodes: React.Dispatch<React.SetStateAction<Node[]>>;
    selectedNode: Node | null;
    setSelectedNode: React.Dispatch<React.SetStateAction<Node | null>>;
}

const Sidebar: React.FC<SidebarProps> = ({
    nodes,
    setNodes,
    selectedNode,
    setSelectedNode,
}) => {
    const onDragStart = (
        event: React.DragEvent<HTMLDivElement>,
        nodeType: string
    ) => {
        event.dataTransfer.setData('application/reactflow', nodeType);
        event.dataTransfer.effectAllowed = 'move';
    };

    const handleNameChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        if (selectedNode) {
            const newName = event.target.value;
            const updatedNode = {
                ...selectedNode,
                data: {
                    ...selectedNode.data,
                    label: newName,
                },
            };

            setNodes((nds) =>
                nds.map((node) => (node.id === selectedNode.id ? updatedNode : node))
            );
            setSelectedNode(updatedNode);
        }
    };

    const handleParameterChange = (event: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
        if (selectedNode) {
            const { name, value, type, checked } = event.target as HTMLInputElement;
            let paramValue: any = value;
            if (type === 'number') {
                paramValue = Number(value);
            } else if (type === 'checkbox') {
                paramValue = checked;
            }

            const updatedParameters = {
                ...selectedNode.data.parameters,
                [name]: paramValue,
            };

            const updatedNode = {
                ...selectedNode,
                data: {
                    ...selectedNode.data,
                    parameters: updatedParameters,
                },
            };

            setNodes((nds) =>
                nds.map((node) => (node.id === selectedNode.id ? updatedNode : node))
            );
            setSelectedNode(updatedNode);
        }
    };

    const renderParameterFields = () => {
        if (!selectedNode || selectedNode.type === 'edge') return null;

        const params = layerParameters[selectedNode.type] || [];
        if (params.length === 0) return null;

        return (
            <>
                <h4 style={{ marginTop: '20px' }}>Parameters</h4>
                {params.map((param) => {
                    const currentValue = selectedNode.data.parameters?.[param.name] ?? param.default;
                    if (param.type === 'boolean') {
                        return (
                            <label key={param.name} style={{ display: 'block', marginTop: '10px' }}>
                                {param.name}:
                                <input
                                    type="checkbox"
                                    name={param.name}
                                    checked={!!currentValue}
                                    onChange={handleParameterChange}
                                    style={{ marginLeft: '5px' }}
                                />
                            </label>
                        );
                    } else if (param.type === 'select' && param.options) {
                        return (
                            <label key={param.name} style={{ display: 'block', marginTop: '10px' }}>
                                {param.name}:
                                <select
                                    name={param.name}
                                    value={currentValue}
                                    onChange={handleParameterChange}
                                    style={{ marginLeft: '5px' }}
                                >
                                    {param.options.map((opt) => (
                                        <option key={opt} value={opt}>{opt}</option>
                                    ))}
                                </select>
                            </label>
                        );
                    } else {
                        // number or text
                        const inputType = param.type === 'number' ? 'number' : 'text';
                        return (
                            <label key={param.name} style={{ display: 'block', marginTop: '10px' }}>
                                {param.name}:
                                <input
                                    type={inputType}
                                    name={param.name}
                                    value={currentValue}
                                    onChange={handleParameterChange}
                                    style={{ marginLeft: '5px', width: '100px' }}
                                />
                            </label>
                        );
                    }
                })}
            </>
        );
    };

    return (
        <aside style={{ padding: '10px', overflowY: 'auto', width: '250px' }}>
            <h2>PyTorch Layers</h2>
            {Object.entries(layerCategories).map(([category, layers]) => (
                <Accordion key={category} disableGutters>
                    <AccordionSummary
                        expandIcon={<ExpandMoreIcon />}
                        aria-controls={`${category}-content`}
                        id={`${category}-header`}
                    >
                        <Typography variant="subtitle1" style={{ fontWeight: 'bold' }}>
                            {category}
                        </Typography>
                    </AccordionSummary>
                    <AccordionDetails style={{ display: 'flex', flexDirection: 'column' }}>
                        {layers.map((layerType) => (
                            <div
                                key={layerType}
                                onDragStart={(event) => onDragStart(event, layerType)}
                                draggable
                                style={{
                                    cursor: 'grab',
                                    padding: '5px',
                                    border: '1px solid #ccc',
                                    borderRadius: '3px',
                                    background: '#f9f9f9',
                                    marginBottom: '5px'
                                }}
                            >
                                {layerType}
                            </div>
                        ))}
                    </AccordionDetails>
                </Accordion>
            ))}

            {selectedNode && selectedNode.type !== 'edge' && (
                <div className="node-editor" style={{ marginTop: '20px' }}>
                    <h3>Edit Node</h3>
                    <label>
                        Name:
                        <input
                            type="text"
                            value={selectedNode.data.label}
                            onChange={handleNameChange}
                            style={{ marginLeft: '5px' }}
                        />
                    </label>
                    {renderParameterFields()}
                </div>
            )}
        </aside>
    );
};

export default Sidebar;
