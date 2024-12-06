// utils/codeGenerator.ts
import { Node, Edge } from 'reactflow';
import { layerParameters } from './layerParameters';

// Define parameter structure for each layer type, same as in Sidebar.
// For brevity, we reuse the same large dictionary from the previous step.
// Ensure this is identical to what you use in Sidebar to keep consistency.
interface LayerParameter {
    name: string;
    type: 'number' | 'text' | 'boolean' | 'select';
    default?: any;
    options?: string[];
}


// Map each node type to its corresponding PyTorch nn class name.
// If a layer is not in this map, we will just print a comment.
export const layerToClassMap: { [key: string]: string } = {
    // Custom
    "MLP": "python.syllable.MLP",

    // New Utility Layers
    "Cat": "", // Handled as a functional operation
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

interface NodeData {
    label: string;
    parameters?: { [key: string]: any };
    configKey?: string;
}

export const generateCode = (nodes: Node[], edges: Edge[], modelName: string): string => {
    // Map nodes by their IDs for easy access
    const nodeMap = nodes.reduce((acc, node) => {
        acc[node.id] = node;
        return acc;
    }, {} as { [key: string]: Node });

    // Build configuration class code
    let configCode = 'class ModelConfig:\n';
    configCode += '    def __init__(self):\n';

    // Set tunable parameters in config
    nodes.forEach((node) => {
        const nodeData = node.data as NodeData;
        const configKey = nodeData.configKey || node.id;
        const params = nodeData.parameters || {};
        for (const [paramName, paramValue] of Object.entries(params)) {
            // Convert JS boolean to Python boolean
            let pythonValue = paramValue;
            if (typeof paramValue === 'boolean') {
                pythonValue = paramValue ? 'True' : 'False';
            } else if (typeof paramValue === 'string') {
                // For strings that represent tuples or lists, ensure they are valid Python syntax
                // For example: "3,3" should be "(3,3)"
                // Here, we'll assume the user inputs valid Python syntax
                pythonValue = paramValue;
            }
            // For numbers, no change
            configCode += `        self.${configKey}_${paramName} = ${pythonValue}\n`;
        }
    });

    // Dependent properties placeholder
    configCode += '\n        # Dependent properties\n';
    configCode += '        self.calculate_dependent_properties()\n\n';
    configCode += '    def calculate_dependent_properties(self):\n';
    configCode += '        pass\n\n';

    // Build the model class code
    let modelCode = `class ${modelName}(nn.Module):\n`;
    modelCode += '    def __init__(self, config):\n';
    modelCode += '        super().__init__()\n';
    modelCode += '        self.config = config\n';

    // Define layers
    nodes.forEach((node) => {
        if (!node.type) throw 'node.type empty';
        const nodeData = node.data as NodeData;
        const nodeName = sanitizeNodeName(nodeData.label);
        const configKey = nodeData.configKey || node.id;
        const params = layerParameters[node.type] || [];
        const className = layerToClassMap[node.type];

        if (node.type === 'Input' || node.type === 'Output' || !className) {
            // If no class name is defined or it's Input/Output, no layer needed
            return;
        }

        if (node.type === 'Sequential') {
            // nn.Sequential requires adding the contained layers in order
            // Collect layers that are children of this Sequential node
            const containedLayerIds = edges
                .filter(edge => edge.source === node.id)
                .map(edge => edge.target);

            const containedLayers = containedLayerIds.map(id => nodeMap[id]).filter(Boolean);

            // Sort containedLayers based on execution order
            const sortedLayers = determineExecutionOrder(containedLayers, edges).map(id => nodeMap[id]);

            // Generate the list of layer instances
            let sequentialLayers = sortedLayers.map(targetNode => {
                const targetNodeName = sanitizeNodeName(targetNode.data.label);
                return `self.${targetNodeName}`;
            }).join(',\n            ');

            // Include comment if provided
            const comment = nodeData.parameters?.comment;
            if (comment) {
                configCode += `        # ${comment}\n`;
            }

            modelCode += `        self.${nodeName} = nn.Sequential(\n`;
            modelCode += `            ${sequentialLayers}\n`;
            modelCode += '        )\n';
            return;
        }

        // Construct the parameter list for this layer
        let paramLines: string[] = [];
        params.forEach((param) => {
            const paramName = param.name;
            // All parameters stored in config as configKey_paramName
            const configParam = `config.${configKey}_${paramName}`;

            // For booleans, we already converted in config. For numbers and text, trust input.
            // For select, also trust input.
            paramLines.push(`            ${paramName}=${configParam}`);
        });

        // Include comment if provided
        const comment = nodeData.parameters?.comment;
        if (comment) {
            configCode += `        # ${comment}\n`;
        }

        modelCode += `        self.${nodeName} = ${className}(\n`;
        modelCode += paramLines.join(',\n') + '\n';
        modelCode += '        )\n';
    });

    // Generate the forward method
    modelCode += '\n    def forward(self, x):\n';

    const executionOrder = determineExecutionOrder(nodes, edges);

    executionOrder.forEach((nodeId) => {
        const node = nodeMap[nodeId];
        const nodeData = node.data as NodeData;
        const nodeName = sanitizeNodeName(nodeData.label);

        if (node.type === 'Input') {
            // Input node maps to x
            modelCode += `        ${node.id} = x\n`;
        } else if (node.type === 'Cat') {
            // torch.cat operation
            const predecessors = edges
                .filter(edge => edge.target === node.id)
                .map(edge => edge.source);
            const inputVars = predecessors.join(', ');
            const dim = nodeData.parameters?.dim ?? 1;
            const comment = nodeData.parameters?.comment;
            if (comment) {
                modelCode += `        # ${comment}\n`;
            }
            modelCode += `        ${node.id} = torch.cat(\n`;
            modelCode += `            (${inputVars}), dim=${dim}\n`;
            modelCode += `        )\n`;
            if (nodeData.parameters?.output_shape) {
                modelCode += `        # ${nodeData.parameters.output_shape}\n`;
            }
        } else if (node.type === 'Sequential') {
            // Sequential layer execution
            const predecessors = edges
                .filter(edge => edge.target === node.id)
                .map(edge => edge.source);
            const inputVar = predecessors.length > 0 ? predecessors.join(', ') : 'x';
            const comment = nodeData.parameters?.comment;
            if (comment) {
                modelCode += `        # ${comment}\n`;
            }
            modelCode += `        ${node.id} = self.${nodeName}(${inputVar})\n`;
        } else {
            const predecessors = edges
                .filter(edge => edge.target === node.id)
                .map(edge => edge.source);

            const inputVars = predecessors.length > 0 ? predecessors.join(', ') : 'x';

            if (node.type === 'Output') {
                // If Output node, just assume it takes one input
                modelCode += `        ${node.id} = ${inputVars}\n`;
            } else {
                if (!node.type) throw 'node.type empty';
                // Call the layer defined above
                if (layerToClassMap[node.type]) {
                    const comment = nodeData.parameters?.comment;
                    if (comment) {
                        modelCode += `        # ${comment}\n`;
                    }
                    modelCode += `        ${node.id} = self.${nodeName}(${inputVars})\n`;
                } else {
                    // Unrecognized layer - just pass input forward
                    modelCode += `        # WARNING: No class mapped for ${node.type}, passing input forward\n`;
                    modelCode += `        ${node.id} = ${inputVars}\n`;
                }
            }
        }
    });

    // Determine the final output node. Prefer the node marked as 'Output'.
    const outputNodeId = executionOrder.find(id => nodeMap[id].type === 'Output') || executionOrder[executionOrder.length - 1];
    modelCode += `        return ${outputNodeId}\n`;

    // Combine the codes
    let code = 'import torch\nimport torch.nn as nn\n\n';
    code += configCode + '\n';
    code += modelCode;

    return code;
};

// Utility function to determine execution order
const determineExecutionOrder = (nodes: Node[], edges: Edge[]): string[] => {
    // Build adjacency list
    const adjList = nodes.reduce((acc, node) => {
        acc[node.id] = [];
        return acc;
    }, {} as { [key: string]: string[] });

    edges.forEach((edge) => {
        if (adjList[edge.source]) {
            adjList[edge.source].push(edge.target);
        }
    });

    // Perform topological sort
    const visited = new Set<string>();
    const stack: string[] = [];

    const visit = (nodeId: string) => {
        if (!visited.has(nodeId)) {
            visited.add(nodeId);
            adjList[nodeId].forEach((neighbor) => visit(neighbor));
            stack.push(nodeId);
        }
    };

    nodes.forEach((node) => {
        if (!visited.has(node.id)) {
            visit(node.id);
        }
    });

    return stack.reverse(); // Execution order
};

// Utility function to sanitize node names for use as Python variable names
const sanitizeNodeName = (name: string): string => {
    // Replace spaces and special characters with underscores
    return name.replace(/\s+/g, '_').replace(/[^a-zA-Z0-9_]/g, '');
};