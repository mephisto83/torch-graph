// utils/codeGenerator.ts
import { Node, Edge } from 'reactflow';

interface NodeData {
    label: string;
    parameters?: { [key: string]: any };
    configKey?: string;
}

export const generateCode = (nodes: Node[], edges: Edge[]): string => {
    // Map nodes by their IDs for easy access
    const nodeMap = nodes.reduce((acc, node) => {
        acc[node.id] = node;
        return acc;
    }, {} as { [key: string]: Node });

    // Build configuration class code
    let configCode = 'class ModelConfig:\n';
    configCode += '    def __init__(self):\n';

    // Tunable parameters
    nodes.forEach((node) => {
        const nodeData = node.data as NodeData;
        const configKey = nodeData.configKey || node.id;
        const params = nodeData.parameters || {};
        for (const [paramName, paramValue] of Object.entries(params)) {
            configCode += `        self.${configKey}_${paramName} = ${paramValue}\n`;
        }
    });

    // Dependent properties (example)
    configCode += '\n        # Dependent properties\n';
    configCode += '        self.calculate_dependent_properties()\n\n';
    configCode += '    def calculate_dependent_properties(self):\n';
    configCode += '        pass  # Implement calculations\n\n';

    // Build the model class code
    let modelCode = 'class GeneratedModel(nn.Module):\n';
    modelCode += '    def __init__(self, config):\n';
    modelCode += '        super(GeneratedModel, self).__init__()\n';
    modelCode += '        self.config = config\n';

    // Define layers
    nodes.forEach((node) => {
        const nodeData = node.data as NodeData;
        const nodeName = nodeData.label.replace(/\s+/g, '_');
        const configKey = nodeData.configKey || node.id;
        const params = nodeData.parameters || {};

        if (node.type === 'Conv2d') {
            modelCode += `        self.${nodeName} = nn.Conv2d(\n`;
            modelCode += `            in_channels=config.${configKey}_in_channels,\n`;
            modelCode += `            out_channels=config.${configKey}_out_channels,\n`;
            modelCode += `            kernel_size=config.${configKey}_kernel_size\n`;
            modelCode += '        )\n';
        }
        // Handle other node types similarly
    });

    // Generate the forward method
    modelCode += '\n    def forward(self, x):\n';

    // Implement forward logic based on edges
    const executionOrder = determineExecutionOrder(nodes, edges);

    executionOrder.forEach((nodeId) => {
        const node = nodeMap[nodeId];
        const nodeData = node.data as NodeData;
        const nodeName = nodeData.label.replace(/\s+/g, '_');

        if (node.type === 'Input') {
            modelCode += `        ${node.id} = x\n`;
        } else {
            const predecessors = edges
                .filter((edge) => edge.target === node.id)
                .map((edge) => edge.source);

            const inputVars = predecessors.map((id) => `${id}`).join(', ');

            modelCode += `        ${node.id} = self.${nodeName}(${inputVars})\n`;
        }
    });

    // Assuming the last node is the output
    const outputNodeId = executionOrder[executionOrder.length - 1];
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
        adjList[edge.source].push(edge.target);
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
