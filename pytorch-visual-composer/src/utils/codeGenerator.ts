// utils/codeGenerator.ts
import { Node, Edge } from 'reactflow';
import { LayerParameter, layerParameters } from './layerParameters';
import { NodeData } from '../types/NodeTypes';
import yaml from 'js-yaml';
// Define parameter structure for each layer type, same as in Sidebar.
// For brevity, we reuse the same large dictionary from the previous step.
// Ensure this is identical to what you use in Sidebar to keep consistency.


// interface NodeData {
//     label: string;
//     parameters?: { [key: string]: any };
//     configKey?: string;
// }

export const generateCode = (all_nodes: Node[], edges: Edge[], modelName: string, layerToClassMap: { [key: string]: string }, layerParameters: { [key: string]: LayerParameter[] }): { code: string, yamlConfig: string } => {
    let nodes = all_nodes.filter(node => {
        const containedLayerIds = edges
            .filter(edge => edge.source === node.id && edge.target && edge.targetHandle);
        return !containedLayerIds.length;
    })
    // Map nodes by their IDs for easy access
    const nodeMap = all_nodes.reduce((acc, node) => {
        acc[node.id] = node;
        return acc;
    }, {} as { [key: string]: Node });


    // Extract ConfigNodes and map parameter names to their values
    const configNodes = all_nodes.filter(node => node.type === 'Config');
    const configParams: { [paramName: string]: { value: string; dependencies: string[] } } = {};

    // Collect all parameter names
    const allParamNames = configNodes.map(node => node.data.parameters?.param_name).filter(Boolean) as string[];

    configNodes.forEach(node => {
        const nodeData = node.data as NodeData;
        const paramName = nodeData.parameters?.param_name;
        const paramValue = nodeData.parameters?.value || '';

        if (paramName) {
            const dependencies = parseDependencies(paramValue, allParamNames);
            configParams[paramName] = { value: paramValue, dependencies };
        }
    });

    // Separate parameters into independent and dependent
    const independentParams: string[] = [];
    const dependentParams: string[] = [];

    Object.entries(configParams).forEach(([param, details]) => {
        if (details.dependencies.length === 0) {
            independentParams.push(param);
        } else {
            dependentParams.push(param);
        }
    });

    // Prepare dependency edges for dependent parameters
    const dependencyEdges: { from: string; to: string }[] = [];
    dependentParams.forEach(param => {
        configParams[param].dependencies.forEach(dep => {
            if (configParams[dep]) { // Only consider dependencies that are defined as ConfigNodes
                dependencyEdges.push({ from: param, to: dep });
            }
        });
    });

    // Perform topological sort on all parameters
    let sortedParams: string[];
    try {
        sortedParams = topologicalSort(allParamNames, dependencyEdges);
    } catch (error: any) {
        throw new Error(`Error generating ModelConfig: ${error.message}`);
    }

    // Build configuration class code
    let configCode = `class ${modelName}Config:\n`;
    configCode += `    def __init__(self`;

    // Add independent parameters as constructor arguments with default values
    independentParams.forEach(param => {
        const paramValue = configParams[param].value;
        let defaultValue = paramValue;

        // Convert JS boolean to Python boolean
        if (typeof paramValue === 'boolean') {
            defaultValue = paramValue ? 'True' : 'False';
        } else if (typeof paramValue === 'string') {
            // Handle numeric, tuple/list, activation functions, and string literals
            if (!isNaN(Number(paramValue))) {
                // Numeric value, no change needed
            } else if (/^[\[\(].*[\)\]]$/.test(paramValue)) {
                // Tuple or list, no change needed
            } else if (/^nn\.\w+\(\)$/.test(paramValue)) {
                // Activation functions or other nn.Module instances, no change needed
            } else {
                // String literal, add quotes
                defaultValue = `"${paramValue}"`;
            }
        }

        configCode += `, ${param}=${defaultValue}`;
    });

    configCode += `):\n`;

    // Assign independent parameters
    independentParams.forEach(param => {
        configCode += `        self.${param} = ${param}\n`;
    });

    // Assign dependent parameters in sorted order
    dependentParams.forEach(param => {
        const paramValue = configParams[param].value;
        configCode += `        self.${param} = ${paramValue}\n`;
    });

    // Build the model class code
    let modelCode = `class ${modelName}(nn.Module):\n`;
    modelCode += '    def __init__(self, config):\n';
    modelCode += `        super(${modelName}).__init__()\n`;
    modelCode += '        self.config = config\n';

    // Define layers
    nodes.forEach((node) => {
        if (!node.type) throw 'node.type empty';
        const nodeData = node.data as NodeData;
        const nodeName = sanitizeNodeName(nodeData.label);
        const configKey = nodeData.configKey || getNodeNameFromId(all_nodes, node.id);
        const params = layerParameters[node.type] || [];
        const className = layerToClassMap[node.type];

        if (node.type === 'ConfigNode' || node.type === 'Input' || node.type === 'Output' || node.type === 'Squeeze') {
            // Skip ConfigNodes and IO nodes
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
        if (node.type === 'ModuleList') {
            const modelTypeLayer = edges
                .filter(edge => edge.target === node.id && edge.targetHandle === 'modelType')
                .map(edge => edge.source);

            const sortedLayers = modelTypeLayer.map(id => nodeMap[id]).filter(Boolean);

            // Generate the list of layer instances
            let sequentialLayers = sortedLayers.map(targetNode => {
                const targetNodeName = buildCodeSection(edges, targetNode, all_nodes, layerToClassMap, nodeData, nodeName);
                return `${targetNodeName}`;
            }).join(',\n            ');

            // Include comment if provided
            const comment = nodeData.parameters?.comment;
            if (comment) {
                configCode += `        # ${comment}\n`;
            }
            const num_layers = nodeData.parameterNames?.find(v => v.name === 'num_layers');

            modelCode += `        self.${nodeName} = nn.ModuleList([\n`;
            modelCode += `            ${sequentialLayers}\n`;
            modelCode += `            for _ in range(${num_layers?.default})\n`;
            modelCode += '        ])\n';
            return;
        }
        if (node.type === 'Cat') return;
        if (node.type === 'Math') {
            let l_params = layerParameters[node.type];
            let a_l_param = l_params.find(v => v.name === 'a');
            let b_l_param = l_params.find(v => v.name === 'b') || node.data.parameters['b'];
            let configa = a_l_param ? findConfig(all_nodes, node, edges, a_l_param) : node.data.parameters['a'];
            let configb = b_l_param ? findConfig(all_nodes, node, edges, b_l_param) : node.data.parameters['b'];
            modelCode += `        self.${nodeName} = a * b\n`;
        }
        // Construct the parameter list for this layer
        let paramLines: string[] = [];
        params.forEach((param) => {
            const paramName = param.name;
            // All parameters stored in config as configKey_paramName
            const configParam = `config.${configKey}_${paramName}`;
            if (node.type) {
                let l_params = layerParameters[node.type];
                let l_param = l_params.find(v => v.name === paramName);
                if (l_param) {
                    console.log(l_param);
                    let config = findConfig(all_nodes, node, edges, l_param);
                    if (config) {
                        if (config.type === 'Config') {
                            paramLines.push(`            ${paramName}=config.${config.data.parameters.param_name}`);
                        }
                        else {
                            if (layerToClassMap[config.type as any]) {
                                paramLines.push(`            ${paramName}=${layerToClassMap[config.type as any]}`);
                            }
                        }
                        return;
                    }
                    else if (!node.data.parameters.hasOwnProperty(l_param.name)) {
                        return;
                    }
                    else {
                        paramLines.push(`            ${paramName}=${node.data.parameters[l_param.name]}`);
                    }
                }
            }
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
    const executionOrder = determineExecutionOrder(nodes, edges);
    let inputNames = executionOrder.map((nodeId) => {
        const node = nodeMap[nodeId];
        const nodeData = node.data as NodeData;
        const nodeName = sanitizeNodeName(nodeData.label);

        if (node.type === 'Input') {
            return nodeName;
        }
        return null;
    }).filter(x => x);
    modelCode += `\n    def forward(self, ${inputNames.join(', ')}):\n`;

    executionOrder.forEach((nodeId) => {
        const node = nodeMap[nodeId];
        const nodeData = node.data as NodeData;
        const nodeName = sanitizeNodeName(nodeData.label);

        if (node.type === 'Input') {
            // Input node maps to x
            modelCode += `        ${getNodeName(node)} = ${nodeData.label}\n`;
        } else if (node.type === 'Cat') {
            // torch.cat operation
            const predecessors = edges
                .filter(edge => edge.target === node.id)
                .map(edge => getNodeNameFromId(all_nodes, edge.source));
            const inputVars = predecessors.join(', ');
            const dim = nodeData.parameters?.dim ?? 1;
            const comment = nodeData.parameters?.comment;
            if (comment) {
                modelCode += `        # ${comment}\n`;
            }
            modelCode += `        ${getNodeName(node)} = torch.cat(\n`;
            modelCode += `            (${inputVars}), dim=${dim}\n`;
            modelCode += `        )\n`;
            if (nodeData.parameters?.output_shape) {
                modelCode += `        # ${nodeData.parameters.output_shape}\n`;
            }
        } else if (node.type === 'Sequential') {
            // Sequential layer execution
            const predecessors = edges
                .filter(edge => edge.target === node.id)
                .map(edge => getNodeNameFromId(all_nodes, edge.source));
            const inputVar = predecessors.length > 0 ? predecessors.join(', ') : 'x';
            const comment = nodeData.parameters?.comment;
            if (comment) {
                modelCode += `        # ${comment}\n`;
            }
            modelCode += `        ${getNodeName(node)} = self.${nodeName}(${inputVar})\n`;
        } else if (node.type === 'ModuleList') {
            // Sequential layer execution
            const predecessors = edges
                .filter(edge => edge.target === node.id && !edge.targetHandle)
                .map(edge => getNodeNameFromId(all_nodes, edge.source));
            const inputVar = predecessors.length > 0 ? predecessors.join(', ') : 'x';
            modelCode += `        ${getNodeName(node)} = ${inputVar}\n`;
            modelCode += `        for layer in self.${nodeName}:\n`;
            modelCode += `            ${getNodeName(node)} = layer(${getNodeName(node)})\n`;
        } else if (node.type === 'Squeeze') {
            const predecessors = edges
                .filter(edge => edge.target === node.id && !edge.targetHandle)
                .map(edge => getNodeNameFromId(all_nodes, edge.source));
            const inputVar = predecessors.length > 0 ? predecessors.join(', ') : 'x';
            modelCode += `        ${getNodeName(node)} = ${inputVar}.squeeze(-1)\n`;
        } else {
            const predecessors = edges
                .filter(edge => edge.target === node.id && !edge.targetHandle)
                .map(edge => getNodeNameFromId(all_nodes, edge.source));

            const inputVars = predecessors.length > 0 ? predecessors.join(', ') : 'x';

            if (node.type === 'Output') {
                // If Output node, just assume it takes one input
                modelCode += `        ${getNodeName(node)} = ${inputVars}\n`;
            } else {
                if (!node.type) throw 'node.type empty';
                // Call the layer defined above
                if (layerToClassMap[node.type]) {
                    const comment = nodeData.parameters?.comment;
                    if (comment) {
                        modelCode += `        # ${comment}\n`;
                    }
                    modelCode += `        ${getNodeName(node)} = self.${nodeName}(${inputVars})\n`;
                } else {
                    // Unrecognized layer - just pass input forward
                    modelCode += `        # WARNING: No class mapped for ${node.type}, passing input forward\n`;
                    modelCode += `        ${getNodeName(node)} = ${inputVars}\n`;
                }
            }
        }
    });

    // Determine the final output node. Prefer the node marked as 'Output'.
    const outputNodeId = executionOrder.find(id => nodeMap[id].type === 'Output') || executionOrder[executionOrder.length - 1];
    modelCode += `        return ${getNodeNameFromId(all_nodes, outputNodeId)}\n`;

    // After generating Python classes, generate YAML configuration
    const yamlConfig = generateYAMLConfig(configParams, independentParams, modelName);

    // Combine the codes
    let code = 'import torch\nimport torch.nn as nn\n\n';
    code += configCode + '\n';
    code += modelCode;

    return { code, yamlConfig };
};

function buildCodeSection(
    edges: Edge[],
    node: Node,
    all_nodes: Node[],
    layerToClassMap: { [key: string]: string; },
    nodeData: NodeData,
    nodeName: string) {
    let modelCode = '';
    const predecessors = edges
        .filter(edge => edge.target === node.id)
        .map(edge => getNodeNameFromId(all_nodes, edge.source));

    if (!node.type) throw 'node.type empty';
    // Call the layer defined above
    if (layerToClassMap[node.type]) {
        //predecessors.length > 0 ? predecessors.join(', ') : 'x';
        const inputVars = (node.data as NodeData).parameterNames?.map((param) => {
            let config = findConfig(all_nodes, node, edges, param);
            if (config) {
                if (config.type === 'Config') {
                    return `${param.name}=config.${config.data.parameters.param_name}`;
                }
                else {
                    if (layerToClassMap[config.type as any]) {
                        return `${param.name}=${layerToClassMap[config.type as any]}`;
                    }
                }
                return;
            }
            return `${param.name}=${param.default}`;
        }).join();
        modelCode += `${node.type}(${inputVars})\n`;
    }

    return modelCode;
}

function getNodeName(node: Node): string {
    const nodeData = node.data as NodeData;
    const nodeName = sanitizeNodeName(nodeData.label);
    return `n_${nodeName}`;
}
function getNodeNameFromId(nodes: Node[], id: any) {
    let node = nodes.find(v => v.id === id);
    if (!node) throw 'node not found';
    return getNodeName(node);
}
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

function findConfig(all_nodes: Node[], node: Node, edges: Edge[], l_param: { name: string }): Node | undefined | null {
    let edge = edges.find(v => v.target === node.id && v.targetHandle === l_param.name);
    if (edge?.source) {
        let configNode = all_nodes.find(v => v.id === edge?.source);
        return configNode;
    }
    return null;
}

/**
 * Performs topological sort on the dependency graph.
 * @param nodes Array of parameter names.
 * @param edges Array of dependency edges (from, to).
 * @returns An array of parameter names sorted in dependency order.
 * @throws Error if a circular dependency is detected.
 */
const topologicalSort = (nodes: string[], edges: { from: string; to: string }[]): string[] => {
    const adjList: { [key: string]: string[] } = {};
    nodes.forEach(node => {
        adjList[node] = [];
    });
    edges.forEach(edge => {
        adjList[edge.from].push(edge.to);
    });

    const visited: { [key: string]: boolean } = {};
    const tempMarks: { [key: string]: boolean } = {};
    const result: string[] = [];

    const visit = (node: string) => {
        if (tempMarks[node]) {
            throw new Error(`Circular dependency detected involving parameter "${node}".`);
        }
        if (!visited[node]) {
            tempMarks[node] = true;
            adjList[node].forEach(dep => visit(dep));
            tempMarks[node] = false;
            visited[node] = true;
            result.push(node);
        }
    };

    nodes.forEach(node => {
        if (!visited[node]) {
            visit(node);
        }
    });

    return result.reverse(); // Return in correct order
};

/**
 * Parses the parameter value to find dependencies on other config parameters.
 * @param value The parameter value as a string.
 * @returns An array of param_names that this parameter depends on.
 */
const parseDependencies = (value: string, allParams: string[]): string[] => {
    const dependencies: string[] = [];
    allParams.forEach((param) => {
        // Create a regex pattern with word boundaries to match exact parameter names
        const regex = new RegExp(`\\b${param}\\b`, 'g');
        if (regex.test(value)) {
            dependencies.push(param);
        }
    });
    return dependencies;
};

/**
 * Generates a YAML configuration string based on independent parameters.
 * @param configParams All configuration parameters with their values and dependencies.
 * @param independentParams List of independent parameter names.
 * @param modelName Name of the model to customize the YAML filename if needed.
 * @returns A string representing the YAML configuration.
 */
const generateYAMLConfig = (
    configParams: { [paramName: string]: { value: string; dependencies: string[] } },
    independentParams: string[],
    modelName: string
): string => {
    const yamlObj: { [key: string]: any } = {};

    independentParams.forEach(param => {
        let value: any = configParams[param].value;

        // Convert certain strings to proper YAML representations
        if (/^nn\.\w+\(\)$/.test(value)) {
            // Activation functions or other nn.Module instances
            value = value; // Keep as is; user will need to handle this in Python
        } else if (!isNaN(Number(value))) {
            // Numeric values remain as numbers
            value = Number(value);
        } else if (/^[\[\(].*[\)\]]$/.test(value)) {
            // Tuples or lists
            // Evaluate the string to convert it to an actual array or tuple
            // Caution: Using eval can be dangerous; ensure input is sanitized
            try {
                value = eval(value);
            } catch (e) {
                // If eval fails, keep the string as is
                value = value;
            }
        } else if (typeof value === 'boolean') {
            // Boolean values
            value = value;
        } else {
            // Treat as string literals
            value = value;
        }

        yamlObj[param] = value;
    });

    // Convert the object to YAML string
    const yamlStr = yaml.dump({
        model: {
            config: yamlObj
        }
    }, { noRefs: true, indent: 4 });

    return yamlStr;
};