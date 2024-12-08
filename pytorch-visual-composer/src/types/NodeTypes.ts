// types/NodeTypes.ts
export type NodeType = 'Input' | 'Conv2d' | 'ReLU' | 'Linear' | 'Output';

// Define the properties for each node type
export interface NodeData {
    label: string;
    parameters?: { [key: string]: any };
    parameterNames?: { name: string, type: string, default: any }[];
    configKey?: string;
}
