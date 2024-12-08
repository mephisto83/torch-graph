// components/CustomNode.tsx
import React from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { useGraph } from '../provider/GraphProvider';
import { Box, Typography, TextField } from '@mui/material';
import { LayerParameter } from '../utils/layerParameters';

interface CustomNodeData {
    label: string;
    parameters?: { [key: string]: any };
    parameterNames?: LayerParameter[]; // List of parameter names expected
}

const CustomNode: React.FC<NodeProps<CustomNodeData>> = ({ id, data }) => {
    const { nodes, edges } = useGraph();
    const currentEdges = edges.filter(edge => edge.target === id);

    // Extract parameter connections based on parameterNames
    const parameterNames = data.parameterNames || [];

    // Function to get the connected node for a given parameter
    const getConnectedNode = (paramName: string) => {
        const edge = currentEdges.find(edge => edge.sourceHandle === paramName);
        if (edge) {
            const sourceNode = nodes.find(node => node.id === edge.source);
            return sourceNode;
        }
        return null;
    };

    return (
        <Box
            sx={{
                padding: '10px',
                border: '1px solid #777',
                borderRadius: '5px',
                background: '#fff',
                minWidth: '200px',
                textAlign: 'center',
                position: 'relative',
            }}
        >
            <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
                {data.label}
            </Typography>

            {/* Display Parameters */}
            <Box sx={{ textAlign: 'left', mt: 1 }}>
                {parameterNames.map(param => {
                    const connectedNode = getConnectedNode(param.name);
                    const paramValue = connectedNode?.data?.value || data.parameters?.[param.name] || param.default;

                    return (
                        <Box key={param.name} sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                            <div style={{ flex: 1, display: 'flex' }}>
                                <Typography variant="body2" sx={{ mr: 1, width: '80px' }}>
                                    {param.name}:
                                </Typography>
                            </div>
                            <div style={{ flex: 1, display: 'flex' }}>
                                <TextField
                                    variant="standard"
                                    value={paramValue}
                                    size="small"
                                />
                            </div>
                        </Box>
                    );
                })}
            </Box>


            {/* Source Handle */}
            <Handle
                key={"target"}
                type="target"
                position={Position.Left}
                style={{
                    top: 20 + (0) * 33,
                    background: 'red',
                    width: '10px',
                    height: '10px',
                }}
            />
            {/* Parameter Input Handles */}
            {parameterNames.map((param, index) => {
                return (
                    <Handle
                        key={param.name}
                        type="target"
                        position={Position.Left}
                        id={param.name} // Unique handle ID per parameter
                        style={{
                            top: 23 + ((index + 1) * 37),
                            background: '#555',
                            width: '10px',
                            height: '10px',
                        }}
                    />
                )
            })}
            <Handle
                type="source"
                position={Position.Right}
                style={{ background: '#555' }}
            />
        </Box>
    );
};

export default CustomNode;
