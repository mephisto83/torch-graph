// components/ConfigNode.tsx
import React from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { Box, Typography, TextField } from '@mui/material';
import { useGraph } from '../provider/GraphProvider';

interface ConfigNodeData {
    label: string;
    parameters?: { [key: string]: any };
}

const ConfigNode: React.FC<NodeProps<ConfigNodeData>> = ({ id, data }) => {
    const { nodes, edges, setNodes, setEdges } = useGraph();

    // Function to update parameter value
    const handleValueChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const { value } = event.target;
        setNodes((nds) =>
            nds.map((node) =>
                node.id === id
                    ? {
                        ...node,
                        data: {
                            ...node.data,
                            parameters: {
                                ...node.data.parameters,
                                value,
                            },
                        },
                    }
                    : node
            )
        );
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

            {/* Parameter Name */}
            <Box sx={{ textAlign: 'left', mt: 1 }}>
                <Typography variant="body2" sx={{ mb: 1 }}>
                    Name: {data.parameters?.param_name || ''}
                </Typography>
                {/* <TextField size='small'
                    variant="standard"
                    value={data.parameters?.param_name || ''}
                    onChange={(e) =>
                        setNodes((nds) =>
                            nds.map((node) =>
                                node.id === id
                                    ? {
                                        ...node,
                                        data: {
                                            ...node.data,
                                            parameters: {
                                                ...node.data.parameters,
                                                param_name: e.target.value,
                                            },
                                        },
                                    }
                                    : node
                            )
                        )
                    }
                    fullWidth
                /> */}
            </Box>

            {/* Parameter Value */}
            <Box sx={{ textAlign: 'left', mt: 2 }}>
                <Typography variant="body2" sx={{ mb: 1 }}>
                    Value: {data.parameters?.value || ''}
                </Typography>
                {/* <TextField
                    variant="standard" size='small'
                    value={data.parameters?.value || ''}
                    onChange={handleValueChange}
                    fullWidth
                /> */}
            </Box>

            {/* Source Handles for Parameters */}
            <Handle
                type="source"
                position={Position.Right}
                id={'param_name'} // Unique handle ID based on parameter name
                style={{
                    top: '50%',
                    background: 'green',
                    width: '12px',
                    height: '12px',
                    borderRadius: '50%',
                }}
            />
        </Box>
    );
};

export default ConfigNode;
