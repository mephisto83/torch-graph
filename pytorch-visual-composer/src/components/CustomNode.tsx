// components/CustomNode.tsx
import React from 'react';
import { Handle, Position, NodeProps } from 'reactflow';

interface CustomNodeData {
    label: string;
    parameters?: { [key: string]: any };
}

const CustomNode: React.FC<NodeProps<CustomNodeData>> = ({ data }) => {
    return (
        <div style={{
            padding: '10px',
            border: '1px solid #777',
            borderRadius: '5px',
            background: '#fff',
            minWidth: '150px',
            textAlign: 'center'
        }}>
            <div style={{ fontWeight: 'bold', marginBottom: '5px' }}>{data.label}</div>
            {/* Add more custom UI elements here if needed */}
            <Handle type="target" position={Position.Left} style={{ background: '#555' }} />
            <Handle type="source" position={Position.Right} style={{ background: '#555' }} />
        </div>
    );
};

export default CustomNode;
