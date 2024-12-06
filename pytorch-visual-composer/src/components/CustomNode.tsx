// components/CustomNode.tsx
import React from 'react';
import { Handle, Position, NodeProps } from 'reactflow';

interface CustomNodeData {
    label: string;
    parameters?: { [key: string]: any };
}

const CustomNode: React.FC<NodeProps<CustomNodeData>> = ({ data }) => {
    return (
        <div className="custom-node">
            <div>{data.label}</div>
            {/* Add UI for displaying parameters if needed */}
            <Handle type="target" position={Position.Left} />
            <Handle type="source" position={Position.Right} />
        </div>
    );
};

export default CustomNode;
