// components/CodeGenerator.tsx
import React from 'react';
import { Edge, Node } from 'reactflow';
import { generateCode } from '../utils/codeGenerator';

interface CodeGeneratorProps {
    nodes: Node[];
    edges: Edge[];
}

const CodeGenerator: React.FC<CodeGeneratorProps> = ({ nodes, edges }) => {
    const [code, setCode] = React.useState('');

    const handleGenerateCode = () => {
        const generatedCode = generateCode(nodes, edges);
        setCode(generatedCode);
    };

    return (
        <div>
            <button onClick={handleGenerateCode}>Generate Code</button>
            <pre>{code}</pre>
        </div>
    );
};

export default CodeGenerator;
