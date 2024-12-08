// components/CodeGenerator.tsx
import React, { useState } from 'react';
import { generateCode } from '../utils/codeGenerator';
import { useGraph } from '../provider/GraphProvider';

const CodeGenerator: React.FC = () => {
    const { nodes, edges, modelName, layerToClassMap, layerParameters } = useGraph();
    const [generatedCode, setGeneratedCode] = useState<string>('');
    const [generatedYamlCode, setGeneratedYamlCode] = useState<string>('');

    const handleGenerateCode = () => {
        const { code, yamlConfig } = generateCode(nodes, edges, modelName, layerToClassMap, layerParameters);
        setGeneratedCode(code);
        setGeneratedYamlCode(yamlConfig)
    };

    const handleDownloadCode = () => {
        if (!generatedCode) {
            alert("Please generate the code first.");
            return;
        }
        const element = document.createElement("a");
        const file = new Blob([generatedCode], { type: 'text/plain' });
        element.href = URL.createObjectURL(file);
        element.download = `${modelName}.py`;
        document.body.appendChild(element); // Required for Firefox
        element.click();
        element.remove();
    };

    return (
        <div style={{ padding: '10px', minWidth: '400px', width: `25vw`, borderLeft: '1px solid #ccc', overflowY: 'auto' }}>
            <h2>Code Generator</h2>
            <button onClick={handleGenerateCode} style={{ marginBottom: '10px' }}>
                Generate Code
            </button>
            {generatedCode && (
                <>
                    <button onClick={handleDownloadCode} style={{ marginBottom: '10px' }}>
                        Download Code
                    </button>
                    <pre style={{ background: '#f5f5f5', padding: '10px', overflowX: 'auto' }}>
                        {generatedCode}
                    </pre>
                    <pre style={{ background: '#f5f5f5', padding: '10px', overflowX: 'auto' }}>
                        {generatedYamlCode}
                    </pre>
                </>
            )}
        </div>
    );
};

export default CodeGenerator;
