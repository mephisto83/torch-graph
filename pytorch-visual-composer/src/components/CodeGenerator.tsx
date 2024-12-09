// components/CodeGenerator.tsx
import React, { useState } from 'react';
import { generateCode } from '../utils/codeGenerator';
import { useGraph } from '../provider/GraphProvider';
import { Button } from '@mui/material';

const CodeGenerator: React.FC = () => {
    const { nodes, edges, modelName, layerToClassMap, layerParameters } = useGraph();
    const [generatedCode, setGeneratedCode] = useState<string>('');
    const [generatedYamlCode, setGeneratedYamlCode] = useState<string>('');

    const handleGenerateCode = () => {
        const { code, yamlConfig } = generateCode(nodes, edges, modelName, layerToClassMap, layerParameters);
        setGeneratedCode(code);
        setGeneratedYamlCode(yamlConfig)
    };
    async function copyToClipboard(text: string): Promise<void> {
        if (navigator.clipboard && typeof navigator.clipboard.writeText === "function") {
            try {
                await navigator.clipboard.writeText(text);
                console.log("Text copied to clipboard successfully!");
            } catch (err) {
                console.error("Failed to copy text to clipboard: ", err);
            }
        } else {
            // Fallback for older browsers
            const textarea = document.createElement("textarea");
            textarea.value = text;
            textarea.style.position = "fixed"; // Prevent scrolling
            textarea.style.opacity = "0"; // Make it invisible
            document.body.appendChild(textarea);
            textarea.focus();
            textarea.select();

            try {
                const successful = document.execCommand("copy");
                if (successful) {
                    console.log("Text copied to clipboard successfully (fallback)!");
                } else {
                    throw new Error("Fallback copy failed");
                }
            } catch (err) {
                console.error("Fallback: Failed to copy text to clipboard: ", err);
            } finally {
                document.body.removeChild(textarea);
            }
        }
    }

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
            <Button
                variant="contained"
                color="primary" onClick={handleGenerateCode} style={{ marginBottom: '10px' }}>
                Generate Code
            </Button>
            {generatedCode && (
                <>
                    <Button
                        variant="contained"
                        color="primary" onClick={handleDownloadCode} style={{ marginBottom: '10px' }}>
                        Download Code
                    </Button>
                    <Button
                        variant="contained"
                        color="primary" onClick={async () => {
                            await copyToClipboard(generatedCode)
                        }} style={{ marginBottom: '10px' }}>
                        Copy to clipboard
                    </Button>
                    <pre style={{ background: '#f5f5f5', padding: '10px', overflowX: 'auto' }}>
                        {generatedCode}
                    </pre>
                    <Button
                        variant="contained"
                        color="primary" onClick={async () => {
                            await copyToClipboard(generatedYamlCode)
                        }} style={{ marginBottom: '10px' }}>
                        Copy Yaml to clipboard
                    </Button>
                    <pre style={{ background: '#f5f5f5', padding: '10px', overflowX: 'auto' }}>
                        {generatedYamlCode}
                    </pre>
                </>
            )}
        </div>
    );
};

export default CodeGenerator;
