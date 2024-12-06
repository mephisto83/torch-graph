// components/Sidebar.tsx
import React, { useRef } from 'react';
import { Node } from 'reactflow';

import Accordion from '@mui/material/Accordion';
import AccordionSummary from '@mui/material/AccordionSummary';
import AccordionDetails from '@mui/material/AccordionDetails';
import Typography from '@mui/material/Typography';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import { LayerParameter, layerParameters } from '../utils/layerParameters';
import { useGraph } from '../provider/GraphProvider';


// Categories
const layerCategories: { [category: string]: string[] } = {
    "Input/Output": ["Input", "Output"],
    "Convolutional": ["Conv1d", "Conv2d", "Conv3d", "ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d"],
    "Linear & Embedding": ["Linear", "Bilinear", "Embedding", "EmbeddingBag"],
    "Recurrent": ["RNN", "LSTM", "GRU"],
    "Normalization": ["BatchNorm1d", "BatchNorm2d", "BatchNorm3d", "GroupNorm", "LayerNorm", "InstanceNorm1d", "InstanceNorm2d", "InstanceNorm3d"],
    "Activation": ["ReLU", "LeakyReLU", "ELU", "SELU", "Sigmoid", "Tanh", "Softmax", "LogSoftmax", "Hardtanh", "Hardshrink", "Hardsigmoid", "Hardswish", "Mish", "GELU"],
    "Pooling": ["MaxPool1d", "MaxPool2d", "MaxPool3d", "AvgPool1d", "AvgPool2d", "AvgPool3d", "AdaptiveMaxPool1d", "AdaptiveMaxPool2d", "AdaptiveMaxPool3d", "AdaptiveAvgPool1d", "AdaptiveAvgPool2d", "AdaptiveAvgPool3d"],
    "Dropout & Regularization": ["Dropout", "Dropout2d", "Dropout3d", "AlphaDropout"],
    "Padding": ["ReflectionPad1d", "ReflectionPad2d", "ReflectionPad3d", "ReplicationPad1d", "ReplicationPad2d", "ReplicationPad3d", "ZeroPad2d"],
    "Upsampling & Resizing": ["Upsample", "UpsamplingNearest2d", "UpsamplingBilinear2d"],
    "Transformers & Attention": ["Transformer", "TransformerEncoder", "TransformerDecoder", "TransformerEncoderLayer", "TransformerDecoderLayer", "MultiheadAttention"],
    "Reshaping & Folding": ["Flatten", "Unfold", "Fold"],
    "Other": ["PixelShuffle", "ChannelShuffle", "Softmax2d"],
    //custom
    "rapstar": ["MLP"],
    "Utilities": ["Cat", "Sequential"], // New category for utility operations
};

interface SidebarProps {
    nodes: Node[];
    setNodes: React.Dispatch<React.SetStateAction<Node[]>>;
    selectedNode: Node | null;
    setSelectedNode: React.Dispatch<React.SetStateAction<Node | null>>;
}


const Sidebar: React.FC = () => {
    const {
        nodes,
        setNodes,
        selectedNode,
        setSelectedNode,
        modelName,
        setModelName,
        saveGraph,
        loadGraph,
    } = useGraph();

    const fileInputRef = useRef<HTMLInputElement>(null);

    const onDragStart = (
        event: React.DragEvent<HTMLDivElement>,
        nodeType: string
    ) => {
        event.dataTransfer.setData('application/reactflow', nodeType);
        event.dataTransfer.effectAllowed = 'move';
    };

    const handleNameChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        if (selectedNode) {
            const newName = event.target.value;
            const updatedNode = {
                ...selectedNode,
                data: {
                    ...selectedNode.data,
                    label: newName,
                },
            };

            setNodes((nds) =>
                nds.map((node) => (node.id === selectedNode.id ? updatedNode : node))
            );
            setSelectedNode(updatedNode);
        }
    };

    const handleParameterChange = (event: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
        if (selectedNode) {
            const { name, value, type, checked } = event.target as HTMLInputElement;
            let paramValue: any = value;
            if (type === 'number') {
                paramValue = Number(value);
            } else if (type === 'checkbox') {
                paramValue = checked;
            }

            const updatedParameters = {
                ...selectedNode.data.parameters,
                [name]: paramValue,
            };

            const updatedNode = {
                ...selectedNode,
                data: {
                    ...selectedNode.data,
                    parameters: updatedParameters,
                },
            };

            setNodes((nds) =>
                nds.map((node) => (node.id === selectedNode.id ? updatedNode : node))
            );
            setSelectedNode(updatedNode);
        }
    };

    const handleModelNameChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        setModelName(event.target.value);
    };

    const handleSaveGraph = () => {
        saveGraph();
    };

    const handleLoadGraph = () => {
        fileInputRef.current?.click();
    };

    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (file) {
            loadGraph(file);
            // Reset the file input
            event.target.value = '';
        }
    };

    const renderParameterFields = () => {
        if (!selectedNode || selectedNode.type === 'edge') return null;
        if (!selectedNode.type) throw 'selectedNode.type empty'
        const params = layerParameters[selectedNode.type] || [];
        if (params.length === 0) return null;

        return (
            <>
                <h4 style={{ marginTop: '20px' }}>Parameters</h4>
                {params.map((param) => {
                    const currentValue = selectedNode.data.parameters?.[param.name] ?? param.default;
                    if (param.type === 'boolean') {
                        return (
                            <label key={param.name} style={{ display: 'block', marginTop: '10px' }}>
                                {param.name}:
                                <input
                                    type="checkbox"
                                    name={param.name}
                                    checked={!!currentValue}
                                    onChange={handleParameterChange}
                                    style={{ marginLeft: '5px' }}
                                />
                            </label>
                        );
                    } else if (param.type === 'select' && param.options) {
                        return (
                            <label key={param.name} style={{ display: 'block', marginTop: '10px' }}>
                                {param.name}:
                                <select
                                    name={param.name}
                                    value={currentValue}
                                    onChange={handleParameterChange}
                                    style={{ marginLeft: '5px' }}
                                >
                                    {param.options.map((opt) => (
                                        <option key={opt} value={opt}>{opt}</option>
                                    ))}
                                </select>
                            </label>
                        );
                    } else {
                        // number or text
                        const inputType = param.type === 'number' ? 'number' : 'text';
                        return (
                            <label key={param.name} style={{ display: 'block', marginTop: '10px' }}>
                                {param.name}:
                                <input
                                    type={inputType}
                                    name={param.name}
                                    value={currentValue}
                                    onChange={handleParameterChange}
                                    style={{ marginLeft: '5px', width: '150px' }}
                                />
                            </label>
                        );
                    }
                })}
            </>
        );
    };

    return (
        <aside style={{ padding: '10px', overflowY: 'auto', width: '300px', borderRight: '1px solid #ccc' }}>
            <h2>PyTorch Layers</h2>
            {Object.entries(layerCategories).map(([category, layers]) => (
                <Accordion key={category} disableGutters>
                    <AccordionSummary
                        expandIcon={<ExpandMoreIcon />}
                        aria-controls={`${category}-content`}
                        id={`${category}-header`}
                    >
                        <Typography variant="subtitle1" style={{ fontWeight: 'bold' }}>
                            {category}
                        </Typography>
                    </AccordionSummary>
                    <AccordionDetails style={{ display: 'flex', flexDirection: 'column' }}>
                        {layers.map((layerType) => (
                            <div
                                key={layerType}
                                onDragStart={(event) => onDragStart(event, layerType)}
                                draggable
                                style={{
                                    cursor: 'grab',
                                    padding: '5px',
                                    border: '1px solid #ccc',
                                    borderRadius: '3px',
                                    background: '#f9f9f9',
                                    marginBottom: '5px'
                                }}
                            >
                                {layerType}
                            </div>
                        ))}
                    </AccordionDetails>
                </Accordion>
            ))}

            <div style={{ marginTop: '20px' }}>
                <h3>Model Name</h3>
                <input
                    type="text"
                    value={modelName}
                    onChange={handleModelNameChange}
                    style={{ width: '100%', padding: '5px' }}
                />
            </div>

            {selectedNode && selectedNode.type !== 'edge' && (
                <div className="node-editor" style={{ marginTop: '20px' }}>
                    <h3>Edit Node</h3>
                    <label>
                        Name:
                        <input
                            type="text"
                            value={selectedNode.data.label}
                            onChange={handleNameChange}
                            style={{ marginLeft: '5px', width: '150px' }}
                        />
                    </label>
                    {renderParameterFields()}
                </div>
            )}

            <div style={{ marginTop: '20px' }}>
                <button onClick={handleSaveGraph} style={{ marginRight: '10px' }}>
                    Save Graph
                </button>
                <button onClick={handleLoadGraph}>
                    Load Graph
                </button>
                <input
                    type="file"
                    accept=".json"
                    ref={fileInputRef}
                    style={{ display: 'none' }}
                    onChange={handleFileChange}
                />
            </div>
        </aside>
    );
};
export default Sidebar;
