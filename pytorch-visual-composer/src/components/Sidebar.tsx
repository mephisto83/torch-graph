// components/Sidebar.tsx
import React, { useRef } from 'react';
import { Node } from 'reactflow';

import Accordion from '@mui/material/Accordion';
import AccordionSummary from '@mui/material/AccordionSummary';
import AccordionDetails from '@mui/material/AccordionDetails';
import Typography from '@mui/material/Typography';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import { useGraph } from '../provider/GraphProvider';
import ClipboardPaster from './ClipboardPaster';
import GraphControls from './GraphControls';
import { validateConfig } from '../utils/customModules';

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
        layerCategories,
        saveModuleToLocalStorage,
        layerParameters
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
        event.preventDefault();
        event.stopPropagation();
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
        event.preventDefault();
        event.stopPropagation();
    };

    const handleModelNameChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        setModelName(event.target.value);
        event.preventDefault();
        event.stopPropagation();
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
            <img src="/logo.png" style={{ maxWidth: 'calc(100% - 2px)' }} />
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
                    <h4>{selectedNode.type}</h4>
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
            <GraphControls handleSaveGraph={handleSaveGraph} handleFileChange={handleFileChange} />

            <div style={{ marginTop: '20px' }}>
                <ClipboardPaster onModelPasted={(model) => {
                    console.log(model);
                    const { valid, errors } = validateConfig(model);
                    if (valid) {
                        saveModuleToLocalStorage(model)
                    }
                    console.error(errors);
                }} />
            </div>
        </aside>
    );
};
export default Sidebar;
