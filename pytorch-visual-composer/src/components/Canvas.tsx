// components/Canvas.tsx
import React, { useCallback } from 'react';
import ReactFlow, {
    addEdge,
    MiniMap,
    Controls,
    Background,
    applyNodeChanges,
    applyEdgeChanges,
    Node,
    Edge,
    ReactFlowProvider,
    OnConnect,
    Position,
    NodeChange,
    EdgeChange,
} from 'reactflow';
import 'reactflow/dist/style.css';

import CustomNode from './CustomNode';
import { useGraph } from '../provider/GraphProvider';
import { layerToClassMap } from '../utils/codeGenerator';

const nodeTypes = {
    // Basic I/O and structural nodes
    Input: CustomNode,
    Output: CustomNode,

    // Convolutional layers
    Conv1d: CustomNode,
    Conv2d: CustomNode,
    Conv3d: CustomNode,
    ConvTranspose1d: CustomNode,
    ConvTranspose2d: CustomNode,
    ConvTranspose3d: CustomNode,

    // Linear and embedding layers
    Linear: CustomNode,
    Bilinear: CustomNode,
    Embedding: CustomNode,
    EmbeddingBag: CustomNode,

    // Recurrent layers
    RNN: CustomNode,
    LSTM: CustomNode,
    GRU: CustomNode,

    // Normalization layers
    BatchNorm1d: CustomNode,
    BatchNorm2d: CustomNode,
    BatchNorm3d: CustomNode,
    GroupNorm: CustomNode,
    LayerNorm: CustomNode,
    InstanceNorm1d: CustomNode,
    InstanceNorm2d: CustomNode,
    InstanceNorm3d: CustomNode,

    // Activation layers
    ReLU: CustomNode,
    LeakyReLU: CustomNode,
    ELU: CustomNode,
    SELU: CustomNode,
    Sigmoid: CustomNode,
    Tanh: CustomNode,
    Softmax: CustomNode,
    LogSoftmax: CustomNode,
    Hardtanh: CustomNode,
    Hardshrink: CustomNode,
    Hardsigmoid: CustomNode,
    Hardswish: CustomNode,
    Mish: CustomNode,
    GELU: CustomNode,

    // Pooling layers
    MaxPool1d: CustomNode,
    MaxPool2d: CustomNode,
    MaxPool3d: CustomNode,
    AvgPool1d: CustomNode,
    AvgPool2d: CustomNode,
    AvgPool3d: CustomNode,
    AdaptiveMaxPool1d: CustomNode,
    AdaptiveMaxPool2d: CustomNode,
    AdaptiveMaxPool3d: CustomNode,
    AdaptiveAvgPool1d: CustomNode,
    AdaptiveAvgPool2d: CustomNode,
    AdaptiveAvgPool3d: CustomNode,

    // Dropout and other regularization layers
    Dropout: CustomNode,
    Dropout2d: CustomNode,
    Dropout3d: CustomNode,
    AlphaDropout: CustomNode,

    // Padding layers
    ReflectionPad1d: CustomNode,
    ReflectionPad2d: CustomNode,
    ReflectionPad3d: CustomNode,
    ReplicationPad1d: CustomNode,
    ReplicationPad2d: CustomNode,
    ReplicationPad3d: CustomNode,
    ZeroPad2d: CustomNode,

    // Upsampling and resizing layers
    Upsample: CustomNode,
    UpsamplingNearest2d: CustomNode,
    UpsamplingBilinear2d: CustomNode,

    // Transformer and attention layers
    Transformer: CustomNode,
    TransformerEncoder: CustomNode,
    TransformerDecoder: CustomNode,
    TransformerEncoderLayer: CustomNode,
    TransformerDecoderLayer: CustomNode,
    MultiheadAttention: CustomNode,

    // Utility layers for reshaping and folding
    Flatten: CustomNode,
    Unfold: CustomNode,
    Fold: CustomNode,

    // Other miscellaneous layers
    PixelShuffle: CustomNode,
    ChannelShuffle: CustomNode, // Often implemented manually, but can be included as a conceptual node
    Softmax2d: CustomNode,
    // ... Add more if required

    MLP: CustomNode,
    CAT: CustomNode,
    Sequential: CustomNode,
};

const Canvas: React.FC = () => {
    const { nodes, setNodes, edges, setEdges, setSelectedNode } = useGraph();

    const onConnect: OnConnect = (params) => setEdges((eds) => addEdge(params, eds));

    const onNodesChange = useCallback(
        (changes: NodeChange[]) => setNodes((nds) => applyNodeChanges(changes, nds)),
        [setNodes]
    );

    const onEdgesChange = useCallback(
        (changes: EdgeChange[]) => setEdges((eds) => applyEdgeChanges(changes, eds)),
        [setEdges]
    );

    const onNodeClick = (event: React.MouseEvent, node: Node) => {
        setSelectedNode(node);
    };

    const onDrop = useCallback(
        (event: React.DragEvent) => {
            event.preventDefault();

            const reactFlowBounds = (event.target as HTMLDivElement).getBoundingClientRect();
            const type = event.dataTransfer.getData('application/reactflow');
            if (!type) return;

            // Check if the dropped element is valid
            const validTypes = Object.keys(nodeTypes);
            if (!validTypes.includes(type) && !layerToClassMap[type]) {
                return;
            }

            // Get the position where the node was dropped
            const position = {
                x: event.clientX - reactFlowBounds.left,
                y: event.clientY - reactFlowBounds.top,
            };

            // Generate a unique ID for the new node
            const id = getId();

            // Define the new node
            const newNode: Node = {
                id,
                type: 'custom', // Use custom node type
                position,
                data: { label: type, parameters: {} },
            };

            setNodes((nds) => nds.concat(newNode));
        },
        [setNodes]
    );

    const onDragOver = useCallback((event: React.DragEvent) => {
        event.preventDefault();
        event.dataTransfer.dropEffect = 'move';
    }, []);

    return (
        <div style={{ height: '100%', flexGrow: 1 }}>
            <ReactFlow
                nodes={nodes}
                edges={edges}
                onNodesChange={onNodesChange}
                onEdgesChange={onEdgesChange}
                onConnect={onConnect}
                nodeTypes={nodeTypes}
                onNodeClick={onNodeClick}
                onDrop={onDrop}
                onDragOver={onDragOver}
                fitView
            >
                <MiniMap />
                <Controls />
                <Background />
            </ReactFlow>
        </div>
    );
};

// Utility function to generate unique IDs
let id = 0;
const getId = () => `node_${id++}`;

export default Canvas;
