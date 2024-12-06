// components/Canvas.tsx
import React, { useCallback } from 'react';
import ReactFlow, {
    ReactFlowProvider,
    addEdge,
    applyNodeChanges,
    applyEdgeChanges,
    Background,
    Controls,
    Connection,
    Edge,
    Node,
    NodeChange,
    EdgeChange,
} from 'reactflow';
import 'reactflow/dist/style.css';

import CustomNode from './CustomNode';

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
};


interface CanvasProps {
    nodes: Node[];
    edges: Edge[];
    setNodes: React.Dispatch<React.SetStateAction<Node[]>>;
    setEdges: React.Dispatch<React.SetStateAction<Edge[]>>;
    setSelectedNode: React.Dispatch<React.SetStateAction<Node | null>>;
}

const Canvas: React.FC<CanvasProps> = ({
    nodes,
    edges,
    setNodes,
    setEdges,
    setSelectedNode,
}) => {
    const onNodesChange = useCallback(
        (changes: NodeChange[]) => setNodes((nds) => applyNodeChanges(changes, nds)),
        [setNodes]
    );

    const onEdgesChange = useCallback(
        (changes: EdgeChange[]) => setEdges((eds) => applyEdgeChanges(changes, eds)),
        [setEdges]
    );

    const onConnect = useCallback(
        (params: Connection) => setEdges((eds) => addEdge(params, eds)),
        [setEdges]
    );

    const onDrop = useCallback(
        (event: React.DragEvent) => {
            event.preventDefault();

            const nodeType = event.dataTransfer.getData('application/reactflow');
            const position = {
                x: event.clientX - event.currentTarget.getBoundingClientRect().left,
                y: event.clientY - event.currentTarget.getBoundingClientRect().top,
            };
            const newNode: Node = {
                id: getId(),
                type: nodeType,
                position,
                data: {
                    label: `${nodeType} Node`,
                    parameters: {},
                    configKey: `node_${id}`,
                },
            };

            setNodes((nds) => nds.concat(newNode));
        },
        [setNodes]
    );

    const onDragOver = (event: React.DragEvent) => {
        event.preventDefault();
        event.dataTransfer.dropEffect = 'move';
    };

    const onNodeClick = useCallback(
        (event: React.MouseEvent, node: Node) => {
            setSelectedNode(node);
        },
        [setSelectedNode]
    );

    return (
        <ReactFlowProvider>
            <div className="canvas-wrapper" style={{ flex: 1 }}>
                <ReactFlow
                    nodes={nodes}
                    edges={edges}
                    onNodesChange={onNodesChange}
                    onEdgesChange={onEdgesChange}
                    onConnect={onConnect}
                    onNodeClick={onNodeClick}
                    onDrop={onDrop}
                    onDragOver={onDragOver}
                    nodeTypes={nodeTypes}
                    fitView
                >
                    <Background />
                    <Controls />
                </ReactFlow>
            </div>
        </ReactFlowProvider>
    );
};

// Utility function to generate unique IDs
let id = 0;
const getId = () => `node_${id++}`;

export default Canvas;
