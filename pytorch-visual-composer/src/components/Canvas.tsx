// components/Canvas.tsx
import React, { useCallback, useEffect, useRef } from 'react';
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
import { useGraph } from '../provider/GraphProvider';


const Canvas: React.FC = () => {
    const { nodes, setNodes, edges, setEdges, setSelectedNode, setSelectedEdge, selectedEdge, selectedNode, nodeTypes, layerToClassMap, layerParameters } = useGraph();
    const divRef = useRef(null);
    const onConnect: OnConnect = (params) => {

        setEdges((eds) => addEdge(params, eds))
    };

    const onNodesChange = useCallback(
        (changes: NodeChange[]) => setNodes((nds) => applyNodeChanges(changes, nds)),
        [setNodes]
    );

    const onEdgesChange = useCallback(
        (changes: EdgeChange[]) => {
            setEdges((eds) => applyEdgeChanges(changes, eds))
        },
        [setEdges]
    );

    const onNodeClick = (event: React.MouseEvent, node: Node) => {
        setSelectedNode(node);
    };

    const onEdgeClick = (event: React.MouseEvent, edge: Edge) => {
        setSelectedEdge(edge);
    }

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
                type: type === 'ConfigNode' ? 'Config' : (type || 'custom'), // Use custom node type
                position,
                data: {
                    label: type,
                    parameters: {},
                    parameterNames: layerParameters[type]
                },
            };

            setNodes((nds) => nds.concat(newNode));
        },
        [setNodes, nodeTypes, layerToClassMap, layerParameters]
    );

    const onDragOver = useCallback((event: React.DragEvent) => {
        event.preventDefault();
        event.dataTransfer.dropEffect = 'move';
    }, []);

    // Handle keydown events for deleting nodes
    useEffect(() => {
        const handleKeyDown = (event: KeyboardEvent) => {
            // Check if the Delete key is pressed
            if (event.key === 'Delete' || event.key === 'Backspace') {
                if (selectedEdge) {
                    // Remove all edges connected to the selected node
                    setEdges((eds) =>
                        eds.filter(
                            (edge) => edge.id !== selectedEdge.id
                        )
                    );
                    setSelectedEdge(null);
                }
                else if (selectedNode) {
                    // Remove the selected node
                    setNodes((nds) => nds.filter((node) => node.id !== selectedNode.id));

                    // Remove all edges connected to the selected node
                    setEdges((eds) =>
                        eds.filter(
                            (edge) => edge.source !== selectedNode.id && edge.target !== selectedNode.id
                        )
                    );

                    // Clear the selected node
                    setSelectedNode(null);
                }
            }
        };

        // Attach the event listener
        (divRef?.current as any)?.addEventListener('keydown', handleKeyDown);

        // Clean up the event listener on unmount
        return () => {
            (divRef?.current as any)?.removeEventListener('keydown', handleKeyDown);
        };
    }, [selectedNode, setNodes, setEdges, setSelectedNode]);
    return (
        <div style={{ height: '100%', flexGrow: 1 }} ref={divRef}>
            <ReactFlow
                nodes={nodes}
                edges={edges}
                onNodesChange={onNodesChange}
                onEdgesChange={onEdgesChange}
                onConnect={onConnect}
                nodeTypes={nodeTypes}
                onNodeClick={onNodeClick}
                onEdgeClick={onEdgeClick}
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
const getId = () => `node_${Date.now()}`;

export default Canvas;
function uuidv4(): string {
    // Helper function to generate a random number between 0 and 255
    const getRandomByte = (): number => Math.floor(Math.random() * 256);

    // Create an array of 16 random bytes
    const randomBytes = Array.from({ length: 16 }, getRandomByte);

    // Set the version to 4 -> (0100xxxx in bits)
    randomBytes[6] = (randomBytes[6] & 0x0f) | 0x40;

    // Set the variant to RFC 4122 -> (10xxxxxx in bits)
    randomBytes[8] = (randomBytes[8] & 0x3f) | 0x80;

    // Convert random bytes to UUID string format
    const byteToHex = (byte: number): string => byte.toString(16).padStart(2, '0');
    const uuid = randomBytes.map(byteToHex).join('');

    // Insert dashes into the UUID string to match the format
    return `${uuid.substring(0, 8)}-${uuid.substring(8, 12)}-${uuid.substring(12, 16)}-${uuid.substring(16, 20)}-${uuid.substring(20)}`;
}