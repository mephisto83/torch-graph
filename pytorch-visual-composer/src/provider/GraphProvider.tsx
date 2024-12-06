// context/GraphContext.tsx
import React, { createContext, useContext, useState, ReactNode } from 'react';
import { Node, Edge } from 'reactflow';

// Define the shape of the context
interface GraphContextProps {
    nodes: Node[];
    edges: Edge[];
    selectedNode: Node | null;
    modelName: string;
    setNodes: React.Dispatch<React.SetStateAction<Node[]>>;
    setEdges: React.Dispatch<React.SetStateAction<Edge[]>>;
    setSelectedNode: React.Dispatch<React.SetStateAction<Node | null>>;
    setModelName: React.Dispatch<React.SetStateAction<string>>;
    saveGraph: () => void;
    loadGraph: (file: File) => void;
}

// Create the Context
const GraphContext = createContext<GraphContextProps | undefined>(undefined);

// Create the Provider Component
export const GraphProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
    const [nodes, setNodes] = useState<Node[]>([]);
    const [edges, setEdges] = useState<Edge[]>([]);
    const [selectedNode, setSelectedNode] = useState<Node | null>(null);
    const [modelName, setModelName] = useState<string>('GeneratedModel');

    // Function to save the graph as a JSON file
    const saveGraph = () => {
        const graph = { nodes, edges, modelName };
        const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(graph, null, 2));
        const downloadAnchorNode = document.createElement('a');
        downloadAnchorNode.setAttribute("href", dataStr);
        downloadAnchorNode.setAttribute("download", `${modelName}_graph.json`);
        document.body.appendChild(downloadAnchorNode); // Required for Firefox
        downloadAnchorNode.click();
        downloadAnchorNode.remove();
    };

    // Function to load the graph from a JSON file
    const loadGraph = (file: File) => {
        const reader = new FileReader();
        reader.onload = (event) => {
            try {
                const result = event.target?.result;
                if (typeof result === 'string') {
                    const graph = JSON.parse(result);
                    setNodes(graph.nodes || []);
                    setEdges(graph.edges || []);
                    setModelName(graph.modelName || 'GeneratedModel');
                }
            } catch (error) {
                console.error("Failed to load graph:", error);
                alert("Failed to load graph. Please ensure the file is a valid JSON.");
            }
        };
        reader.readAsText(file);
    };

    return (
        <GraphContext.Provider value={{
            nodes,
            edges,
            selectedNode,
            modelName,
            setNodes,
            setEdges,
            setSelectedNode,
            setModelName,
            saveGraph,
            loadGraph,
        }}>
            {children}
        </GraphContext.Provider>
    );
};

// Custom Hook to use the Graph Context
export const useGraph = () => {
    const context = useContext(GraphContext);
    if (context === undefined) {
        throw new Error('useGraph must be used within a GraphProvider');
    }
    return context;
};
