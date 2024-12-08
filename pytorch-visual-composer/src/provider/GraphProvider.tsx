// context/GraphContext.tsx
import React, { createContext, useContext, useState, ReactNode, useEffect } from 'react';
import { Node, Edge } from 'reactflow';
import { nodeTypes } from '../utils/nodeTypes';
import { PytorchModuleConfig } from '../utils/customModules';
import { LayerParameter, layerParameters } from '../utils/layerParameters';
import { defaultLayerCategories } from '../utils/layerCategories';
import { defaultLayerToClassMap } from '../utils/layerToClassMap';
import CustomNode from '../components/CustomNode';

// Define the shape of the context
interface GraphContextProps {
    nodes: Node[];
    edges: Edge[];
    selectedNode: Node | null;
    selectedEdge: Edge | null;
    modelName: string;
    setNodes: React.Dispatch<React.SetStateAction<Node[]>>;
    setEdges: React.Dispatch<React.SetStateAction<Edge[]>>;
    setSelectedNode: React.Dispatch<React.SetStateAction<Node | null>>;
    setSelectedEdge: React.Dispatch<React.SetStateAction<Edge | null>>;
    setModelName: React.Dispatch<React.SetStateAction<string>>;
    saveGraph: () => void;
    loadGraph: (file: File) => void;
    nodeTypes: { [key: string]: any }
    // Local Storage Functions
    saveModuleToLocalStorage: (config: PytorchModuleConfig) => void;
    loadModelFromLocalStorage: (name: string) => void;
    getSavedModelNames: () => string[];
    deleteModelFromLocalStorage: (name: string) => void;
    layerParameters: { [key: string]: LayerParameter[] };
    layerCategories: { [category: string]: string[] };
    layerToClassMap: { [key: string]: string }
}

// Local Storage Key
const LOCAL_STORAGE_KEY = 'savedModels';
// Helper function to retrieve saved models from Local Storage
const getSavedModels = (): { [key: string]: PytorchModuleConfig } => {
    const saved = localStorage.getItem(LOCAL_STORAGE_KEY);
    if (saved) {
        try {
            return JSON.parse(saved);
        } catch (error) {
            console.error('Failed to parse saved models from Local Storage:', error);
            return {};
        }
    }
    return {};
};

// Helper function to save models back to Local Storage
const setSavedModels = (models: { [key: string]: PytorchModuleConfig }) => {
    try {
        localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(models, null, 2));
    } catch (error) {
        console.error('Failed to save models to Local Storage:', error);
    }
};
// Create the Context
const GraphContext = createContext<GraphContextProps | undefined>(undefined);

// Create the Provider Component
export const GraphProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
    const [nodes, setNodes] = useState<Node[]>([]);
    const [edges, setEdges] = useState<Edge[]>([]);
    const [selectedNode, setSelectedNode] = useState<Node | null>(null);
    const [selectedEdge, setSelectedEdge] = useState<Edge | null>(null);
    const [graphNodeTypes, setGraphNodeTypes] = useState<{ [key: string]: any }>({});
    const [graphLayerParameters, setGraphLayerParameters] = useState<{ [key: string]: any }>({});
    const [graphLayerCategories, setGraphLayerCategories] = useState<{ [category: string]: string[] }>({});
    const [graphLayerToClassMap, setGraphLayerToClassMap] = useState<{ [key: string]: string }>({});
    const [modelsUpdated, setModelsUpdated] = useState(0);
    const [modelName, setModelName] = useState<string>('GeneratedModel');
    useEffect(() => {
        let types = nodeTypes();
        let models = loadAllModelsFromLocalStorage();
        let update: any = { ...types };
        models.forEach(model => {
            if (model) {
                let temp: any = {};
                Object.keys(model.nodeTypes).map((key) => {
                    temp[key] = CustomNode
                })
                update = {
                    ...update,
                    ...temp
                }
            }
        })

        setGraphNodeTypes(update);
    }, [modelsUpdated, JSON.stringify(Object.keys(nodeTypes()))])
    useEffect(() => {
        let params = layerParameters();
        let models = loadAllModelsFromLocalStorage();
        let update: any = { ...params };
        models.forEach(model => {
            if (model) {
                update = {
                    ...update,
                    ...model.layerParameters
                }
            }
        })
        setGraphLayerParameters(update);
    }, [modelsUpdated, JSON.stringify(Object.keys(layerParameters()))])
    useEffect(() => {
        let models = loadAllModelsFromLocalStorage();
        let update: any = { ...defaultLayerCategories };
        models.forEach(model => {
            if (model) {
                Object.entries(model.layerCategories).map(([key, value]) => {
                    if (update[key]) {
                        let temp = [...update[key], value]
                        update[key] = temp;
                    }
                    else {
                        update[key] = [value];
                    }
                })
                update = {
                    ...update,
                }
            }
        })
        setGraphLayerCategories({ ...update })
    }, [modelsUpdated])
    useEffect(() => {
        let models = loadAllModelsFromLocalStorage();
        let update = {};
        models.forEach(model => {
            update = {
                ...update,
                ...model?.layerToClassMap
            }
        })
        setGraphLayerToClassMap({ ...defaultLayerToClassMap, ...update })
    }, [modelsUpdated])

    useEffect(() => {
        let models = loadAllModelsFromLocalStorage();
        setModelsUpdated(Date.now())
        console.log(models);
    }, [])

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
    // Local Storage Functions

    /**
     * Save the current graph model to Local Storage under the specified name.
     * If the name already exists, it will overwrite the existing model.
     * @param name Unique name for the model
     */
    const saveModuleToLocalStorage = (config: PytorchModuleConfig) => {
        const models = getSavedModels();
        setSavedModels({
            ...models,
            [config.name]: config
        });
        console.log(`Pytorch Module Config "${config.name}" saved to Local Storage.`);
        setModelsUpdated(Date.now());
    };

    /**
     * Load a graph model from Local Storage by its name.
     * @param name Name of the model to load
     */
    const loadModelFromLocalStorage = (name: string): PytorchModuleConfig | null => {
        const models = getSavedModels();
        const model = models[name];
        if (model) {
            console.log(`Pytorch Module Config "${name}" loaded from Local Storage.`);
            return model
        } else {
            console.warn(`Pytorch Module Config "${name}" does not exist in Local Storage.`);
            alert(`Pytorch Module Config "${name}" not found in Local Storage.`);
        }
        return null;
    };
    const loadAllModelsFromLocalStorage = () => {
        let modelNames = getSavedModelNames();
        let models = modelNames.map(modelName => {
            return loadModelFromLocalStorage(modelName)
        }).filter(x => x)
        return models;
    }

    /**
     * Retrieve all saved model names from Local Storage.
     * @returns Array of model names
     */
    const getSavedModelNames = (): string[] => {
        const models = getSavedModels();
        return Object.keys(models);
    };

    /**
     * Delete a specific model from Local Storage by its name.
     * @param name Name of the model to delete
     */
    const deleteModelFromLocalStorage = (name: string) => {
        const models = getSavedModels();
        if (models[name]) {
            delete models[name];
            setSavedModels(models);
            console.log(`Pytorch Module Config "${name}" deleted from Local Storage.`);
        } else {
            console.warn(`Pytorch Module Config "${name}" does not exist in Local Storage.`);
            alert(`Pytorch Module Config "${name}" not found in Local Storage.`);
        }
    };

    return (
        <GraphContext.Provider value={{
            nodes,
            edges,
            selectedNode,
            selectedEdge,
            modelName,
            setNodes,
            setEdges,
            setSelectedNode,
            setSelectedEdge,
            setModelName,
            saveGraph,
            loadGraph,
            nodeTypes: graphNodeTypes,
            layerParameters: graphLayerParameters,
            layerCategories: graphLayerCategories,
            layerToClassMap: graphLayerToClassMap,
            // Local Storage Functions
            saveModuleToLocalStorage,
            loadModelFromLocalStorage,
            getSavedModelNames,
            deleteModelFromLocalStorage,
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
