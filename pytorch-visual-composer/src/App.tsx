// App.tsx
import React from 'react';
import Canvas from './components/Canvas';
import Sidebar from './components/Sidebar';
import CodeGenerator from './components/CodeGenerator';
import { Node, Edge } from 'reactflow';
import './App.css';

const App: React.FC = () => {
  const [nodes, setNodes] = React.useState<Node[]>([]);
  const [edges, setEdges] = React.useState<Edge[]>([]);
  const [selectedNode, setSelectedNode] = React.useState<Node | null>(null);

  return (
    <div className="app">
      <Sidebar
        nodes={nodes}
        setNodes={setNodes}
        selectedNode={selectedNode}
        setSelectedNode={setSelectedNode}
      />
      <Canvas
        nodes={nodes}
        edges={edges}
        setNodes={setNodes}
        setEdges={setEdges}
        setSelectedNode={setSelectedNode}
      />
      <CodeGenerator nodes={nodes} edges={edges} />
    </div>
  );
};

export default App;
