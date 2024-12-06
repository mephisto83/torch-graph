// App.tsx
import React from 'react';
import Canvas from './components/Canvas';
import Sidebar from './components/Sidebar';
import CodeGenerator from './components/CodeGenerator';
import { Node, Edge } from 'reactflow';
import './App.css';
import { GraphProvider } from './provider/GraphProvider';

const App: React.FC = () => {
  const [nodes, setNodes] = React.useState<Node[]>([]);
  const [edges, setEdges] = React.useState<Edge[]>([]);
  const [selectedNode, setSelectedNode] = React.useState<Node | null>(null);

  return (
    <GraphProvider>
      <div className="app" style={{ display: 'flex', height: '100vh' }}>
        <Sidebar />
        <Canvas />
        <CodeGenerator />
      </div>
    </GraphProvider>
  );
};

export default App;
