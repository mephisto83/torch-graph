// App.tsx
import React from 'react';
import Canvas from './components/Canvas';
import Sidebar from './components/Sidebar';
import CodeGenerator from './components/CodeGenerator';
import './App.css';
import { GraphProvider } from './provider/GraphProvider';

const App: React.FC = () => {
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
