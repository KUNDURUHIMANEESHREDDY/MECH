import './utils/ipcFetch';
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './styles.css';
import './design/styles/global.css';
import './plugins/registry';

const rootEl = document.getElementById('root');
if (!rootEl) throw new Error('root element not found');

ReactDOM.createRoot(rootEl).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
