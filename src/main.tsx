import React from 'react';
import ReactDOM from 'react-dom/client';
import loadBlake3 from 'blake3/browser-async';
import wasmUrl from 'blake3/dist/wasm/web/blake3_js_bg.wasm?url';
import App from './App';
import './styles.css';

const container = document.getElementById('root');

if (!container) {
  throw new Error('Failed to find application root element');
}

const bootstrap = async () => {
  try {
    // Ensure the BLAKE3 wasm module is registered before any hashing occurs.
    await loadBlake3(wasmUrl);
  } catch (error) {
    console.error('Failed to initialize BLAKE3 WebAssembly module', error);
    throw error;
  }

  ReactDOM.createRoot(container).render(
    <React.StrictMode>
      <App />
    </React.StrictMode>,
  );
};

void bootstrap();
