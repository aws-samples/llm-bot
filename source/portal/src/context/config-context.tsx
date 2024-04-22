import React from 'react';
export interface Config {
  websocket: string;
  apiUrl: string;
  docsS3Bucket: string;
  workspaceId: string;
}
const ConfigContext = React.createContext<Config | null>(null);
export default ConfigContext;