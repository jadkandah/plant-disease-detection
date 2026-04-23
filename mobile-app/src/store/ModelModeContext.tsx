import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';

type ModelMode = 'online' | 'offline';

interface ModelModeContextType {
  modelMode: ModelMode;
  setModelMode: (mode: ModelMode) => void;
  isOnlineMode: boolean;
}

const MODEL_MODE_KEY = 'app_model_mode';

export const ModelModeContext = createContext<ModelModeContextType>({
  modelMode: 'online',
  setModelMode: () => {},
  isOnlineMode: true,
});

export const ModelModeProvider = ({ children }: { children: ReactNode }) => {
  const [modelMode, setModelModeState] = useState<ModelMode>('online');

  useEffect(() => {
    loadModelMode();
  }, []);

  const loadModelMode = async () => {
    try {
      const saved = await AsyncStorage.getItem(MODEL_MODE_KEY);
      if (saved === 'online' || saved === 'offline') {
        setModelModeState(saved);
      }
    } catch {
      // Default to online
    }
  };

  const setModelMode = async (mode: ModelMode) => {
    setModelModeState(mode);
    await AsyncStorage.setItem(MODEL_MODE_KEY, mode);
  };

  return (
    <ModelModeContext.Provider value={{ modelMode, setModelMode, isOnlineMode: modelMode === 'online' }}>
      {children}
    </ModelModeContext.Provider>
  );
};

/**
 * Convenience hook for accessing model mode.
 */
export const useModelMode = () => useContext(ModelModeContext);
