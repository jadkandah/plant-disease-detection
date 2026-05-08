import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useNetworkStatus } from '../services/network/useNetworkStatus';

type ModelMode = 'online' | 'offline';

interface ModelModeContextType {
  modelMode: ModelMode;
  selectedModelMode: ModelMode;
  setModelMode: (mode: ModelMode) => Promise<void>;
  isOnlineMode: boolean;
  canUseOnlineMode: boolean;
}

const MODEL_MODE_KEY = 'app_model_mode';

export const ModelModeContext = createContext<ModelModeContextType>({
  modelMode: 'online',
  selectedModelMode: 'online',
  setModelMode: async () => {},
  isOnlineMode: true,
  canUseOnlineMode: true,
});

export const ModelModeProvider = ({ children }: { children: ReactNode }) => {
  const [selectedModelMode, setSelectedModelMode] = useState<ModelMode>('online');
  const { isConnected, isInternetReachable } = useNetworkStatus();
  const canUseOnlineMode = isConnected && isInternetReachable !== false;
  const modelMode: ModelMode = canUseOnlineMode ? selectedModelMode : 'offline';

  useEffect(() => {
    loadModelMode();
  }, []);

  const loadModelMode = async () => {
    try {
      const saved = await AsyncStorage.getItem(MODEL_MODE_KEY);
      if (saved === 'online' || saved === 'offline') {
        setSelectedModelMode(saved);
      }
    } catch {
      // Default to online
    }
  };

  const setModelMode = async (mode: ModelMode) => {
    if (mode === 'online' && !canUseOnlineMode) return;
    setSelectedModelMode(mode);
    await AsyncStorage.setItem(MODEL_MODE_KEY, mode);
  };

  return (
    <ModelModeContext.Provider
      value={{
        modelMode,
        selectedModelMode,
        setModelMode,
        isOnlineMode: modelMode === 'online',
        canUseOnlineMode,
      }}
    >
      {children}
    </ModelModeContext.Provider>
  );
};

/**
 * Convenience hook for accessing model mode.
 */
export const useModelMode = () => useContext(ModelModeContext);
