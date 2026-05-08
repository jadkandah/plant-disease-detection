import { useEffect, useRef } from 'react';
import { AppState, AppStateStatus, Alert } from 'react-native';
import { useNetworkStatus } from '../network/useNetworkStatus';
import { syncOfflineQueue, getOfflineQueue } from '../offline/offlineQueue';
import { useModelMode } from '../../store/ModelModeContext';

/**
 * Hook that automatically syncs offline predictions when the device
 * comes back online or the app returns to foreground.
 */
export function useAutoSync() {
  const { isConnected } = useNetworkStatus();
  const { isOnlineMode } = useModelMode();
  const prevConnected = useRef<boolean>(isConnected);

  // Sync when network reconnects
  useEffect(() => {
    if (isConnected && isOnlineMode && !prevConnected.current) {
      attemptSync();
    }
    prevConnected.current = isConnected;
  }, [isConnected, isOnlineMode]);

  // Sync when the tester switches back from local/offline mode.
  useEffect(() => {
    if (isConnected && isOnlineMode) {
      attemptSync();
    }
  }, [isConnected, isOnlineMode]);

  // Sync when app comes to foreground
  useEffect(() => {
    const handleAppStateChange = (nextState: AppStateStatus) => {
      if (nextState === 'active' && isConnected && isOnlineMode) {
        attemptSync();
      }
    };

    const sub = AppState.addEventListener('change', handleAppStateChange);
    return () => sub.remove();
  }, [isConnected, isOnlineMode]);

  const attemptSync = async () => {
    try {
      const queue = await getOfflineQueue();
      if (queue.length === 0) return;

      const synced = await syncOfflineQueue();
      if (synced > 0) {
        Alert.alert('Sync Complete', `${synced} offline prediction(s) synced successfully!`);
      }
    } catch (error) {
      console.log('Auto-sync failed:', error);
    }
  };
}
