import { useCallback, useEffect, useRef, useState } from 'react';
import { AppState, AppStateStatus } from 'react-native';
import { syncOfflineQueue, getOfflineQueue, subscribeOfflineQueue } from '../offline/offlineQueue';
import { useModelMode } from '../../store/ModelModeContext';

const SYNC_POLL_INTERVAL_MS = 7000;
const SYNC_SUCCESS_VISIBLE_MS = 5000;

export type SyncPhase = 'idle' | 'waiting' | 'syncing' | 'synced' | 'failed';

export interface AutoSyncState {
  phase: SyncPhase;
  pendingCount: number;
  syncedCount: number;
  visible: boolean;
}

const IDLE_SYNC_STATE: AutoSyncState = {
  phase: 'idle',
  pendingCount: 0,
  syncedCount: 0,
  visible: false,
};





export function useAutoSync(enabled = true): AutoSyncState {
  const { canUseOnlineMode } = useModelMode();
  const prevCanSync = useRef<boolean>(canUseOnlineMode);
  const isSyncing = useRef(false);
  const hideTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [syncState, setSyncState] = useState<AutoSyncState>(IDLE_SYNC_STATE);

  const clearHideTimer = useCallback(() => {
    if (hideTimer.current) {
      clearTimeout(hideTimer.current);
      hideTimer.current = null;
    }
  }, []);

  const showSyncedState = useCallback((syncedCount: number, remainingCount: number) => {
    clearHideTimer();
    setSyncState({
      phase: 'synced',
      pendingCount: remainingCount,
      syncedCount,
      visible: true,
    });

    hideTimer.current = setTimeout(() => {
      if (remainingCount > 0) {
        setSyncState({
          phase: 'waiting',
          pendingCount: remainingCount,
          syncedCount: 0,
          visible: true,
        });
      } else {
        setSyncState(IDLE_SYNC_STATE);
      }
    }, SYNC_SUCCESS_VISIBLE_MS);
  }, [clearHideTimer]);

  const attemptSync = useCallback(async () => {
    if (!enabled || !canUseOnlineMode || isSyncing.current) return;

    isSyncing.current = true;
    try {
      const queue = await getOfflineQueue();
      if (queue.length === 0) {
        setSyncState(IDLE_SYNC_STATE);
        return;
      }

      clearHideTimer();
      setSyncState({
        phase: 'syncing',
        pendingCount: queue.length,
        syncedCount: 0,
        visible: true,
      });

      const result = await syncOfflineQueue();
      if (result.synced > 0) {
        showSyncedState(result.synced, result.remaining);
      } else if (result.remaining > 0) {
        setSyncState({
          phase: 'failed',
          pendingCount: result.remaining,
          syncedCount: 0,
          visible: true,
        });
      } else {
        setSyncState(IDLE_SYNC_STATE);
      }
    } catch (error) {
      console.log('Auto-sync failed:', error);
      const remaining = await getOfflineQueue();
      setSyncState(
        remaining.length > 0
          ? {
              phase: 'failed',
              pendingCount: remaining.length,
              syncedCount: 0,
              visible: true,
            }
          : IDLE_SYNC_STATE
      );
    } finally {
      isSyncing.current = false;
    }
  }, [canUseOnlineMode, clearHideTimer, enabled, showSyncedState]);

  useEffect(() => {
    if (!enabled) {
      clearHideTimer();
      setSyncState(IDLE_SYNC_STATE);
      return;
    }

    return subscribeOfflineQueue((queue) => {
      if (isSyncing.current) {
        setSyncState((current) =>
          current.phase === 'syncing'
            ? { ...current, pendingCount: queue.length }
            : current
        );
        return;
      }

      if (queue.length === 0) {
        setSyncState((current) => (current.phase === 'synced' ? current : IDLE_SYNC_STATE));
        return;
      }

      if (canUseOnlineMode) {
        attemptSync();
      } else {
        clearHideTimer();
        setSyncState({
          phase: 'waiting',
          pendingCount: queue.length,
          syncedCount: 0,
          visible: true,
        });
      }
    });
  }, [attemptSync, canUseOnlineMode, clearHideTimer, enabled]);


  useEffect(() => {
    if (enabled && canUseOnlineMode && !prevCanSync.current) {
      attemptSync();
    }
    prevCanSync.current = canUseOnlineMode;
  }, [attemptSync, canUseOnlineMode, enabled]);



  useEffect(() => {
    if (enabled && canUseOnlineMode) {
      attemptSync();
    }
  }, [attemptSync, canUseOnlineMode, enabled]);


  useEffect(() => {
    const handleAppStateChange = (nextState: AppStateStatus) => {
      if (nextState === 'active' && enabled && canUseOnlineMode) {
        attemptSync();
      }
    };

    const sub = AppState.addEventListener('change', handleAppStateChange);
    return () => sub.remove();
  }, [attemptSync, canUseOnlineMode, enabled]);


  useEffect(() => {
    if (!enabled || !canUseOnlineMode) return;

    const interval = setInterval(() => {
      attemptSync();
    }, SYNC_POLL_INTERVAL_MS);

    return () => clearInterval(interval);
  }, [attemptSync, canUseOnlineMode, enabled]);

  useEffect(() => {
    return () => clearHideTimer();
  }, [clearHideTimer]);

  return syncState;
}
