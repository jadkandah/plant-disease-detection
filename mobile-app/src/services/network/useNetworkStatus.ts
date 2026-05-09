import { useState, useEffect } from 'react';
import { AppState, AppStateStatus, Platform } from 'react-native';
import NetInfo, { NetInfoState } from '@react-native-community/netinfo';
import { BASE_URL } from '../auth/apiClient';

const INTERNET_CHECK_URL = 'https://clients3.google.com/generate_204';
const BACKEND_HEALTH_URL = `${BASE_URL}/health/`;
const CHECK_TIMEOUT_MS = 2500;
const POLL_INTERVAL_MS = 5000;

const isBrowserOnline = () => {
  if (Platform.OS !== 'web') return true;
  if (typeof navigator === 'undefined') return true;
  return navigator.onLine;
};

const hasUsableInternet = (state: NetInfoState) => {
  return state.isConnected === true && state.isInternetReachable !== false && isBrowserOnline();
};

const probeUrl = async (url: string, init: RequestInit = {}) => {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), CHECK_TIMEOUT_MS);

  try {
    await fetch(url, {
      method: 'GET',
      cache: 'no-store',
      ...init,
      signal: controller.signal,
    });
    return true;
  } catch {
    return false;
  } finally {
    clearTimeout(timeout);
  }
};

const probeConnectivity = async () => {
  const netInfoState = await NetInfo.fetch();
  const netInfoOnline = hasUsableInternet(netInfoState);

  if (!netInfoOnline) {
    return false;
  }

  const [internetReachable, backendReachable] = await Promise.all([
    probeUrl(INTERNET_CHECK_URL, Platform.OS === 'web' ? { mode: 'no-cors' } : {}),
    probeUrl(BACKEND_HEALTH_URL),
  ]);

  return internetReachable && backendReachable;
};

/**
 * Custom hook that monitors network connectivity.
 * Returns usable internet connectivity, not only Wi-Fi/interface status.
 */
export function useNetworkStatus() {
  const [isConnected, setIsConnected] = useState<boolean>(isBrowserOnline());
  const [isInternetReachable, setIsInternetReachable] = useState<boolean | null>(isBrowserOnline());

  useEffect(() => {
    const updateFromState = (state: NetInfoState) => {
      const usableInternet = hasUsableInternet(state);
      setIsConnected(usableInternet);
      setIsInternetReachable(usableInternet);
    };

    const refresh = async () => {
      try {
        const usableConnection = await probeConnectivity();
        setIsConnected(usableConnection);
        setIsInternetReachable(usableConnection);
      } catch {
        setIsConnected(false);
        setIsInternetReachable(false);
      }
    };

    const unsubscribe = NetInfo.addEventListener((state: NetInfoState) => {
      if (hasUsableInternet(state)) {
        refresh();
      } else {
        updateFromState(state);
      }
    });

    refresh();
    const pollInterval = setInterval(refresh, POLL_INTERVAL_MS);

    const appStateSub = AppState.addEventListener('change', (nextState: AppStateStatus) => {
      if (nextState === 'active') {
        refresh();
      }
    });

    const handleBrowserConnectivityChange = () => {
      if (Platform.OS === 'web' && !isBrowserOnline()) {
        setIsConnected(false);
        setIsInternetReachable(false);
      }
      refresh();
    };

    if (Platform.OS === 'web' && typeof window !== 'undefined') {
      window.addEventListener('online', handleBrowserConnectivityChange);
      window.addEventListener('offline', handleBrowserConnectivityChange);
    }

    return () => {
      clearInterval(pollInterval);
      unsubscribe();
      appStateSub.remove();
      if (Platform.OS === 'web' && typeof window !== 'undefined') {
        window.removeEventListener('online', handleBrowserConnectivityChange);
        window.removeEventListener('offline', handleBrowserConnectivityChange);
      }
    };
  }, []);

  return { isConnected, isInternetReachable };
}
