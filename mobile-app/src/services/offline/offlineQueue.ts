import AsyncStorage from '@react-native-async-storage/async-storage';
import apiClient from '../auth/apiClient';

const OFFLINE_QUEUE_KEY = 'offline_prediction_queue';

export interface OfflinePrediction {
  id: string; // UUID generated client-side
  imageUri: string;
  sourceType: 'camera' | 'gallery';
  timestamp: string;
  // After local mock result (optional)
  mockResult?: {
    crop_name: string;
    disease_name: string;
    confidence: number;
    is_healthy: boolean;
  };
}

/**
 * Add a prediction request to the offline queue.
 */
export async function enqueueOfflinePrediction(prediction: OfflinePrediction): Promise<void> {
  const queue = await getOfflineQueue();
  queue.push(prediction);
  await AsyncStorage.setItem(OFFLINE_QUEUE_KEY, JSON.stringify(queue));
}

/**
 * Get all pending offline predictions.
 */
export async function getOfflineQueue(): Promise<OfflinePrediction[]> {
  try {
    const raw = await AsyncStorage.getItem(OFFLINE_QUEUE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

/**
 * Clear the entire offline queue (after successful sync).
 */
export async function clearOfflineQueue(): Promise<void> {
  await AsyncStorage.removeItem(OFFLINE_QUEUE_KEY);
}

/**
 * Remove a single item from the queue by its id.
 */
export async function removeFromQueue(id: string): Promise<void> {
  const queue = await getOfflineQueue();
  const filtered = queue.filter((item) => item.id !== id);
  await AsyncStorage.setItem(OFFLINE_QUEUE_KEY, JSON.stringify(filtered));
}

/**
 * Attempt to sync all queued offline predictions to the server.
 * Returns the number of successfully synced items.
 */
export async function syncOfflineQueue(): Promise<number> {
  const queue = await getOfflineQueue();
  if (queue.length === 0) return 0;

  let syncedCount = 0;

  for (const item of queue) {
    try {
      const formData = new FormData();
      const filename = item.imageUri.split('/').pop() || 'offline.jpg';
      const match = /\.(\w+)$/.exec(filename);
      const fileType = match ? `image/${match[1]}` : 'image/jpeg';

      formData.append('image', { uri: item.imageUri, name: filename, type: fileType } as any);
      formData.append('source_type', item.sourceType);
      formData.append('mode', 'offline');

      await apiClient.post('/predict/', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      // Remove successfully synced item
      await removeFromQueue(item.id);
      syncedCount++;
    } catch (error) {
      // If any item fails, stop syncing (server may be down)
      console.log(`Failed to sync prediction ${item.id}:`, error);
      break;
    }
  }

  return syncedCount;
}
