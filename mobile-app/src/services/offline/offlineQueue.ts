import AsyncStorage from '@react-native-async-storage/async-storage';
import apiClient from '../auth/apiClient';

const OFFLINE_QUEUE_KEY = 'offline_prediction_queue';

export interface OfflinePredictionResult {
  id: string;
  predictionKey: string;
  sourceType: 'camera' | 'gallery';
  predictedAt: string;
  cropName: string;
  diseaseNameEn: string;
  diseaseNameAr: string;
  confidence: number;
  isHealthy: boolean;
}

/**
 * Add a locally inferred result to the offline queue.
 */
export async function enqueueOfflineResult(prediction: OfflinePredictionResult): Promise<void> {
  const queue = await getOfflineQueue();
  queue.push(prediction);
  await AsyncStorage.setItem(OFFLINE_QUEUE_KEY, JSON.stringify(queue));
}

/**
 * Get all pending offline prediction results.
 */
export async function getOfflineQueue(): Promise<OfflinePredictionResult[]> {
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
 * Attempt to sync all queued local results to the server.
 * Returns the number of successfully synced items.
 */
export async function syncOfflineQueue(): Promise<number> {
  const queue = await getOfflineQueue();
  if (queue.length === 0) return 0;

  const records = queue.map((item) => ({
    crop_name: item.cropName,
    disease_name_en: item.diseaseNameEn,
    disease_name_ar: item.diseaseNameAr,
    confidence: item.confidence,
    is_healthy: item.isHealthy,
    source_type: item.sourceType,
    sync_status: 'synced',
  }));

  try {
    await apiClient.post('/history/sync/', { records });
    await clearOfflineQueue();
    return queue.length;
  } catch (error) {
    console.log('Failed to sync offline results:', error);
    return 0;
  }
}
