import AsyncStorage from '@react-native-async-storage/async-storage';
import apiClient from '../auth/apiClient';

const OFFLINE_QUEUE_KEY = 'offline_prediction_queue';

type OfflineQueueListener = (queue: OfflinePredictionResult[]) => void;

const queueListeners = new Set<OfflineQueueListener>();

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

export interface OfflineSyncResult {
  attempted: number;
  synced: number;
  remaining: number;
}

async function notifyQueueListeners() {
  const queue = await getOfflineQueue();
  queueListeners.forEach((listener) => listener(queue));
}

export function subscribeOfflineQueue(listener: OfflineQueueListener): () => void {
  queueListeners.add(listener);
  getOfflineQueue()
    .then(listener)
    .catch(() => listener([]));

  return () => {
    queueListeners.delete(listener);
  };
}




export async function enqueueOfflineResult(prediction: OfflinePredictionResult): Promise<void> {
  const queue = await getOfflineQueue();
  queue.push(prediction);
  await AsyncStorage.setItem(OFFLINE_QUEUE_KEY, JSON.stringify(queue));
  await notifyQueueListeners();
}




export async function getOfflineQueue(): Promise<OfflinePredictionResult[]> {
  try {
    const raw = await AsyncStorage.getItem(OFFLINE_QUEUE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}




export async function clearOfflineQueue(): Promise<void> {
  await AsyncStorage.removeItem(OFFLINE_QUEUE_KEY);
  await notifyQueueListeners();
}




export async function removeFromQueue(id: string): Promise<void> {
  const queue = await getOfflineQueue();
  const filtered = queue.filter((item) => item.id !== id);
  await AsyncStorage.setItem(OFFLINE_QUEUE_KEY, JSON.stringify(filtered));
  await notifyQueueListeners();
}





export async function syncOfflineQueue(): Promise<OfflineSyncResult> {
  const queue = await getOfflineQueue();
  if (queue.length === 0) {
    return { attempted: 0, synced: 0, remaining: 0 };
  }

  let synced = 0;

  for (const item of queue) {
    const record = {
      crop_name: item.cropName,
      disease_name_en: item.diseaseNameEn,
      disease_name_ar: item.diseaseNameAr,
      confidence: item.confidence,
      is_healthy: item.isHealthy,
      source_type: item.sourceType,
      sync_status: 'synced',
      model_mode: 'offline',
      predicted_at: item.predictedAt,
    };

    try {
      const response = await apiClient.post('/history/sync/', { records: [record] });
      const syncedRecords = response.data?.synced_records || [];
      if (syncedRecords.length > 0) {
        await removeFromQueue(item.id);
        synced += 1;
      }
    } catch (error) {
      console.log('Failed to sync offline result:', (error as any)?.response?.data || (error as any)?.message || error);
    }
  }

  const remaining = await getOfflineQueue();
  return { attempted: queue.length, synced, remaining: remaining.length };
}
