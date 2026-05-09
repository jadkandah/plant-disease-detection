import { Platform } from 'react-native';
import { Asset } from 'expo-asset';
import { buildDiseaseInfo } from './diseaseMetadata';

const MODEL_ASSET = require('../../../assets/mobile_models/offline_model.onnx');
const MODEL_MANIFEST = require('../../../assets/mobile_models/offline_model_manifest.json');

const IMAGE_SIZE = MODEL_MANIFEST.image_size || 512;
const INPUT_NAME = MODEL_MANIFEST.input_name || 'image';
const OUTPUT_NAME = MODEL_MANIFEST.output_name || 'logits';
const NORM_MEAN = MODEL_MANIFEST.preprocessing?.normalize_mean || [0.485, 0.456, 0.406];
const NORM_STD = MODEL_MANIFEST.preprocessing?.normalize_std || [0.229, 0.224, 0.225];
const INDEX_TO_CLASS: Record<string, string> = MODEL_MANIFEST.index_to_class || {};

type InferenceSession = any;
type OrtRuntime = {
  env: { wasm: { numThreads?: number; wasmPaths?: string | Record<string, string> } };
  Tensor: new (type: 'float32', data: Float32Array, dims: number[]) => any;
  InferenceSession: {
    create: (modelUrl: string, options?: Record<string, unknown>) => Promise<InferenceSession>;
  };
};

declare global {
  interface Window {
    ort?: OrtRuntime;
  }
}

let ortPromise: Promise<OrtRuntime> | null = null;
let sessionPromise: Promise<InferenceSession> | null = null;

export interface LocalPredictionResult {
  success: true;
  mode: 'offline';
  prediction_key: string;
  confidence: number;
  is_healthy: boolean;
  disease_info: ReturnType<typeof buildDiseaseInfo>;
}

function loadScript(src: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const existing = document.querySelector(`script[src="${src}"]`);
    if (existing) {
      resolve();
      return;
    }

    const script = document.createElement('script');
    script.src = src;
    script.async = true;
    script.onload = () => resolve();
    script.onerror = () => reject(new Error(`Could not load ${src}`));
    document.head.appendChild(script);
  });
}

async function getOrt(): Promise<OrtRuntime> {
  if (Platform.OS !== 'web' || typeof window === 'undefined') {
    throw new Error('ONNX Runtime Web is available only in the browser demo.');
  }

  if (window.ort) {
    window.ort.env.wasm.numThreads = 1;
    window.ort.env.wasm.wasmPaths = '/ort/';
    return window.ort;
  }

  if (!ortPromise) {
    ortPromise = (async () => {
      await loadScript('/ort/ort.wasm.min.js');
      if (!window.ort) {
        throw new Error('ONNX Runtime Web did not initialize.');
      }
      window.ort.env.wasm.numThreads = 1;
      window.ort.env.wasm.wasmPaths = '/ort/';
      return window.ort;
    })();
  }

  return ortPromise;
}

async function getSession() {
  if (Platform.OS !== 'web') {
    throw new Error('Local offline inference is configured for the web demo only.');
  }

  if (!sessionPromise) {
    sessionPromise = (async () => {
      const ort = await getOrt();
      const asset = Asset.fromModule(MODEL_ASSET);
      await asset.downloadAsync();
      const modelUrl = asset.localUri || asset.uri;
      if (!modelUrl) throw new Error('Offline model asset could not be resolved.');
      return ort.InferenceSession.create(modelUrl, { executionProviders: ['wasm'] });
    })();
  }

  return sessionPromise;
}

function loadBrowserImage(imageUri: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.crossOrigin = 'anonymous';
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error('Could not load image for offline inference.'));
    image.src = imageUri;
  });
}

async function imageUriToTensor(imageUri: string) {
  if (typeof document === 'undefined') {
    throw new Error('Browser image APIs are unavailable.');
  }

  const image = await loadBrowserImage(imageUri);
  const canvas = document.createElement('canvas');
  canvas.width = IMAGE_SIZE;
  canvas.height = IMAGE_SIZE;
  const context = canvas.getContext('2d');
  if (!context) {
    throw new Error('Could not prepare browser canvas for offline inference.');
  }

  context.drawImage(image, 0, 0, IMAGE_SIZE, IMAGE_SIZE);
  const { data } = context.getImageData(0, 0, IMAGE_SIZE, IMAGE_SIZE);
  const imageArea = IMAGE_SIZE * IMAGE_SIZE;

  const tensor = new Float32Array(3 * imageArea);

  for (let pixel = 0; pixel < imageArea; pixel++) {
    const source = pixel * 4;
    const r = data[source] / 255;
    const g = data[source + 1] / 255;
    const b = data[source + 2] / 255;

    tensor[pixel] = (r - NORM_MEAN[0]) / NORM_STD[0];
    tensor[imageArea + pixel] = (g - NORM_MEAN[1]) / NORM_STD[1];
    tensor[2 * imageArea + pixel] = (b - NORM_MEAN[2]) / NORM_STD[2];
  }

  return tensor;
}

function softmax(logits: Float32Array | number[]) {
  let max = -Infinity;
  for (const value of logits) max = Math.max(max, value);

  const scores = new Float32Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    const score = Math.exp(logits[i] - max);
    scores[i] = score;
    sum += score;
  }

  let bestIdx = 0;
  let bestScore = 0;
  for (let i = 0; i < scores.length; i++) {
    const probability = scores[i] / sum;
    if (probability > bestScore) {
      bestScore = probability;
      bestIdx = i;
    }
  }

  return { bestIdx, confidence: Number(bestScore.toFixed(4)) };
}

export async function predictOffline(imageUri: string): Promise<LocalPredictionResult> {
  const session = await getSession();
  const ort = await getOrt();
  const inputTensor = await imageUriToTensor(imageUri);
  const feeds = {
    [INPUT_NAME]: new ort.Tensor('float32', inputTensor, [1, 3, IMAGE_SIZE, IMAGE_SIZE]),
  };

  const output = await session.run(feeds, [OUTPUT_NAME]);
  const logits = output[OUTPUT_NAME].data as Float32Array;
  const { bestIdx, confidence } = softmax(logits);
  const predictionKey = INDEX_TO_CLASS[String(bestIdx)];
  if (!predictionKey) {
    throw new Error(`Offline model returned unknown class index ${bestIdx}.`);
  }

  const diseaseInfo = buildDiseaseInfo(predictionKey);
  return {
    success: true,
    mode: 'offline',
    prediction_key: predictionKey,
    confidence,
    is_healthy: diseaseInfo.health_status === 'healthy',
    disease_info: diseaseInfo,
  };
}
