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

interface LeafColorCheckResult {
  isLeaf: boolean;
  leafRatio: number;
  greenRatio: number;
  yellowRatio: number;
  brownRatio: number;
  largestComponentRatio: number;
  largestComponentLeafFraction: number;
  edgeRatio: number;
  saturationStd: number;
  valueStd: number;
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

function rgbToOpenCvHsv(r: number, g: number, b: number) {
  const rf = r / 255;
  const gf = g / 255;
  const bf = b / 255;
  const max = Math.max(rf, gf, bf);
  const min = Math.min(rf, gf, bf);
  const delta = max - min;

  let hueDegrees = 0;
  if (delta !== 0) {
    if (max === rf) {
      hueDegrees = 60 * (((gf - bf) / delta) % 6);
    } else if (max === gf) {
      hueDegrees = 60 * ((bf - rf) / delta + 2);
    } else {
      hueDegrees = 60 * ((rf - gf) / delta + 4);
    }
  }

  if (hueDegrees < 0) hueDegrees += 360;

  return {
    h: hueDegrees / 2,
    s: max === 0 ? 0 : (delta / max) * 255,
    v: max * 255,
  };
}

function getLargestLeafComponent(mask: Uint8Array, width: number, height: number) {
  const totalPixels = width * height;
  const visited = new Uint8Array(totalPixels);
  const stack = new Int32Array(totalPixels);
  let largestArea = 0;
  let largestExtent = 0;

  for (let start = 0; start < totalPixels; start++) {
    if (!mask[start] || visited[start]) continue;

    let stackSize = 0;
    let area = 0;
    let minX = width;
    let minY = height;
    let maxX = 0;
    let maxY = 0;

    visited[start] = 1;
    stack[stackSize++] = start;

    while (stackSize > 0) {
      const current = stack[--stackSize];
      const x = current % width;
      const y = Math.floor(current / width);

      area++;
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);

      const neighbors = [current - 1, current + 1, current - width, current + width];
      for (const next of neighbors) {
        if (next < 0 || next >= totalPixels || visited[next] || !mask[next]) continue;
        const nextX = next % width;
        if ((next === current - 1 && nextX !== x - 1) || (next === current + 1 && nextX !== x + 1)) continue;
        visited[next] = 1;
        stack[stackSize++] = next;
      }
    }

    if (area > largestArea) {
      const bboxArea = Math.max((maxX - minX + 1) * (maxY - minY + 1), 1);
      largestArea = area;
      largestExtent = area / bboxArea;
    }
  }

  return { largestArea, largestExtent };
}

function checkLeafColors(data: Uint8ClampedArray, width: number, height: number): LeafColorCheckResult {
  const totalPixels = width * height;
  const mask = new Uint8Array(totalPixels);
  let greenPixels = 0;
  let yellowPixels = 0;
  let brownPixels = 0;
  let leafPixels = 0;
  let edgePixels = 0;
  let saturationSum = 0;
  let saturationSquareSum = 0;
  let valueSum = 0;
  let valueSquareSum = 0;
  const luminance = new Float32Array(totalPixels);

  for (let pixel = 0; pixel < totalPixels; pixel++) {
    const source = pixel * 4;
    const { h, s, v } = rgbToOpenCvHsv(data[source], data[source + 1], data[source + 2]);
    luminance[pixel] = data[source] * 0.299 + data[source + 1] * 0.587 + data[source + 2] * 0.114;

    const isGreen = h >= 35 && h <= 85 && s >= 40 && v >= 40;
    const isYellow = h >= 20 && h <= 35 && s >= 40 && v >= 40;
    const isBrown = h >= 10 && h <= 20 && s >= 50 && v >= 20 && v <= 200;

    if (isGreen) greenPixels++;
    if (isYellow) yellowPixels++;
    if (isBrown) brownPixels++;
    if (isGreen || isYellow || isBrown) {
      mask[pixel] = 1;
      leafPixels++;
      saturationSum += s;
      saturationSquareSum += s * s;
      valueSum += v;
      valueSquareSum += v * v;
    }
  }

  for (let y = 0; y < height - 1; y++) {
    for (let x = 0; x < width - 1; x++) {
      const pixel = y * width + x;
      if (!mask[pixel]) continue;

      const right = pixel + 1;
      const down = pixel + width;
      const rightDiff = mask[right] ? Math.abs(luminance[pixel] - luminance[right]) : 0;
      const downDiff = mask[down] ? Math.abs(luminance[pixel] - luminance[down]) : 0;
      if (rightDiff > 22 || downDiff > 22) edgePixels++;
    }
  }

  const { largestArea, largestExtent } = getLargestLeafComponent(mask, width, height);
  const greenRatio = greenPixels / totalPixels;
  const yellowRatio = yellowPixels / totalPixels;
  const brownRatio = brownPixels / totalPixels;
  const leafRatio = leafPixels / totalPixels;
  const largestComponentRatio = largestArea / totalPixels;
  const largestComponentLeafFraction = largestArea / Math.max(leafPixels, 1);
  const edgeRatio = edgePixels / Math.max(leafPixels, 1);
  const saturationMean = saturationSum / Math.max(leafPixels, 1);
  const valueMean = valueSum / Math.max(leafPixels, 1);
  const saturationStd = Math.sqrt(Math.max(saturationSquareSum / Math.max(leafPixels, 1) - saturationMean ** 2, 0));
  const valueStd = Math.sqrt(Math.max(valueSquareSum / Math.max(leafPixels, 1) - valueMean ** 2, 0));

  const hasGreenLeafArea = greenRatio > 0.04;
  const hasStressedLeafArea = greenRatio > 0.015 && (yellowRatio + brownRatio > 0.06);
  const hasEnoughLeafPixels = leafRatio > 0.08;
  const hasContiguousLeafRegion = largestComponentRatio > 0.035 && largestComponentLeafFraction > 0.35;
  const hasNaturalVariation = edgeRatio > 0.006 || saturationStd > 18 || valueStd > 18;
  const isFlatSurface = leafRatio > 0.55 && largestComponentLeafFraction > 0.75 && !hasNaturalVariation;
  const isUniformArtificialShape = largestExtent > 0.9 && !hasNaturalVariation;

  return {
    isLeaf:
      (hasGreenLeafArea || hasStressedLeafArea) &&
      hasEnoughLeafPixels &&
      hasContiguousLeafRegion &&
      !isFlatSurface &&
      !isUniformArtificialShape,
    leafRatio,
    greenRatio,
    yellowRatio,
    brownRatio,
    largestComponentRatio,
    largestComponentLeafFraction,
    edgeRatio,
    saturationStd,
    valueStd,
  };
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
  const leafCheck = checkLeafColors(data, IMAGE_SIZE, IMAGE_SIZE);
  console.log(
    `[LeafCheck] green=${leafCheck.greenRatio.toFixed(2)}, yellow=${leafCheck.yellowRatio.toFixed(2)}, ` +
      `brown=${leafCheck.brownRatio.toFixed(2)}, total=${leafCheck.leafRatio.toFixed(2)}, ` +
      `largest=${leafCheck.largestComponentRatio.toFixed(2)}, edge=${leafCheck.edgeRatio.toFixed(3)}, ` +
      `s_std=${leafCheck.saturationStd.toFixed(1)}, v_std=${leafCheck.valueStd.toFixed(1)}`
  );

  if (!leafCheck.isLeaf) {
    throw new Error(`Rejected: Not a leaf (ratio=${leafCheck.leafRatio.toFixed(2)})`);
  }

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
