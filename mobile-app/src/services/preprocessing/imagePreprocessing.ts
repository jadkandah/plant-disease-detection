/**
 * Frontend Image Preprocessing Module
 *
 * Performs common image quality checks on the client side before
 * sending to the backend.  This keeps the backend pipeline focused
 * on heavy-weight tasks (SAM background removal, etc.).
 *
 * Quality checks (mirror the backend quality.py logic):
 *   - Corrupt / unreadable image
 *   - Near-black image
 *   - Blurriness (Laplacian variance approximation)
 *   - Too dark / too bright
 *   - Low contrast
 */

import { Platform } from 'react-native';

export interface PreprocessResult {

  valid: boolean;

  reason: string;




  processedUri: string;
}



const BLUR_THRESHOLD = 15;       // Laplacian variance
const TOO_DARK_THRESHOLD = 15;   // Mean brightness
const TOO_BRIGHT_THRESHOLD = 245;
const LOW_CONTRAST_THRESHOLD = 8; // Std-dev of brightness
const BLACK_THRESHOLD = 5;        // Overall mean pixel value


const TARGET_SIZE = 512;




function loadImage(uri: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('Could not load image for preprocessing.'));
    img.src = uri;
  });
}





function drawToCanvas(
  img: HTMLImageElement,
  size: number,
): { canvas: HTMLCanvasElement; data: Uint8ClampedArray } {
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('Canvas 2D context unavailable.');
  ctx.drawImage(img, 0, 0, size, size);
  const imageData = ctx.getImageData(0, 0, size, size);
  return { canvas, data: imageData.data };
}




function toGrayscale(data: Uint8ClampedArray, pixelCount: number): Float64Array {
  const gray = new Float64Array(pixelCount);
  for (let i = 0; i < pixelCount; i++) {
    const offset = i * 4;

    gray[i] = 0.299 * data[offset] + 0.587 * data[offset + 1] + 0.114 * data[offset + 2];
  }
  return gray;
}

function mean(arr: Float64Array): number {
  let sum = 0;
  for (let i = 0; i < arr.length; i++) sum += arr[i];
  return sum / arr.length;
}

function stdDev(arr: Float64Array, avg: number): number {
  let sum = 0;
  for (let i = 0; i < arr.length; i++) {
    const d = arr[i] - avg;
    sum += d * d;
  }
  return Math.sqrt(sum / arr.length);
}





function laplacianVariance(
  gray: Float64Array,
  width: number,
  height: number,
): number {



  let sum = 0;
  let sumSq = 0;
  let count = 0;

  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const idx = y * width + x;
      const lap =
        gray[idx - width] +        // top
        gray[idx - 1] +             // left
        -4 * gray[idx] +            // center
        gray[idx + 1] +             // right
        gray[idx + width];           // bottom
      sum += lap;
      sumSq += lap * lap;
      count++;
    }
  }

  const avg = sum / count;
  return sumSq / count - avg * avg; // variance
}


function meanPixelValue(data: Uint8ClampedArray, pixelCount: number): number {
  let sum = 0;
  for (let i = 0; i < pixelCount; i++) {
    const o = i * 4;
    sum += (data[o] + data[o + 1] + data[o + 2]) / 3;
  }
  return sum / pixelCount;
}













export async function preprocessImage(imageUri: string): Promise<PreprocessResult> {

  if (Platform.OS !== 'web' || typeof document === 'undefined') {
    return { valid: true, reason: '', processedUri: imageUri };
  }

  try {
    const img = await loadImage(imageUri);
    const { canvas, data } = drawToCanvas(img, TARGET_SIZE);
    const pixelCount = TARGET_SIZE * TARGET_SIZE;
    const gray = toGrayscale(data, pixelCount);
    const grayMean = mean(gray);
    const grayStd = stdDev(gray, grayMean);


    const overallMean = meanPixelValue(data, pixelCount);
    if (overallMean < BLACK_THRESHOLD) {
      return { valid: false, reason: 'Image is nearly black — please retake the photo.', processedUri: imageUri };
    }


    const lapVar = laplacianVariance(gray, TARGET_SIZE, TARGET_SIZE);
    if (lapVar < BLUR_THRESHOLD) {
      return { valid: false, reason: 'Image is too blurry — try holding the camera steady.', processedUri: imageUri };
    }


    if (grayMean < TOO_DARK_THRESHOLD) {
      return { valid: false, reason: 'Image is too dark — try better lighting.', processedUri: imageUri };
    }


    if (grayMean > TOO_BRIGHT_THRESHOLD) {
      return { valid: false, reason: 'Image is too bright — reduce exposure or avoid direct sunlight.', processedUri: imageUri };
    }


    if (grayStd < LOW_CONTRAST_THRESHOLD) {
      return { valid: false, reason: 'Image has very low contrast — ensure the leaf is clearly visible.', processedUri: imageUri };
    }


    const blob = await new Promise<Blob>((resolve, reject) => {
      canvas.toBlob(
        (b) => (b ? resolve(b) : reject(new Error('Canvas export failed.'))),
        'image/jpeg',
        0.85,
      );
    });

    const processedUri = URL.createObjectURL(blob);
    return { valid: true, reason: '', processedUri };
  } catch (err: any) {

    console.warn('[preprocessing] Client-side quality check failed, falling through:', err?.message);
    return { valid: true, reason: '', processedUri: imageUri };
  }
}
