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
  /** true when the image passes all quality checks */
  valid: boolean;
  /** Human-readable rejection reason (empty string when valid) */
  reason: string;
  /**
   * On web: the image drawn on a sized canvas, returned as a Blob URL.
   * On native: the original URI (quality checks are all we can do client-side).
   */
  processedUri: string;
}

// ── Thresholds (match backend quality.py) ──────────────────────

const BLUR_THRESHOLD = 15;       // Laplacian variance
const TOO_DARK_THRESHOLD = 15;   // Mean brightness
const TOO_BRIGHT_THRESHOLD = 245;
const LOW_CONTRAST_THRESHOLD = 8; // Std-dev of brightness
const BLACK_THRESHOLD = 5;        // Overall mean pixel value

// ── Target size for the preprocessed image (matches model input) ──
const TARGET_SIZE = 512;

// ── Helpers ────────────────────────────────────────────────────

/** Load an image from a URI in the browser and return an HTMLImageElement. */
function loadImage(uri: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('Could not load image for preprocessing.'));
    img.src = uri;
  });
}

/**
 * Draw the image onto a canvas at TARGET_SIZE × TARGET_SIZE and return the
 * raw RGBA pixel data along with the canvas reference.
 */
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

// ── Quality-check functions ────────────────────────────────────

/** Compute grayscale values (luminance) from RGBA pixel buffer. */
function toGrayscale(data: Uint8ClampedArray, pixelCount: number): Float64Array {
  const gray = new Float64Array(pixelCount);
  for (let i = 0; i < pixelCount; i++) {
    const offset = i * 4;
    // ITU-R BT.601 luma weights
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

/**
 * Approximate the Laplacian variance (blur metric) by computing a
 * simplified 3×3 Laplacian kernel convolution on the grayscale image.
 */
function laplacianVariance(
  gray: Float64Array,
  width: number,
  height: number,
): number {
  // Laplacian kernel:  [0, 1, 0]
  //                    [1,-4, 1]
  //                    [0, 1, 0]
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

/** Overall mean pixel intensity (across RGB, not just luma). */
function meanPixelValue(data: Uint8ClampedArray, pixelCount: number): number {
  let sum = 0;
  for (let i = 0; i < pixelCount; i++) {
    const o = i * 4;
    sum += (data[o] + data[o + 1] + data[o + 2]) / 3;
  }
  return sum / pixelCount;
}

// ── Main preprocessing entry point ────────────────────────────

/**
 * Run common image preprocessing on the given image URI.
 *
 * On **web**: performs quality checks using canvas APIs and returns a
 * resized image as a Blob URL ready for upload.
 *
 * On **native**: returns the original URI unchanged (quality checks
 * rely on canvas APIs unavailable on native; the backend still has
 * them as a fallback).
 */
export async function preprocessImage(imageUri: string): Promise<PreprocessResult> {
  // ── Native: skip client-side pixel checks (no canvas API) ──
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

    // 1. Near-black
    const overallMean = meanPixelValue(data, pixelCount);
    if (overallMean < BLACK_THRESHOLD) {
      return { valid: false, reason: 'Image is nearly black — please retake the photo.', processedUri: imageUri };
    }

    // 2. Blur
    const lapVar = laplacianVariance(gray, TARGET_SIZE, TARGET_SIZE);
    if (lapVar < BLUR_THRESHOLD) {
      return { valid: false, reason: 'Image is too blurry — try holding the camera steady.', processedUri: imageUri };
    }

    // 3. Too dark
    if (grayMean < TOO_DARK_THRESHOLD) {
      return { valid: false, reason: 'Image is too dark — try better lighting.', processedUri: imageUri };
    }

    // 4. Too bright
    if (grayMean > TOO_BRIGHT_THRESHOLD) {
      return { valid: false, reason: 'Image is too bright — reduce exposure or avoid direct sunlight.', processedUri: imageUri };
    }

    // 5. Low contrast
    if (grayStd < LOW_CONTRAST_THRESHOLD) {
      return { valid: false, reason: 'Image has very low contrast — ensure the leaf is clearly visible.', processedUri: imageUri };
    }

    // ── All checks passed — export the resized canvas as a Blob URL ──
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
    // If preprocessing itself fails, let the backend handle it
    console.warn('[preprocessing] Client-side quality check failed, falling through:', err?.message);
    return { valid: true, reason: '', processedUri: imageUri };
  }
}
