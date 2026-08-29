/**
 * Centralized API Configuration & Network Layer
 * All frontend requests route through this module to ensure configurable backend URLs,
 * timeout controls, and unified error parsing.
 */

const rawApiUrl = import.meta.env.VITE_API_URL;

if (!rawApiUrl && import.meta.env.DEV) {
  console.warn('[API Config] VITE_API_URL is not configured. Using local / relative API routes.');
}

export const API_BASE_URL = (rawApiUrl || '').replace(/\/+$/, '');

// 120 seconds (120,000 ms) timeout to accommodate Render Free cold-start & sequential inference
const DEFAULT_TIMEOUT_MS = 120000;
const HEALTH_TIMEOUT_MS = 30000;

export async function fetchWithTimeout(url, options = {}, timeoutMs = DEFAULT_TIMEOUT_MS) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const fullUrl = url.startsWith('http') ? url : `${API_BASE_URL}${url}`;
    const response = await fetch(fullUrl, {
      ...options,
      signal: controller.signal,
    });
    return response;
  } catch (error) {
    if (error.name === 'AbortError') {
      throw new Error(`Request timed out after ${timeoutMs / 1000}s. The server may be busy.`);
    }
    throw new Error('Unable to connect to the backend API. Please ensure the server is running.');
  } finally {
    clearTimeout(timeoutId);
  }
}

export async function checkBackendHealth() {
  const res = await fetchWithTimeout('/api/health', {}, HEALTH_TIMEOUT_MS);
  if (!res.ok) throw new Error('Backend health check failed');
  return res.json();
}

export async function checkBackendReady() {
  const res = await fetchWithTimeout('/api/ready', {}, HEALTH_TIMEOUT_MS);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail?.message || 'Inference engine not ready yet.');
  }
  return res.json();
}

export async function fetchSamplesList() {
  const res = await fetchWithTimeout('/api/samples', {}, HEALTH_TIMEOUT_MS);
  if (!res.ok) throw new Error('Failed to load sample audio files.');
  return res.json();
}

export function getSampleAudioStreamUrl(sampleId) {
  return `${API_BASE_URL}/api/sample-audio/${encodeURIComponent(sampleId)}`;
}

export async function submitPrediction(audioFile, sampleId) {
  const formData = new FormData();
  if (audioFile) {
    formData.append('file', audioFile);
  } else if (sampleId) {
    formData.append('sample_id', sampleId);
  } else {
    throw new Error('No audio file or sample provided for prediction.');
  }

  // Note: Do NOT manually set Content-Type header when body is FormData;
  // the browser automatically sets the multipart/form-data boundary.
  const res = await fetchWithTimeout('/api/predict', {
    method: 'POST',
    body: formData,
  }, DEFAULT_TIMEOUT_MS);

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    let errorMessage = errorData.detail;

    if (typeof errorMessage === 'object' && errorMessage !== null) {
      errorMessage = errorMessage.message || JSON.stringify(errorMessage);
    }

    if (!errorMessage) {
      if (res.status === 413) errorMessage = 'Audio file exceeds the maximum 25 MB limit.';
      else if (res.status === 429) errorMessage = 'Rate limit reached (15 requests/min). Please wait a moment.';
      else if (res.status === 422) errorMessage = 'The audio file is corrupt or unreadable.';
      else if (res.status === 503) errorMessage = 'Inference models are currently initializing. Please retry in a few seconds.';
      else errorMessage = `Prediction failed (HTTP ${res.status})`;
    }

    throw new Error(errorMessage);
  }

  return res.json();
}
