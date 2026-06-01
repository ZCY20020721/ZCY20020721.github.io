/**
 * Send cropped bird image to CLIP backend and get best-matching cards.
 * @param {String} imageDataUrl - data:image/jpeg;base64,... of the cropped object
 * @returns {Promise<{matches: Array<{category, text, score}>}>}
 */

const CLIP_API = "http://localhost:5007/match";

export const matchCards = async (imageDataUrl) => {
  const base64 = imageDataUrl.split(",")[1];  // strip data:image/...;base64, prefix
  const resp = await fetch(CLIP_API, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: base64 }),
  });
  if (!resp.ok) {
    throw new Error(`CLIP API error: ${resp.status}`);
  }
  return resp.json();
};

/**
 * Crop detected object from original image by bounding box
 * @param {HTMLImageElement} image - original uploaded image
 * @param {Number[]} bbox - [x, y, w, h] in display coordinates
 * @returns {String} dataURL of cropped region (max 224x224 for CLIP)
 */
export const cropObject = (image, bbox) => {
  const [x, y, w, h] = bbox;
  const canvas = document.createElement("canvas");

  // CLIP expects 224x224, keep aspect ratio with padding
  const size = 224;
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");

  // Scale from display dimensions to natural image dimensions
  const sx = image.naturalWidth / image.width;
  const sy = image.naturalHeight / image.height;

  const srcW = w * sx;
  const srcH = h * sy;
  const srcX = x * sx;
  const srcY = y * sy;

  // Fit into 224x224 with black padding
  const scale = Math.min(size / srcW, size / srcH, 2.0);
  const dw = srcW * scale;
  const dh = srcH * scale;
  const dx = (size - dw) / 2;
  const dy = (size - dh) / 2;

  ctx.fillStyle = "#000000";
  ctx.fillRect(0, 0, size, size);
  ctx.drawImage(image, srcX, srcY, srcW, srcH, dx, dy, dw, dh);

  return canvas.toDataURL("image/jpeg", 0.85);
};
