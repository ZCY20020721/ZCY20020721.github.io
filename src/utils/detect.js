import cv from "@techstark/opencv-js";
import { Tensor } from "onnxruntime-web";
import { renderBoxes, Colors } from "./renderBox";
import labels from "./labels.json";

const colors = new Colors();
const numClass = labels.length;

/**
 * Detect Image
 */
export const detectImage = async (
  image,
  canvas,
  session,
  topk,
  iouThreshold,
  scoreThreshold,
  inputShape
) => {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

  const [modelWidth, modelHeight] = inputShape.slice(2);
  const maxSize = Math.max(modelWidth, modelHeight);
  const [input, xRatio, yRatio] = preprocessing(image, modelWidth, modelHeight);

  const tensor = new Tensor("float32", input.data32F, inputShape);
  const config = new Tensor(
    "float32",
    new Float32Array([numClass, topk, iouThreshold, scoreThreshold])
  );
  const { output0, output1 } = await session.net.run({ images: tensor });
  const { selected } = await session.nms.run({ detection: output0, config: config });

  const boxes = [];
  let overlay = new Tensor("uint8", new Uint8Array(modelHeight * modelWidth * 4), [
    modelHeight, modelWidth, 4,
  ]);

  for (let idx = 0; idx < selected.dims[1]; idx++) {
    const data = selected.data.slice(idx * selected.dims[2], (idx + 1) * selected.dims[2]);
    let box = data.slice(0, 4);
    const scores = data.slice(4, 4 + numClass);
    const score = Math.max(...scores);
    const label = scores.indexOf(score);
    const color = colors.get(label);

    box = overflowBoxes(
      [box[0] - 0.5 * box[2], box[1] - 0.5 * box[3], box[2], box[3]],
      maxSize
    );

    const [x, y, w, h] = overflowBoxes(
      [
        Math.floor(box[0] * xRatio),
        Math.floor(box[1] * yRatio),
        Math.floor(box[2] * xRatio),
        Math.floor(box[3] * yRatio),
      ],
      maxSize
    );

    boxes.push({
      label: labels[label],
      probability: score,
      color: color,
      bounding: [x, y, w, h],
    });

    const mask = new Tensor(
      "float32",
      new Float32Array([...box, ...data.slice(4 + numClass)])
    );
    const maskConfig = new Tensor(
      "float32",
      new Float32Array([maxSize, x, y, w, h, ...Colors.hexToRgba(color, 120)])
    );
    const { mask_filter } = await session.mask.run({
      detection: mask,
      mask: output1,
      config: maskConfig,
      overlay: overlay,
    });
    overlay = mask_filter;
  }

  const mask_img = new ImageData(new Uint8ClampedArray(overlay.data), modelHeight, modelWidth);
  ctx.putImageData(mask_img, 0, 0);

  renderBoxes(ctx, boxes);
  input.delete();
};

/**
 * Preprocessing image
 */
const preprocessing = (source, modelWidth, modelHeight, stride = 32) => {
  const mat = cv.imread(source);
  const matC3 = new cv.Mat(mat.rows, mat.cols, cv.CV_8UC3);
  cv.cvtColor(mat, matC3, cv.COLOR_RGBA2BGR);

  const [w, h] = divStride(stride, matC3.cols, matC3.rows);
  cv.resize(matC3, matC3, new cv.Size(w, h));

  const maxSize = Math.max(matC3.rows, matC3.cols);
  const xPad = maxSize - matC3.cols, xRatio = maxSize / matC3.cols;
  const yPad = maxSize - matC3.rows, yRatio = maxSize / matC3.rows;
  const matPad = new cv.Mat();
  cv.copyMakeBorder(matC3, matPad, 0, yPad, 0, xPad, cv.BORDER_CONSTANT);

  const input = cv.blobFromImage(
    matPad, 1 / 255.0,
    new cv.Size(modelWidth, modelHeight),
    new cv.Scalar(0, 0, 0), true, false
  );

  mat.delete(); matC3.delete(); matPad.delete();

  return [input, xRatio, yRatio];
};

const divStride = (stride, width, height) => {
  if (width % stride !== 0) {
    if (width % stride >= stride / 2) width = (Math.floor(width / stride) + 1) * stride;
    else width = Math.floor(width / stride) * stride;
  }
  if (height % stride !== 0) {
    if (height % stride >= stride / 2) height = (Math.floor(height / stride) + 1) * stride;
    else height = Math.floor(height / stride) * stride;
  }
  return [width, height];
};

const overflowBoxes = (box, maxSize) => {
  box[0] = box[0] >= 0 ? box[0] : 0;
  box[1] = box[1] >= 0 ? box[1] : 0;
  box[2] = box[0] + box[2] <= maxSize ? box[2] : maxSize - box[0];
  box[3] = box[1] + box[3] <= maxSize ? box[3] : maxSize - box[1];
  return box;
};
