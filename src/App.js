import React, { useState, useRef } from "react";
import cv from "@techstark/opencv-js";
import { Tensor, InferenceSession } from "onnxruntime-web";
import Loader from "./components/loader";
import { detectImage } from "./utils/detect";
import { matchCards, cropObject } from "./utils/clip";
import "./style/App.css";

/** Bottom panel showing CLIP-matched cards */
const CardPanel = ({ boxes, imageRef }) => {
  const [matches, setMatches] = useState(null);
  const [bestLabel, setBestLabel] = useState("");
  const [loading, setLoading] = useState(false);
  const lastImg = useRef(null);

  if (imageRef.current && imageRef.current !== lastImg.current && boxes?.length) {
    lastImg.current = imageRef.current;
    const best = boxes.reduce((a, b) => a.probability > b.probability ? a : b);
    setBestLabel(best.label);
    setLoading(true);

    try {
      const crop = cropObject(imageRef.current, best.bounding);
      matchCards(crop).then((res) => {
        setMatches(res.matches);
        setLoading(false);
      }).catch(() => {
        setMatches(null);
        setLoading(false);
      });
    } catch {
      setMatches(null);
      setLoading(false);
    }
  }

  if (!boxes?.length) return null;

  return (
    <div className="card-panel">
      <h3>
        {/*emoji*/}ﾟ Best matches for {bestLabel}
        {loading && <span className="card-loading">...</span>}
      </h3>
      {matches ? (
        matches.map((m, i) => (
          <div key={i} className="card-item">
            <span className="card-score">{(m.score * 100).toFixed(0)}%</span>
            <span className="card-text">{m.text}</span>
          </div>
        ))
      ) : (
        <p className="card-err">
          {loading ? "Contacting CLIP backend..." : "CLIP backend not available (start clip_server.py)"}
        </p>
      )}
    </div>
  );
};

const App = () => {
  const [session, setSession] = useState(null);
  const [loading, setLoading] = useState({ text: "Loading OpenCV.js", progress: null });
  const [image, setImage] = useState(null);
  const [boxes, setBoxes] = useState(null);
  const inputImage = useRef(null);
  const imageRef = useRef(null);
  const canvasRef = useRef(null);

  const modelName = "model.onnx";
  const modelInputShape = [1, 3, 640, 640];
  const topk = 100;
  const iouThreshold = 0.45;
  const scoreThreshold = 0.25;

  cv["onRuntimeInitialized"] = async () => {
    setLoading({ text: "Loading model...", progress: null });
    const yolov8 = await InferenceSession.create('./model.onnx');

    setLoading({ text: "Warming up nms...", progress: null });
    const nms = await InferenceSession.create('./nms-yolov8.onnx');

    setLoading({ text: "Warming up mask...", progress: null });
    const mask = await InferenceSession.create('./mask-yolov8-seg.onnx');

    setLoading({ text: "Warming up model...", progress: null });
    const tensor = new Tensor(
      "float32",
      new Float32Array(modelInputShape.reduce((a, b) => a * b)),
      modelInputShape
    );
    await yolov8.run({ images: tensor });

    setSession({ net: yolov8, nms: nms, mask: mask });
    setLoading(null);
  };

  return (
    <div className="App">
      {loading && (
        <Loader>
          {loading.progress ? `${loading.text} - ${loading.progress}%` : loading.text}
        </Loader>
      )}
      <div className="header">
        <h1>Parrot Instance Segmentation</h1>
        <p>
          YOLOv8s-seg object detection live on browser powered by{" "}
          <code>onnxruntime-web</code>
        </p>
        <p>
          Model : <code className="code">{modelName}</code>
        </p>
      </div>

      <div className="content">
        <img
          ref={imageRef}
          src="#"
          alt=""
          style={{ display: image ? "block" : "none" }}
          onLoad={async () => {
            const result = await detectImage(
              imageRef.current,
              canvasRef.current,
              session,
              topk,
              iouThreshold,
              scoreThreshold,
              modelInputShape
            );
            setBoxes(result || []);
          }}
        />
        <canvas
          id="canvas"
          width={modelInputShape[2]}
          height={modelInputShape[3]}
          ref={canvasRef}
        />
      </div>

      <input
        type="file"
        ref={inputImage}
        accept="image/*"
        style={{ display: "none" }}
        onChange={(e) => {
          if (image) {
            URL.revokeObjectURL(image);
            setImage(null);
            setBoxes(null);
          }
          const url = URL.createObjectURL(e.target.files[0]);
          imageRef.current.src = url;
          setImage(url);
        }}
      />
      <div className="btn-container">
        <button onClick={() => { inputImage.current.click(); }}>
          Open local image
        </button>
        {image && (
          <button onClick={() => {
            inputImage.current.value = "";
            imageRef.current.src = "#";
            URL.revokeObjectURL(image);
            setImage(null);
            setBoxes(null);
          }}>
            Close image
          </button>
        )}
      </div>

      <CardPanel boxes={boxes} imageRef={imageRef} />
    </div>
  );
};
export default App;
