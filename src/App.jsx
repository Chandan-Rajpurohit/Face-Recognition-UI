// src/App.jsx
import { useEffect, useRef, useState } from "react";

const API_URL = "http://127.0.0.1:8000"; // change if your backend runs elsewhere

export default function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [showCamera, setShowCamera] = useState(false);

  // Clean up camera on unmount
  useEffect(() => {
    return () => stopCamera();
  }, []);

  const openCamera = async () => {
    // open only when needed
    if (videoRef.current?.srcObject) return;
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: "user" },
      audio: false,
    });
    setShowCamera(true);
    videoRef.current.srcObject = stream;
    await videoRef.current.play();
    await new Promise((r) => setTimeout(r, 150)); // small settle delay
  };

  const stopCamera = () => {
    const v = videoRef.current;
    if (v?.srcObject) {
      v.srcObject.getTracks().forEach((t) => t.stop());
      v.srcObject = null;
    }
    setShowCamera(false);
    clearOverlay();
  };

  const clearOverlay = () => {
    const c = canvasRef.current;
    if (!c) return;
    const ctx = c.getContext("2d");
    ctx.clearRect(0, 0, c.width, c.height);
  };

  const snapshotDataURL = (w = 480) => {
    const v = videoRef.current;
    if (!v || v.videoWidth === 0) return null;
    const h = Math.round((v.videoHeight / v.videoWidth) * w);
    const canvas = document.createElement("canvas");
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(v, 0, 0, w, h);
    return canvas.toDataURL("image/jpeg", 0.7);
  };

  const captureFrames = async (count = 8, intervalMs = 250) => {
    const frames = [];
    for (let i = 0; i < count; i++) {
      const d = snapshotDataURL(480);
      if (d) frames.push(d);
      await new Promise((r) => setTimeout(r, intervalMs));
    }
    return frames;
  };

  const drawResults = (res) => {
    clearOverlay();
    const v = videoRef.current;
    const c = canvasRef.current;
    if (!v || !c || !res?.results) return;

    // match canvas to video element size
    c.width = v.clientWidth;
    c.height = v.clientHeight;

    const ctx = c.getContext("2d");
    const sx = c.width / res.width;
    const sy = c.height / res.height;

    res.results.forEach((r) => {
      const x1 = Math.round(r.bbox.x1 * sx);
      const y1 = Math.round(r.bbox.y1 * sy);
      const x2 = Math.round(r.bbox.x2 * sx);
      const y2 = Math.round(r.bbox.y2 * sy);
      const isUnknown = r.label === "Unknown";

      // neon rectangle
      ctx.shadowBlur = 12;
      ctx.shadowColor = isUnknown ? "#ff3b6b" : "#00f0ff";
      ctx.lineWidth = 2;
      ctx.strokeStyle = isUnknown ? "#ff3b6b" : "#00f0ff";
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

      // label
      const label = `${r.label} (${r.score.toFixed(2)})`;
      ctx.shadowBlur = 0;
      ctx.fillStyle = "rgba(10,10,10,0.6)";
      const pad = 6;
      const textY = Math.max(18, y1 - 8);
      const textX = x1 + 4;

      ctx.font = "600 14px 'Inter', system-ui, sans-serif";
      const tw = ctx.measureText(label).width;
      ctx.fillRect(textX - pad, textY - 16, tw + pad * 2, 20);
      ctx.fillStyle = isUnknown ? "#ff8aa6" : "#7efcff";
      ctx.fillText(label, textX, textY);
    });
  };

  const withBusy = async (fn) => {
    if (busy) return;
    setBusy(true);
    try {
      await fn();
    } catch (e) {
      console.error(e);
      setMessage(e?.message || "Something went wrong");
    } finally {
      setBusy(false);
    }
  };

  const handleEnroll = () =>
    withBusy(async () => {
      if (!name.trim()) {
        setMessage("Please enter a name to enroll.");
        return;
      }
      setMessage("Opening camera...");
      await openCamera();

      // Optional short countdown
      setMessage("Capturing samples...");
      const frames = await captureFrames(8, 250);

      setMessage("Uploading...");
      const res = await fetch(`${API_URL}/enroll`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: name.trim(), images: frames, min_samples: 6 }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setMessage(data.message || "Enrollment complete");

      // auto close camera
      setTimeout(() => stopCamera(), 600);
    });

  const handleRecognize = () =>
    withBusy(async () => {
      setMessage("Opening camera...");
      await openCamera();

      setMessage("Capturing...");
      const snap = snapshotDataURL(480);
      if (!snap) throw new Error("No frame captured.");

      setMessage("Recognizing...");
      const res = await fetch(`${API_URL}/recognize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: snap }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();

      if (!data.results?.length) {
        setMessage("No faces detected.");
        clearOverlay();
      } else {
        setMessage(
          "Result: " + data.results.map((r) => `${r.label} (${r.score.toFixed(2)})`).join(", ")
        );
        drawResults(data);
      }

      // keep camera visible briefly so user sees overlay, then auto close
      setTimeout(() => stopCamera(), 1200);
    });

  return (
    <div className="wrap">
      <StyleTag />

      <div className="card glass">
        <h1 className="title">Face Recognition Test Page</h1>
        <p className="subtitle">Enroll and Recognize</p>

        <div className="controls">
          <div className="field">
            <label>Display name</label>
            <input
              className="input"
              placeholder="Enter name to enroll"
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
          </div>

          <div className="buttons">
            <button className="btn btn-enroll" onClick={handleEnroll} disabled={busy || !name.trim()}>
              {busy ? "Working..." : "Enroll"}
            </button>
            <button className="btn btn-recognize" onClick={handleRecognize} disabled={busy}>
              {busy ? "Working..." : "Recognize"}
            </button>
          </div>
        </div>

        <div className={`video-area ${showCamera ? "show" : ""}`}>
          <div className="video-frame">
            <video
              ref={videoRef}
              className="video"
              autoPlay
              muted
              playsInline
              style={{ display: showCamera ? "block" : "none" }}
            />
            <canvas ref={canvasRef} className="overlay" />
          </div>
        </div>

        <div className="status">
          <span className="dot" />
          <span>{message || "Idle"}</span>
        </div>
      </div>
    </div>
  );
}

function StyleTag() {
  // Inline global styles (single-file JSX)
  return (
    <style>{`
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

      :root {
        --bg0: #0a0c16;
        --bg1: #101528;
        --glass: rgba(255,255,255,0.06);
        --border: rgba(255,255,255,0.12);
        --text: #e6f2ff;
        --muted: #94a3b8;
        --neon-cyan: #00f0ff;
        --neon-pink: #ff3b6b;
        --neon-purple: #8a2be2;
        --shadow: 0 10px 30px rgba(0,0,0,0.45);
      }

      * { box-sizing: border-box; }
      html, body, #root { height: 100%; }
      body {
        margin: 0;
        background: radial-gradient(1200px 600px at 20% 10%, #10162a, #070a12 60%),
                    linear-gradient(160deg, #11132a 0%, #0a0c16 60%, #06070d 100%);
        color: var(--text);
        font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
      }

      .wrap {
        min-height: 100%;
        display: grid;
        place-items: center;
        padding: 24px;
      }

      .glass {
        background: var(--glass);
        border: 1px solid var(--border);
        border-radius: 16px;
        box-shadow: var(--shadow);
        backdrop-filter: saturate(140%) blur(10px);
      }

      .card {
        width: min(980px, 92vw);
        padding: 24px 24px 20px;
      }

      .title {
        margin: 0 0 6px;
        font-size: 28px;
        font-weight: 800;
        letter-spacing: 0.3px;
        text-shadow: 0 0 8px rgba(0, 240, 255, 0.22);
      }

      .subtitle {
        margin: 0 0 18px;
        color: var(--muted);
        font-size: 14px;
      }

      .controls {
        display: grid;
        grid-template-columns: 1fr auto;
        gap: 14px;
        align-items: end;
      }
      @media (max-width: 820px) {
        .controls {
          grid-template-columns: 1fr;
        }
      }

      .field label {
        font-size: 12px;
        color: var(--muted);
        display: inline-block;
        margin-bottom: 6px;
      }
      .input {
        width: 100%;
        padding: 12px 14px;
        border-radius: 12px;
        border: 1px solid var(--border);
        background: rgba(8,10,18,0.6);
        outline: none;
        color: var(--text);
        transition: border 0.2s ease, box-shadow 0.2s ease;
      }
      .input:focus {
        border-color: rgba(0,240,255,0.5);
        box-shadow: 0 0 0 3px rgba(0,240,255,0.14);
      }

      .buttons {
        display: flex;
        gap: 12px;
        align-items: center;
        justify-content: flex-end;
      }
      @media (max-width: 820px) {
        .buttons {
          justify-content: stretch;
        }
        .btn { flex: 1; }
      }

      .btn {
        cursor: pointer;
        padding: 14px 20px;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.1);
        color: #0a0c16;
        font-weight: 700;
        letter-spacing: 0.4px;
        text-transform: uppercase;
        font-size: 13px;
        transition: transform 0.08s ease, box-shadow 0.2s ease, opacity 0.2s ease;
        box-shadow: inset 0 0 0 1px rgba(255,255,255,0.05),
          0 10px 20px rgba(0,0,0,0.35), 0 0 18px rgba(0, 240, 255, 0.28);
      }
      .btn:disabled {
        opacity: 0.6;
        cursor: default;
        box-shadow: none;
      }
      .btn:active {
        transform: translateY(1px) scale(0.99);
      }

      .btn-enroll {
        background-image: linear-gradient(135deg, #00f0ff 0%, #8a2be2 100%);
        color: #042226;
      }
      .btn-recognize {
        background-image: linear-gradient(135deg, #ff3b6b 0%, #8a2be2 100%);
        color: #2b0616;
      }

      .video-area {
        margin-top: 18px;
        display: grid;
        place-items: center;
        height: 0;
        overflow: hidden;
        transition: height 0.25s ease;
      }
      .video-area.show {
        height: 520px; /* space to reveal video frame */
      }
      .video-frame {
        position: relative;
        width: 720px;
        max-width: 92vw;
        aspect-ratio: 4 / 3;
        border-radius: 16px;
        border: 1px solid var(--border);
        overflow: hidden;
        background: #000;
        box-shadow: var(--shadow);
      }
      .video {
        width: 100%;
        height: 100%;
        object-fit: cover;
        display: block;
        filter: saturate(115%);
      }
      .overlay {
        position: absolute;
        inset: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
      }

      .status {
        margin-top: 14px;
        color: var(--muted);
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 14px;
      }
      .dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: radial-gradient(circle at 35% 35%, #00f0ff, #007a81 70%);
        box-shadow: 0 0 12px rgba(0,240,255,0.6);
      }
    `}</style>
  );
}