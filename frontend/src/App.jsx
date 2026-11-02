import { useCallback, useEffect, useRef, useState } from "react";
import {
  Activity, Camera, Check, ChevronRight, CircleHelp, Clock3, CloudUpload,
  Cpu, Eye, FileImage, Film, Gauge, Github, ImagePlus, LoaderCircle,
  ScanFace, ShieldCheck, Sparkles, Upload, Video, X
} from "lucide-react";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";
const EMPTY_RESULT = null;

function App() {
  const [mode, setMode] = useState("image");
  const [blur, setBlur] = useState(true);
  const [result, setResult] = useState(EMPTY_RESULT);
  const [preview, setPreview] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [apiStatus, setApiStatus] = useState("checking");
  const [cameraOn, setCameraOn] = useState(false);
  const [liveResult, setLiveResult] = useState(null);
  const [videoFile, setVideoFile] = useState(null);
  const [videoProgress, setVideoProgress] = useState(0);
  const [videoFrames, setVideoFrames] = useState([]);
  const [history, setHistory] = useState([]);
  const [fps, setFps] = useState(0);
  const fileRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const cameraRef = useRef(null);
  const streamRef = useRef(null);
  const liveBusyRef = useRef(false);
  const lastFrameRef = useRef(0);

  useEffect(() => {
    fetch(`${API_BASE}/health`)
      .then((r) => { if (!r.ok) throw new Error(); return r.json(); })
      .then((d) => setApiStatus(d.ready ? "online" : "loading"))
      .catch(() => setApiStatus("offline"));
  }, []);

  useEffect(() => () => {
    if (streamRef.current) streamRef.current.getTracks().forEach((track) => track.stop());
    if (preview.startsWith("blob:")) URL.revokeObjectURL(preview);
  }, [preview]);

  const addHistory = (data, label) => {
    setHistory((items) => [{
      id: Date.now(), label, count: data.face_count,
      time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      ms: data.processing_ms
    }, ...items].slice(0, 5));
  };

  const sendFrame = async (blob, endpoint = "image") => {
    const form = new FormData();
    form.append("file", blob, endpoint === "frame" ? "webcam-frame.jpg" : "upload.jpg");
    form.append("blur_faces", String(blur));
    const response = await fetch(`${API_BASE}/api/v1/detect/${endpoint}`, { method: "POST", body: form });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.detail || "Detection request failed.");
    }
    return response.json();
  };

  const handleImage = async (file) => {
    if (!file) return;
    setError(""); setBusy(true); setMode("image"); setResult(null);
    const localUrl = URL.createObjectURL(file);
    setPreview(localUrl);
    try {
      const data = await sendFrame(file, "image");
      setResult(data); addHistory(data, file.name);
      setApiStatus("online");
    } catch (e) { setError(e.message); setApiStatus("offline"); }
    finally { setBusy(false); }
  };

  const stopCamera = useCallback(() => {
    if (streamRef.current) streamRef.current.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (cameraRef.current) cameraRef.current.srcObject = null;
    setCameraOn(false); setLiveResult(null); setFps(0);
  }, []);

  const startCamera = async () => {
    setError("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
      streamRef.current = stream;
      if (cameraRef.current) {
        cameraRef.current.srcObject = stream;
        await cameraRef.current.play();
      }
      setCameraOn(true);
    } catch {
      setError("Camera access was blocked. Allow camera permission and use localhost or HTTPS.");
    }
  };

  useEffect(() => {
    if (!cameraOn) return undefined;
    let active = true;
    const tick = async () => {
      const video = cameraRef.current;
      if (!active || !video || video.readyState < 2 || liveBusyRef.current) return;
      liveBusyRef.current = true;
      const now = performance.now();
      try {
        const canvas = document.createElement("canvas");
        canvas.width = video.videoWidth; canvas.height = video.videoHeight;
        canvas.getContext("2d").drawImage(video, 0, 0);
        const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.78));
        if (!blob || !active) return;
        const data = await sendFrame(blob, "frame");
        if (active) {
          setLiveResult(data);
          if (lastFrameRef.current) setFps(Math.max(0, Math.round(1000 / (now - lastFrameRef.current))));
          lastFrameRef.current = now;
        }
      } catch (e) { if (active) setError(e.message); }
      finally { liveBusyRef.current = false; }
    };
    const timer = window.setInterval(tick, 850);
    return () => { active = false; window.clearInterval(timer); };
  }, [cameraOn, blur]);

  const chooseVideo = (file) => {
    if (!file) return;
    setVideoFile(file); setVideoFrames([]); setVideoProgress(0); setError("");
    setPreview(URL.createObjectURL(file)); setMode("video"); setResult(null);
  };

  const analyzeVideo = async () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || !videoFile) return;
    setBusy(true); setError(""); setVideoFrames([]); setVideoProgress(0);
    const ctx = canvas.getContext("2d");
    const duration = video.duration;
    const samples = Math.min(20, Math.max(1, Math.floor(duration / 2) + 1));
    const collected = [];
    try {
      for (let i = 0; i < samples; i++) {
        video.currentTime = Math.min(i * (duration / samples), Math.max(0, duration - 0.05));
        await new Promise((resolve) => { video.onseeked = resolve; });
        canvas.width = video.videoWidth; canvas.height = video.videoHeight;
        ctx.drawImage(video, 0, 0);
        const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.8));
        const data = await sendFrame(blob, "frame");
        collected.push({ second: Math.round(video.currentTime), ...data });
        setVideoFrames([...collected]);
        setVideoProgress(Math.round(((i + 1) / samples) * 100));
      }
      const best = collected.reduce((a, b) => b.face_count > a.face_count ? b : a, collected[0]);
      setResult(best); addHistory(best, videoFile.name);
    } catch (e) { setError(e.message); }
    finally { setBusy(false); }
  };

  const activeResult = mode === "webcam" ? liveResult : result;
  const faceCount = activeResult?.face_count ?? 0;
  const processing = activeResult?.processing_ms;

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="#"><span className="brand-mark"><ScanFace size={23} /></span><span>FaceSight<span className="brand-ai"> AI</span><small>VISION CONSOLE</small></span></a>
        <div className="side-label">WORKSPACE</div>
        <button className={mode === "image" ? "nav-item active" : "nav-item"} onClick={() => setMode("image")}><ImagePlus size={18}/> Image analysis</button>
        <button className={mode === "video" ? "nav-item active" : "nav-item"} onClick={() => setMode("video")}><Film size={18}/> Video analysis</button>
        <button className={mode === "webcam" ? "nav-item active" : "nav-item"} onClick={() => setMode("webcam")}><Camera size={18}/> Live webcam <span className="live-dot"/></button>
        <div className="side-spacer"/>
        <div className="privacy-card"><ShieldCheck size={19}/><strong>Privacy first</strong><p>Detection only. No identity recognition or biometric profiling.</p><span><Check size={13}/> Frames processed in memory</span></div>
        <a className="github-link" href="https://github.com/ViperZenG/FaceSight-AI" target="_blank" rel="noreferrer"><Github size={17}/> View on GitHub <ChevronRight size={15}/></a>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div className="breadcrumb">Workspace <ChevronRight size={14}/> <strong>{mode === "image" ? "Image analysis" : mode === "video" ? "Video analysis" : "Live webcam"}</strong></div>
          <div className="top-status"><span className={apiStatus === "online" ? "status-dot good" : apiStatus === "offline" ? "status-dot bad" : "status-dot"}/>{apiStatus === "online" ? "API connected" : apiStatus === "offline" ? "API unavailable" : "Connecting API"}</div>
        </header>

        <section className="page-heading">
          <div><div className="eyebrow"><Sparkles size={14}/> COMPUTER VISION / 01</div><h1>Vision <span>workspace.</span></h1><p>Detect, inspect and understand faces in visual data — privately.</p></div>
          <div className="model-pill"><span className="model-icon"><Cpu size={17}/></span><span><b>YuNet</b><small>Pretrained detection model</small></span><span className="model-ready">READY</span></div>
        </section>

        <section className="metrics-grid">
          <Metric icon={<ScanFace size={18}/>} label="FACES DETECTED" value={activeResult ? faceCount : "—"} note={activeResult ? "In current analysis" : "Awaiting input"}/>
          <Metric icon={<Gauge size={18}/>} label="PROCESSING TIME" value={processing != null ? `${processing} ms` : "—"} note="Backend inference time"/>
          <Metric icon={<Activity size={18}/>} label="LIVE FRAME RATE" value={mode === "webcam" && cameraOn ? `~${fps} FPS` : "—"} note="Browser capture estimate"/>
          <Metric icon={<ShieldCheck size={18}/>} label="PRIVACY MODE" value={blur ? "Blur on" : "Standard"} note="Face blur preference"/>
        </section>

        <div className="workspace-grid">
          <section className="panel analysis-panel">
            <div className="panel-header"><div><span className="panel-kicker">INPUT SOURCE</span><h2>{mode === "image" ? "Image detection" : mode === "video" ? "Video sampling" : "Live camera feed"}</h2></div><span className="panel-index">01 / INPUT</span></div>
            <div className="mode-tabs">
              <button className={mode === "image" ? "tab selected" : "tab"} onClick={() => setMode("image")}><FileImage size={16}/> Image</button>
              <button className={mode === "video" ? "tab selected" : "tab"} onClick={() => setMode("video")}><Video size={16}/> Video</button>
              <button className={mode === "webcam" ? "tab selected" : "tab"} onClick={() => setMode("webcam")}><Camera size={16}/> Webcam</button>
            </div>

            {mode === "image" && <div className="input-content">
              <input ref={fileRef} type="file" accept="image/png,image/jpeg,image/webp" hidden onChange={(e) => handleImage(e.target.files?.[0])}/>
              {!preview || !result && !busy ? <button className="dropzone" onClick={() => fileRef.current?.click()}><span className="upload-icon"><CloudUpload size={25}/></span><strong>Drop an image here</strong><span>or <u>browse files</u></span><small>JPG, PNG or WEBP · Max 10 MB</small></button> :
                <div className="preview-wrap"><img src={result?.annotated_image_base64 ? `data:image/jpeg;base64,${result.annotated_image_base64}` : preview} alt="Image analysis preview"/>{busy && <div className="image-loading"><LoaderCircle className="spin"/> Analyzing image…</div>}</div>}
              {preview && <div className="action-row"><button className="button secondary" onClick={() => fileRef.current?.click()}><Upload size={16}/> Choose another</button>{busy && <span className="muted"><LoaderCircle className="spin" size={15}/> Processing</span>}</div>}
            </div>}

            {mode === "video" && <div className="input-content">
              <input type="file" accept="video/mp4,video/webm,video/quicktime" hidden id="video-input" onChange={(e) => chooseVideo(e.target.files?.[0])}/>
              {!videoFile ? <label className="dropzone" htmlFor="video-input"><span className="upload-icon"><Film size={25}/></span><strong>Choose a video file</strong><span>MP4, WEBM or MOV</span><small>Up to 20 sampled frames per video</small></label> :
                <><video ref={videoRef} className="video-preview" src={preview} controls muted onLoadedMetadata={() => setVideoProgress(0)}/><div className="video-meta"><Film size={16}/><span>{videoFile.name}</span><button onClick={() => {setVideoFile(null);setPreview("");setVideoFrames([])}}><X size={15}/></button></div><button className="button primary full" disabled={busy} onClick={analyzeVideo}>{busy ? <><LoaderCircle className="spin" size={16}/> Analyzing frames {videoProgress}%</> : <><ScanFace size={16}/> Analyze video frames</>}</button>{busy && <div className="progress-track"><span style={{width:`${videoProgress}%`}}/></div>}</>}
              <canvas ref={canvasRef} hidden/>
              {videoFrames.length > 0 && <div className="frame-strip"><b>{videoFrames.length} frames analyzed</b><span>Peak: {Math.max(...videoFrames.map(f => f.face_count))} faces</span></div>}
            </div>}

            {mode === "webcam" && <div className="input-content">
              <div className="camera-stage">{cameraOn ? <video ref={cameraRef} autoPlay muted playsInline/> : <div className="camera-placeholder"><span className="camera-ring"><Camera size={28}/></span><b>Camera is turned off</b><small>Your camera stream stays in this browser.</small></div>}{cameraOn && <span className="camera-live"><i/> LIVE</span>}</div>
              <div className="action-row">{!cameraOn ? <button className="button primary" onClick={startCamera}><Camera size={16}/> Start camera</button> : <button className="button danger" onClick={stopCamera}><X size={16}/> Stop camera</button>}<span className="muted">No audio is captured</span></div>
            </div>}

            {error && <div className="error-box"><CircleHelp size={16}/>{error}</div>}
            <div className="panel-footer"><label className="toggle-row"><span className="toggle-copy"><ShieldCheck size={17}/><span><b>Privacy blur</b><small>Blur detected face regions</small></span></span><input type="checkbox" checked={blur} onChange={(e) => setBlur(e.target.checked)}/><span className="toggle-ui"/></label><span className="privacy-note"> <ShieldCheck size={13}/> Processed in memory</span></div>
          </section>

          <section className="panel results-panel">
            <div className="panel-header"><div><span className="panel-kicker">MODEL OUTPUT</span><h2>Detection results</h2></div><span className="panel-index">02 / OUTPUT</span></div>
            <div className="result-count"><div><span className="count-label">TOTAL FACES</span><div className="count-number">{activeResult ? faceCount.toString().padStart(2,"0") : "—"}</div></div><div className="count-art"><ScanFace size={47} strokeWidth={1.2}/><span/></div></div>
            {activeResult ? <div className="result-details"><div className="detail-line"><span>Detection status</span><b className="success-text"><Check size={14}/> Complete</b></div><div className="detail-line"><span>Image dimensions</span><b>{activeResult.image_width} × {activeResult.image_height}</b></div><div className="detail-line"><span>Model confidence</span><b>{activeResult.faces?.length ? `${Math.round(activeResult.faces.reduce((s,f) => s + f.confidence,0) / activeResult.faces.length * 100)}% avg.` : "No detections"}</b></div><div className="detail-line"><span>Face blur</span><b>{activeResult.blur_faces ? "Enabled" : "Disabled"}</b></div></div> : <div className="empty-results"><Eye size={24}/><b>Waiting for visual input</b><span>Run an image, video or webcam analysis to see detection data.</span></div>}
            <div className="faces-list"><div className="list-heading">DETECTED FACES <span>{activeResult?.faces?.length || 0}</span></div>{activeResult?.faces?.length ? activeResult.faces.map((face, i) => <div className="face-row" key={i}><span className="face-number">{String(i+1).padStart(2,"0")}</span><span className="face-name">Face region {i+1}<small>{face.box.width} × {face.box.height} px</small></span><b>{Math.round(face.confidence*100)}%</b></div>) : <div className="no-faces">{activeResult ? "No faces found in this input." : "No face regions to display yet."}</div>}</div>
          </section>
        </div>

        <section className="panel history-panel"><div className="panel-header"><div><span className="panel-kicker">SESSION ACTIVITY</span><h2>Recent analyses</h2></div><span className="panel-index">LOCAL SESSION</span></div>{history.length ? <div className="history-list">{history.map((item) => <div className="history-item" key={item.id}><span className="history-file"><FileImage size={17}/></span><span className="history-name"><b>{item.label}</b><small>{item.time}</small></span><span className="history-faces">{item.count} faces</span><span className="history-time"><Clock3 size={13}/>{item.ms} ms</span></div>)}</div> : <div className="history-empty">Your recent analyses will appear here during this session.</div>}</section>
        <footer className="footer"><span>FACESIGHT AI <i>•</i> COMPUTER VISION CONSOLE</span><span>DETECTION ONLY · NO IDENTITY RECOGNITION</span></footer>
      </main>
    </div>
  );
}

function Metric({ icon, label, value, note }) {
  return <div className="metric-card"><span className="metric-icon">{icon}</span><span className="metric-label">{label}</span><strong>{value}</strong><small>{note}</small></div>;
}

export default App;