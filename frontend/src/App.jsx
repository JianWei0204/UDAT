import React, { useState } from "react";
import UploadForm from "./components/UploadForm";
import DetectionResult from "./components/DetectionResult";
import HistoryList from "./components/HistoryList";

function App() {
  const [result, setResult] = useState(null);

  return (
    <div style={{width:"600px", margin:"auto"}}>
      <h1>YOLOv8 目标检测系统</h1>
      <UploadForm setResult={setResult} />
      {result && <DetectionResult result={result} />}
      <h2>检测历史</h2>
      <HistoryList setResult={setResult} />
    </div>
  );
}

export default App;