import React from "react";

export default function DetectionResult({ result }) {
  const { result_image, boxes } = result;
  return (
    <div>
      <h2>检测结果</h2>
      <img
        src={`http://localhost:8000/api/result?result_image=${encodeURIComponent(result_image)}`}
        alt="检测结果"
        style={{width: "100%"}}
      />
      <h4>检测框：</h4>
      <pre style={{maxHeight:200, overflow:"auto"}}>{JSON.stringify(boxes, null, 2)}</pre>
    </div>
  );
}