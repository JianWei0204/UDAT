import React, { useEffect, useState } from "react";
import axios from "axios";

export default function HistoryList({ setResult }) {
  const [history, setHistory] = useState([]);

  useEffect(() => {
    axios.get("http://localhost:8000/api/history/").then(res => {
      setHistory(res.data.history);
    });
  }, []);

  return (
    <ul>
      {history.map(item => (
        <li key={item.id} style={{marginBottom:10}}>
          <span>{item.filename} ({item.created_at}) </span>
          <button onClick={() => setResult({result_image: item.result_image, boxes: item.boxes})}>
            查看检测结果
          </button>
        </li>
      ))}
    </ul>
  );
}