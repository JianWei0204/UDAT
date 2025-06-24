import React, { useRef, useState } from "react";
import axios from "axios";

export default function UploadForm({ setResult }) {
  const inputRef = useRef();
  const [loading, setLoading] = useState(false);

  const handleSubmit = async e => {
    e.preventDefault();
    setLoading(true);
    const formData = new FormData();
    formData.append("file", inputRef.current.files[0]);
    try {
      const res = await axios.post("http://localhost:8000/api/detect/", formData, {
        headers: { "Content-Type": "multipart/form-data" }
      });
      setResult(res.data);
    } catch (err) {
      alert("检测失败：" + err.response?.data.detail || err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <input type="file" ref={inputRef} required />
      <button type="submit" disabled={loading}>上传并检测</button>
    </form>
  );
}