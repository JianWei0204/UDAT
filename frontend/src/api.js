import axios from "axios";
const API_BASE = "http://localhost:8000/api/";

export function uploadFile(file) {
  const formData = new FormData();
  formData.append("file", file);
  return axios.post(API_BASE + "detect/", formData, {
    headers: { "Content-Type": "multipart/form-data" }
  });
}

export function getHistory() {
  return axios.get(API_BASE + "history/");
}