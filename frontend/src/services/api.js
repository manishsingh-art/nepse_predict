import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
});

export const getStocks = async () => {
  const response = await api.get('/stocks');
  return response.data;
};

export const getStockDetail = async (slug) => {
  const response = await api.get(`/stock/${slug}`);
  return response.data;
};

export const runPrediction = async (symbol) => {
  const response = await api.post(`/predict/${symbol}`);
  return response.data;
};

export default api;
