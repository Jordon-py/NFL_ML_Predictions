// src/utils/dataFetcher.js
import { API_BASE } from '../api/client'; // Use the centralized API client config

const dataFetcher = async () => {
  try {
    const url = `${API_BASE}/predict/next-week`;
    const response = await fetch(url, {
      headers: {
        'Accept': 'application/json'
      }
    });

    if (!response.ok) {
      throw new Error(`Network response was not ok: ${response.statusText}`);
    }

    return response.json();

  } catch (error) {
    console.error('Error fetching data in dataFetcher:', error);
    // Re-throw the error so the calling component can handle it
    throw error;
  }
};

export {
  dataFetcher
}