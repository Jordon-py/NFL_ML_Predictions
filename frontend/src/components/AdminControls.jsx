
import { useEffect, useState } from "react";
import { getModelsStatus, reloadSystem, retrainModel } from "../api/client";
import "./AdminControls.css"; // We'll create a simple CSS or inline styles

export default function AdminControls() {
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [msg, setMsg] = useState("");
  const [isOpen, setIsOpen] = useState(false);

  useEffect(() => {
    if (isOpen) fetchStatus();
  }, [isOpen]);

  const fetchStatus = async () => {
    try {
      const data = await getModelsStatus();
      setStatus(data);
    } catch (err) {
      setMsg(`Error fetching status: ${err.message}`);
    }
  };

  const handleReload = async () => {
    setLoading(true);
    setMsg("");
    try {
      const res = await reloadSystem();
      setMsg(`Reload success: ${JSON.stringify(res)}`);
      fetchStatus();
    } catch (err) {
      setMsg(`Reload failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleRetrain = async () => {
    if (!window.confirm("Retraining simulates a long process. Continue?")) return;
    setLoading(true);
    setMsg("Training started...");
    try {
      // Assuming retrain simulates a blocking call or we poll; 
      // strict MVP for now just awaits response
      const res = await retrainModel();
      setMsg(`Training success: ${JSON.stringify(res)}`);
      fetchStatus();
    } catch (err) {
      setMsg(`Training failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="admin-controls-container" style={{ border: "1px solid #333", padding: "1rem", marginTop: "2rem", borderRadius: "8px", background: "#111", color: "#eee" }}>
      <button 
        onClick={() => setIsOpen(!isOpen)} 
        style={{ cursor: "pointer", background: "none", border: "none", color: "#aaa", fontSize: "0.9rem" }}
      >
        {isOpen ? "▼ Hide Admin Controls" : "▶ Show Admin Controls"}
      </button>

      {isOpen && (
        <div style={{ marginTop: "1rem" }}>
          <div style={{ display: "flex", gap: "1rem", marginBottom: "1rem" }}>
            <button disabled={loading} onClick={handleReload} className="admin-btn">
              {loading ? "Processing..." : "Reload Backend System"}
            </button>
            <button disabled={loading} onClick={handleRetrain} className="admin-btn" style={{ borderColor: "#d55" }}>
              {loading ? "Processing..." : "Retrain Models (Dev)"}
            </button>
            <button onClick={fetchStatus} className="admin-btn">
              Refresh Status
            </button>
          </div>

          {msg && <div style={{ padding: "0.5rem", background: "#222", marginBottom: "1rem", borderRadius: "4px", fontSize: "0.85rem", color: "#8f8" }}>{msg}</div>}

          {status && (
            <div style={{ fontSize: "0.8rem", color: "#ccc" }}>
              <h4>Current System Status</h4>
              <pre style={{ background: "#000", padding: "0.5rem", overflowX: "auto" }}>
                {JSON.stringify(status, null, 2)}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
