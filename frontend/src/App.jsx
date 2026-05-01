import React, { useEffect, useState } from "react";

var API_BASE_URL = "http://127.0.0.1:8000";

function ResultCard(props) {
  if (!props.result) {
    return (
      <div className="card">
        <h3>Prediction Result</h3>
        <p className="muted">Run a prediction to see the full model output.</p>
      </div>
    );
  }

  return (
    <div className="card">
      <h3>Prediction Result</h3>
      <p><strong>School:</strong> {props.result.predicted_school}</p>
      <p><strong>Confidence:</strong> {props.result.confidence_score}</p>
      <p><strong>Complexity Index:</strong> {props.result.complexity_index}</p>
      <div>
        <strong>Top 3 Recommendations:</strong>
        <ol>
          {props.result.top_3_recommendations.map(function mapItem(item, index) {
            return <li key={index}>{item}</li>;
          })}
        </ol>
      </div>
    </div>
  );
}

function HealthBadge(props) {
  var cls = "badge pending";
  var label = "Checking API";
  if (props.status === "ok") {
    cls = "badge ok";
    label = "API Online";
  }
  if (props.status === "error") {
    cls = "badge error";
    label = "API Offline";
  }
  return <span className={cls}>{label}</span>;
}

function App() {
  var [text, setText] = useState("");
  var [file, setFile] = useState(null);
  var [loading, setLoading] = useState(false);
  var [result, setResult] = useState(null);
  var [error, setError] = useState("");
  var [healthStatus, setHealthStatus] = useState("pending");

  useEffect(function onMount() {
    checkHealth();
  }, []);

  async function checkHealth() {
    try {
      var response = await fetch(API_BASE_URL + "/health");
      if (!response.ok) {
        throw new Error("Health check failed");
      }
      var data = await response.json();
      if (data.status === "ok") {
        setHealthStatus("ok");
      } else {
        setHealthStatus("error");
      }
    } catch (err) {
      setHealthStatus("error");
    }
  }

  function onTextChange(event) {
    setText(event.target.value);
    setFile(null);
  }

  function onFileChange(event) {
    if (event.target.files && event.target.files.length > 0) {
      setFile(event.target.files[0]);
      setText("");
    }
  }

  async function submitTextPrediction() {
    var payload = { text: text };
    var response = await fetch(API_BASE_URL + "/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(payload)
    });
    if (!response.ok) {
      var errBody = await response.json();
      throw new Error(errBody.detail || "Prediction failed");
    }
    return response.json();
  }

  async function submitFilePrediction() {
    var formData = new FormData();
    formData.append("file", file);

    var response = await fetch(API_BASE_URL + "/predict", {
      method: "POST",
      body: formData
    });
    if (!response.ok) {
      var errBody = await response.json();
      throw new Error(errBody.detail || "File prediction failed");
    }
    return response.json();
  }

  async function onSubmit(event) {
    event.preventDefault();
    setError("");
    setResult(null);
    setLoading(true);

    try {
      var prediction;
      if (file) {
        prediction = await submitFilePrediction();
      } else if (text.trim().length > 0) {
        prediction = await submitTextPrediction();
      } else {
        throw new Error("Please provide either text input or a .txt file.");
      }
      setResult(prediction);
    } catch (err) {
      setError(err.message || "Unexpected error");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="container">
      <header className="header">
        <h1>Philosophical Text Engine</h1>
        <p className="muted">Interactive frontend for the full ML API pipeline.</p>
        <HealthBadge status={healthStatus} />
      </header>

      <section className="grid">
        <form className="card" onSubmit={onSubmit}>
          <h3>Input</h3>
          <label htmlFor="text-input">Raw Text</label>
          <textarea
            id="text-input"
            rows="9"
            value={text}
            placeholder="Paste philosophical text here..."
            onChange={onTextChange}
          />

          <div className="divider">OR</div>

          <label htmlFor="file-input">Upload .txt</label>
          <input id="file-input" type="file" accept=".txt" onChange={onFileChange} />

          <button type="submit" disabled={loading}>
            {loading ? "Running..." : "Predict"}
          </button>
          {error ? <p className="error">{error}</p> : null}
        </form>

        <ResultCard result={result} />
      </section>
    </main>
  );
}

export default App;
