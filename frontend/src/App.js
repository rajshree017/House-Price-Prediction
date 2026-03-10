import { useState } from "react";
import "./App.css";

function App() {
  const [form, setForm] = useState({
    MedInc: "",
    HouseAge: "",
    AveRooms: "",
    AveBedrms: "",
    Population: "",
    AveOccup: "",
    Latitude: "",
    Longitude: "",
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handlePredict = async () => {
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch("http://localhost:8000/api/predict/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });

      const data = await response.json();

      if (data.status === "success") {
        setResult(data.predicted_price);
      } else {
        setError("Something went wrong. Please check your inputs.");
      }
    } catch (err) {
      setError("Cannot connect to server. Make sure Django is running.");
    }

    setLoading(false);
  };

  const fields = [
    { name: "MedInc", label: "Median Income", placeholder: "e.g. 5.0" },
    { name: "HouseAge", label: "House Age (years)", placeholder: "e.g. 20" },
    { name: "AveRooms", label: "Average Rooms", placeholder: "e.g. 6" },
    { name: "AveBedrms", label: "Average Bedrooms", placeholder: "e.g. 1.5" },
    { name: "Population", label: "Population", placeholder: "e.g. 1500" },
    { name: "AveOccup", label: "Average Occupancy", placeholder: "e.g. 3.0" },
    { name: "Latitude", label: "Latitude", placeholder: "e.g. 37.5" },
    { name: "Longitude", label: "Longitude", placeholder: "e.g. -122.0" },
  ];

  return (
    <div className="container">
      <div className="card">
        <h1>🏠 House Price Predictor</h1>
        <p className="subtitle">Enter house details to get predicted price</p>

        <div className="form-grid">
          {fields.map((field) => (
            <div className="input-group" key={field.name}>
              <label>{field.label}</label>
              <input
                type="number"
                name={field.name}
                placeholder={field.placeholder}
                value={form[field.name]}
                onChange={handleChange}
              />
            </div>
          ))}
        </div>

        <button onClick={handlePredict} disabled={loading}>
          {loading ? "Predicting..." : "Predict Price"}
        </button>

        {result && (
          <div className="result">
            <h2>Predicted Price</h2>
            <p className="price">${result.toLocaleString()}</p>
          </div>
        )}

        {error && <div className="error">{error}</div>}
      </div>
    </div>
  );
}

export default App;