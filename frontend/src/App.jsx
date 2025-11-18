// Import useState hook from React for managing component state
import { useState } from "react";
// Import axios for making HTTP requests
import axios from "axios";
// Import necessary chart components from Recharts for plotting
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";

function App() {
  // State to store the chemical formula input by the user
  const [formula, setFormula] = useState("");
  // State to store prediction results from the backend
  const [predictions, setPredictions] = useState([]);
  // State to show loading status when prediction is running
  const [loading, setLoading] = useState(false);
  // State to store error messages
  const [error, setError] = useState("");
  // State to toggle between table and plot view
  const [showPlot, setShowPlot] = useState(false); // <-- new state for visualization

  // Function that handles prediction when user clicks "Predict"
  const handlePredict = async () => {
    // If the input is empty, show error and stop
    if (!formula) {
      setError("Please enter a chemical formula");
      return;
    }

    // Start loading
    setLoading(true);
    // Clear previous errors
    setError("");
    // Hide plot view when new prediction is made
    setShowPlot(false);
    // Clear old predictions
    setPredictions([]);

    try {
      // Send POST request to the FastAPI backend
      const response = await axios.post("http://127.0.0.1:8000/predict", { formula });

      // If backend returned predictions
      if (response.data.predictions) {
        // Sort predictions by formation energy (ascending)
        const sorted = [...response.data.predictions].sort((a, b) => a.formation_energy - b.formation_energy);

        // Extract top 5 spacegroups with lowest formation energies
        const top5Spacegroups = sorted.slice(0, 5).map(p => p.spacegroup);

        // Annotate all predictions with a highlight flag for plotting and table highlighting
        const annotated = response.data.predictions.map(p => ({
          ...p,
          highlight: top5Spacegroups.includes(p.spacegroup)
        }));

        // Save annotated predictions into state
        setPredictions(annotated);

      // If backend returned an error
      } else if (response.data.error) {
        setError(response.data.error);
      }
    } catch (err) {
      // Handle network or server errors
      setError("Failed to fetch prediction");
    } finally {
      // Stop loading
      setLoading(false);
    }
  };

  // When user clicks "Visualize Results", show the plot
  const handleVisualize = () => {
    setShowPlot(true);
  };

  return (
    // Main container styling
    <div style={{ maxWidth: 800, margin: "50px auto", fontFamily: "Arial, sans-serif" }}>
      <h1>Formation Energy Predictor</h1>

      {/* Input field for entering chemical formula */}
      <input
        type="text"
        placeholder="Enter chemical formula (e.g., FeO2)"
        value={formula}
        onChange={(e) => setFormula(e.target.value)} // Update formula when user types
        style={{ width: "100%", padding: "10px", fontSize: "16px", marginBottom: "10px" }}
      />

      {/* Predict button */}
      <button
        onClick={handlePredict} // Trigger prediction
        style={{
          padding: "10px 20px",
          fontSize: "16px",
          backgroundColor: "#4CAF50",
          color: "white",
          border: "none",
          cursor: "pointer",
          marginRight: "10px"
        }}
        disabled={loading} // Disable button while loading
      >
        {loading ? "Predicting..." : "Predict"} {/* Show loading text */}
      </button>

      {/* Show visualize button only if predictions exist */}
      {predictions.length > 0 && (
        <button
          onClick={handleVisualize} // Show plot
          style={{
            padding: "10px 20px",
            fontSize: "16px",
            backgroundColor: "#2196F3",
            color: "white",
            border: "none",
            cursor: "pointer"
          }}
        >
          Visualize Results
        </button>
      )}

      {/* Display error message */}
      {error && <div style={{ marginTop: 20, color: "red" }}>{error}</div>}

      {/* Prediction results table */}
      {predictions.length > 0 && (
        <div style={{ marginTop: 20, maxHeight: 300, overflowY: "scroll" }}>
          <table style={{ width: "100%", borderCollapse: "collapse" }}>
            <thead>
              <tr>
                <th style={{ border: "1px solid #ccc", padding: "5px" }}>Spacegroup</th>
                <th style={{ border: "1px solid #ccc", padding: "5px" }}>Formation Energy (eV/atom)</th>
              </tr>
            </thead>
            <tbody>
              {/* Map through prediction data and render each row */}
              {predictions.map((p) => (
                <tr key={p.spacegroup} style={{ backgroundColor: p.highlight ? "#ffefc1" : "transparent" }}>
                  <td style={{ border: "1px solid #ccc", padding: "5px", textAlign: "center" }}>{p.spacegroup}</td>
                  <td style={{ border: "1px solid #ccc", padding: "5px", textAlign: "center" }}>{p.formation_energy.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* --------------------- Plot Section --------------------- */}
      {showPlot && predictions.length > 0 && (
        <div style={{ marginTop: 40 }}>
          <h3>Formation Energy vs Spacegroup</h3>

          {/* Responsive chart container */}
          <ResponsiveContainer width="100%" height={400}>
            <ScatterChart>
              {/* Grid lines */}
              <CartesianGrid strokeDasharray="3 3" />

              {/* X-axis showing spacegroup numbers */}
              <XAxis dataKey="spacegroup" label={{ value: "Spacegroup", position: "insideBottomRight", offset: 0 }} />

              {/* Y-axis showing formation energy */}
              <YAxis label={{ value: "Formation Energy (eV/atom)", angle: -90, position: "insideLeft" }} />

              {/* Tooltip showing values when hovering */}
              <Tooltip formatter={(value) => value.toFixed(4)} cursor={{ strokeDasharray: "3 3" }} />

              {/* Scatter plot for all points */}
              <Scatter data={predictions} fill="#8884d8" line={{ stroke: "#8884d8", strokeWidth: 2 }} />

              {/* Highlighted (top 5) points in red */}
              <Scatter data={predictions.filter(p => p.highlight)} fill="#ff4d4f" />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

export default App; // Export component so it can be rendered by React
