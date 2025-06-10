document.addEventListener("DOMContentLoaded", () => {
  const uploadBtn = document.getElementById("uploadBtn");
  const uploadSpinner = document.getElementById("uploadSpinner");
  uploadBtn.addEventListener("click", async () => {
    const f = document.getElementById("csvFile").files[0];
    if (!f) return alert("Please select a CSV file.");
    uploadSpinner.style.visibility = "visible";
    const form = new FormData(); form.append("file", f);
    const res = await fetch("/smarttraffic/api/upload/", { method: "POST", body: form });
    const json = await res.json();
    alert(json.message);
    uploadSpinner.style.visibility = "hidden";
  });

  const predictBtn = document.getElementById("predictBtn");
  const predictSpinner = document.getElementById("predictSpinner");
  const predictMsg = document.getElementById("predictMsg");
  const predictCtx = document.getElementById("predictChart").getContext("2d");
  let predictChart;
  predictBtn.addEventListener("click", async () => {
    const model = document.getElementById("modelPredict").value;
    predictSpinner.style.visibility = "visible";
    const res = await fetch("/smarttraffic/api/predict/", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model })
    });
    const json = await res.json();
    predictMsg.innerText = json.message;
    predictSpinner.style.visibility = "hidden";
    const labels = ["Intersection A","B","C"];
    const data = [Math.random()*1000, Math.random()*1000, Math.random()*1000];
    if (predictChart) predictChart.destroy();
    predictChart = new Chart(predictCtx, {
      type: "bar",
      data: { labels, datasets: [{ label: "Predicted Traffic Volume", data, backgroundColor: ["#007bff","#28a745","#dc3545"] }] }
    });
  });
const optimizeBtn = document.getElementById("optimizeBtn");
const optimizeSpinner = document.getElementById("optimizeSpinner");
const optimizeCtx = document.getElementById("optChart").getContext("2d");

let optChart;

optimizeBtn.addEventListener("click", async () => {
  optimizeSpinner.style.visibility = "visible";

  try {
    const res = await fetch("/smarttraffic/api/optimize/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ intersections: ["A", "B", "C"], model: "rf" })
    });

    if (!res.ok) throw new Error("Request failed");

    const json = await res.json();
    optimizeSpinner.style.visibility = "hidden";

    const labels = Object.keys(json.optimized_durations || {});
    const data = Object.values(json.optimized_durations || {});

    if (optChart) optChart.destroy();

    optChart = new Chart(optimizeCtx, {
      type: "bar",
      data: {
        labels,
        datasets: [{
          label: "Green Light Duration (secs)",
          data,
          backgroundColor: "#28a745"
        }]
      }
    });
  } catch (error) {
    console.error("Optimization error:", error);
    optimizeSpinner.style.visibility = "hidden";
    alert("Failed to optimize traffic. Please check backend or data.");
  }
});


  const congestionBtn = document.getElementById("congestionBtn");
  const congestionSpinner = document.getElementById("congestionSpinner");
  let map, congestionLayer;
  congestionBtn.addEventListener("click", async () => {
    congestionSpinner.style.visibility = "visible";
    const res = await fetch("api/congestion/", { method: "POST" });
    const json = await res.json();
    congestionSpinner.style.visibility = "hidden";
    if (!map) {
      map = L.map("congestionMap").setView([20.3, 85.8], 13);  // Pune center
      L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 19 }).addTo(map);
    }
    if (congestionLayer) congestionLayer.clearLayers();
    congestionLayer = L.layerGroup().addTo(map);
    json.forEach((r, i) => {
      const lat = 20.3 + (Math.random()-0.5)*0.1;
      const lon = 85.8 + (Math.random()-0.5)*0.1;
      L.marker([lat, lon]).addTo(congestionLayer)
        .bindPopup(`Cluster: ${r.cluster}; Volume: ${r.traffic_volume}`);
    });
  });

  const realtimeBtn = document.getElementById("realtimeBtn");
  const realtimeSpinner = document.getElementById("realtimeSpinner");
  const rtCtx = document.getElementById("realtimeChart").getContext("2d");
  const rtChart = new Chart(rtCtx, {
    type: "line",
    data: { labels: [], datasets: [{ label: "Avg Traffic Volume", data: [], borderColor: "#17a2b8", fill: false }] },
    options: { responsive: true, scales: { y: { beginAtZero: true } } }
  });
  realtimeBtn.addEventListener("click", async () => {
    realtimeSpinner.style.visibility = "visible";
    const res = await fetch("/smarttraffic/api/realtime/process/", { method: "POST" });
    const json = await res.json();
    realtimeSpinner.style.visibility = "hidden";
    const now = new Date().toLocaleTimeString();
    rtChart.data.labels.push(now);
    rtChart.data.datasets[0].data.push(json.overall_avg_traffic_volume);
    rtChart.update();
  });
});

const simulateBtn = document.getElementById("simulateBtn");
const simulateSpinner = document.getElementById("simulateSpinner");
const simulationLogs = document.getElementById("simulationLogs");
const simulationSummary = document.getElementById("simulationSummary");

simulateBtn.addEventListener("click", async () => {
  simulateSpinner.style.visibility = "visible";
  simulationLogs.innerHTML = "";
  simulationSummary.innerHTML = "";

  try {
    const res = await fetch("/smarttraffic/api/run-simulation/");
    const json = await res.json();

    simulateSpinner.style.visibility = "hidden";

    if (!json || !json.results || json.results.length === 0) {
      simulationSummary.innerHTML = "<strong>❌ No results returned.</strong>";
      return;
    }

    const last = json.results[json.results.length - 1];
    const clusterDetails = Object.entries(last.cluster_volume_avg)
      .map(([k, v]) => `Cluster ${k}: ${v.toFixed(2)}`)
      .join(", ");

    const lightTimings = Object.entries(last.optimized_lights)
      .map(([k, v]) => `${k}: ${v}s`)
      .join(", ");

    simulationSummary.innerHTML = `
      <strong>✅ Simulation Completed</strong><br>
      Total Chunks Processed: ${json.chunks_processed}<br>
      Last RF Avg: <b>${last.rf_avg_pred.toFixed(2)}</b><br>
      Last LSTM Pred: <b>${last.lstm_pred.toFixed(2)}</b><br>
      Last Cluster Avg: ${clusterDetails}<br>
      Optimized Lights: ${lightTimings}
    `;

    simulationLogs.innerHTML = json.results.map(r =>
      `<li>🧩 Chunk ${r.chunk_id}: RF=${r.rf_avg_pred.toFixed(2)}, LSTM=${r.lstm_pred.toFixed(2)}</li>`
    ).join("");

  } catch (err) {
    simulateSpinner.style.visibility = "hidden";
    simulationSummary.innerHTML = "<span style='color:red;'>❌ Error occurred during simulation.</span>";
    console.error(err);
  }
});
