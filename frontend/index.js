const CONFIG = {
  apiUrl: "https://api.example.com/predict",
};

function getTomorrow() {
  const d = new Date();
  d.setDate(d.getDate() + 1);
  return d.toISOString().slice(0, 10); // YYYY-MM-DD
}

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("forecast-form");
  const dateInput = document.getElementById("date");
  const submitBtn = document.getElementById("submit-btn");
  const loadingEl = document.getElementById("loading");
  const errorEl = document.getElementById("error");
  const resultsEl = document.getElementById("results");
  const resultsBody = document.getElementById("results-body");

  // Default date to tomorrow
  dateInput.value = getTomorrow();

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    // --- Gather & trim inputs ---
    const storeId = document.getElementById("store-id").value.trim();
    const itemId = document.getElementById("item-id").value.trim();
    const date = dateInput.value;
    const horizonRaw = document.getElementById("horizon").value.trim();

    // --- Validate ---
    if (!storeId || !itemId || !date || !horizonRaw) {
      showError("All fields are required.");
      return;
    }

    const horizon = Number(horizonRaw);
    if (!Number.isInteger(horizon) || horizon < 1 || horizon > 30) {
      showError("Horizon must be a whole number between 1 and 30.");
      return;
    }

    // --- UI: loading state ---
    hideResults();
    hideError();
    showLoading();
    setButtonDisabled(true);

    // --- API call ---
    try {
      const response = await fetch(CONFIG.apiUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          store_id: storeId,
          item_id: itemId,
          date: date,
          horizon: horizon,
        }),
      });

      if (!response.ok) {
        throw new Error(`Server responded with ${response.status} ${response.statusText}`);
      }

      const data = await response.json();

      // Build table rows  (expect { predictions: [{date, demand}, ...] })
      if (!data.predictions || !Array.isArray(data.predictions)) {
        throw new Error("Unexpected response format from server.");
      }

      resultsBody.innerHTML = "";
      data.predictions.forEach((p) => {
        const tr = document.createElement("tr");
        tr.innerHTML =
          `<td>${escapeHtml(String(p.date))}</td>` +
          `<td>${formatDemand(p.demand)}</td>`;
        resultsBody.appendChild(tr);
      });

      showResults();
    } catch (err) {
      showError(err.message || "An unexpected error occurred. Please try again.");
    } finally {
      hideLoading();
      setButtonDisabled(false);
    }
  });

  // --- UI helpers ---
  function showLoading() {
    loadingEl.classList.remove("hidden");
  }
  function hideLoading() {
    loadingEl.classList.add("hidden");
  }

  function showError(msg) {
    errorEl.textContent = msg;
    errorEl.classList.remove("hidden");
  }
  function hideError() {
    errorEl.textContent = "";
    errorEl.classList.add("hidden");
  }

  function showResults() {
    resultsEl.classList.remove("hidden");
  }
  function hideResults() {
    resultsBody.innerHTML = "";
    resultsEl.classList.add("hidden");
  }

  function setButtonDisabled(disabled) {
    submitBtn.disabled = disabled;
    submitBtn.textContent = disabled ? "Loading…" : "Get Forecast";
  }

  function formatDemand(value) {
    const n = Number(value);
    return Number.isFinite(n) ? n.toFixed(1) : "—";
  }

  function escapeHtml(str) {
    const map = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
    return str.replace(/[&<>"']/g, (c) => map[c]);
  }
});
