(() => {
  "use strict";

  /* ── Elements ──────────────────────────────────────────────────── */
  const dropZone    = document.getElementById("drop-zone");
  const fileInput   = document.getElementById("file-input");
  const previewWrap = document.getElementById("preview-wrapper");
  const previewImg  = document.getElementById("preview-img");
  const clearBtn    = document.getElementById("clear-btn");
  const submitBtn   = document.getElementById("submit-btn");
  const btnText     = document.getElementById("btn-text");
  const spinner     = document.getElementById("spinner");
  const form        = document.getElementById("predict-form");

  // Input fields
  const tempInput = document.getElementById("temperature");
  const humInput  = document.getElementById("humidity");
  const moisInput = document.getElementById("moisture");

  let selectedFile = null;

  /* ── Preview ───────────────────────────────────────────────────── */
  function showPreview(file) {
    previewImg.src = URL.createObjectURL(file);
    previewWrap.style.display = "block";
    updateSubmitState();
  }

  function clearUpload() {
    selectedFile = null;
    fileInput.value = "";
    previewImg.src = "";
    previewWrap.style.display = "none";
    updateSubmitState();
  }

  function updateSubmitState() {
    submitBtn.disabled = !selectedFile;
  }

  /* ── File selection ────────────────────────────────────────────── */
  fileInput.addEventListener("change", () => {
    const f = fileInput.files[0];
    if (!f) return;
    if (!f.type.startsWith("image/")) {
      showFieldError("file-error", "Please select a valid image file.");
      return;
    }
    clearFieldError("file-error");
    selectedFile = f;
    showPreview(f);
  });

  clearBtn.addEventListener("click", (e) => { e.stopPropagation(); clearUpload(); });

  /* ── Drag & drop ───────────────────────────────────────────────── */
  dropZone.addEventListener("dragover",  (e) => { e.preventDefault(); dropZone.classList.add("dragover"); });
  dropZone.addEventListener("dragleave", ()  => dropZone.classList.remove("dragover"));
  dropZone.addEventListener("dragend",   ()  => dropZone.classList.remove("dragover"));
  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    const f = e.dataTransfer?.files[0];
    if (!f || !f.type.startsWith("image/")) {
      showFieldError("file-error", "Please drop a valid image file.");
      return;
    }
    clearFieldError("file-error");
    selectedFile = f;

    // Sync the dropped file into the <input type="file"> so native form
    // submission includes it.  DataTransfer is supported in modern browsers.
    try {
      const dt = new DataTransfer();
      dt.items.add(f);
      fileInput.files = dt.files;
    } catch (_) {
      // Fallback handled in submit via FormData
    }

    showPreview(f);
  });

  /* ── Validation helpers ────────────────────────────────────────── */
  function showFieldError(id, msg) {
    const el = document.getElementById(id);
    if (el) { el.textContent = msg; el.style.display = "block"; }
  }
  function clearFieldError(id) {
    const el = document.getElementById(id);
    if (el) { el.textContent = ""; el.style.display = "none"; }
  }
  function clearAllErrors() {
    ["file-error", "temp-error", "hum-error", "mois-error"]
      .forEach(clearFieldError);
    [tempInput, humInput, moisInput].forEach(el => el.classList.remove("field-error"));
  }

  function validateInputs() {
    let valid = true;
    clearAllErrors();

    if (!selectedFile) {
      showFieldError("file-error", "Please upload a fruit image.");
      valid = false;
    }

    const temp = parseFloat(tempInput.value);
    if (isNaN(temp) || temp < -50 || temp > 80) {
      showFieldError("temp-error", "Enter a temperature between -50 and 80 °C.");
      tempInput.classList.add("field-error");
      valid = false;
    }

    const hum = parseFloat(humInput.value);
    if (isNaN(hum) || hum < 0 || hum > 100) {
      showFieldError("hum-error", "Humidity must be 0–100 %.");
      humInput.classList.add("field-error");
      valid = false;
    }

    const mois = parseFloat(moisInput.value);
    if (isNaN(mois) || mois < 0 || mois > 100) {
      showFieldError("mois-error", "Moisture must be 0–100 %.");
      moisInput.classList.add("field-error");
      valid = false;
    }

    return valid;
  }

  /* ── Form submit ───────────────────────────────────────────────── */
  form.addEventListener("submit", (e) => {
    e.preventDefault();                 // always prevent default — we submit via fetch
    if (!validateInputs()) return;

    // Show loading state
    submitBtn.disabled    = true;
    btnText.textContent   = "Analysing…";
    spinner.style.display = "block";

    // Build FormData manually so the dropped file is always included
    const fd = new FormData();
    fd.append("file",        selectedFile, selectedFile.name);
    fd.append("temperature", tempInput.value);
    fd.append("humidity",    humInput.value);
    fd.append("moisture",    moisInput.value);

    fetch(form.action || "/predict", { method: "POST", body: fd })
      .then((res) => {
        // The server returns an HTML page (result or error).
        // Replace the current document with the response.
        return res.text();
      })
      .then((html) => {
        document.open();
        document.write(html);
        document.close();
      })
      .catch((err) => {
        showFieldError("file-error", "Network error — please try again.");
        submitBtn.disabled    = false;
        btnText.textContent   = "Analyse";
        spinner.style.display = "none";
      });
  });

  /* ── Result page: animate bars ─────────────────────────────────── */
  function animateBars() {
    document.querySelectorAll("[data-pct]").forEach(el => {
      const pct = parseFloat(el.dataset.pct) || 0;
      el.style.width = "0%";
      requestAnimationFrame(() =>
        requestAnimationFrame(() => { el.style.width = pct + "%"; })
      );
    });
  }

  if (document.querySelector(".result-layout")) {
    animateBars();
  }

  /* ── Initial state ─────────────────────────────────────────────── */
  updateSubmitState();
})();
