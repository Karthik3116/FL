(function () {
    "use strict";

    const dropzone = document.getElementById("dropzone");
    if (!dropzone) return;

    const fileInput = document.getElementById("file-input");
    const dropzoneEmpty = document.getElementById("dropzone-empty");
    const dropzonePreview = document.getElementById("dropzone-preview");
    const previewImg = document.getElementById("preview-img");
    const previewName = document.getElementById("preview-name");
    const clearBtn = document.getElementById("clear-btn");
    const submitBtn = document.getElementById("submit-btn");
    const form = document.getElementById("upload-form");
    const formError = document.getElementById("form-error");
    const resultsSection = document.getElementById("results-section");

    function showError(msg) {
        formError.textContent = msg;
        formError.hidden = false;
    }
    function hideError() {
        formError.hidden = true;
        formError.textContent = "";
    }

    function setSelected(file) {
        if (!file) return;
        if (!file.type.startsWith("image/")) {
            showError("Selected file is not an image.");
            return;
        }
        hideError();
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImg.src = e.target.result;
            previewName.textContent = file.name;
            dropzoneEmpty.hidden = true;
            dropzonePreview.hidden = false;
            submitBtn.disabled = false;
        };
        reader.readAsDataURL(file);

        const dt = new DataTransfer();
        dt.items.add(file);
        fileInput.files = dt.files;
    }

    function clearSelected() {
        fileInput.value = "";
        previewImg.src = "";
        previewName.textContent = "";
        dropzoneEmpty.hidden = false;
        dropzonePreview.hidden = true;
        submitBtn.disabled = true;
        hideError();
        if (resultsSection) {
            resultsSection.hidden = true;
            resultsSection.innerHTML = "";
        }
    }

    dropzone.addEventListener("click", (e) => {
        if (e.target.closest("#dropzone-preview")) return;
        fileInput.click();
    });
    dropzone.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            fileInput.click();
        }
    });

    ["dragenter", "dragover"].forEach((evt) => {
        dropzone.addEventListener(evt, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add("is-dragover");
        });
    });
    ["dragleave", "drop"].forEach((evt) => {
        dropzone.addEventListener(evt, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove("is-dragover");
        });
    });
    dropzone.addEventListener("drop", (e) => {
        const files = e.dataTransfer.files;
        if (files && files.length > 0) {
            setSelected(files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        const f = e.target.files && e.target.files[0];
        if (f) setSelected(f);
    });

    clearBtn.addEventListener("click", clearSelected);

    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        if (!fileInput.files || fileInput.files.length === 0) {
            showError("Please choose an image first.");
            return;
        }
        hideError();
        submitBtn.classList.add("is-loading");
        submitBtn.disabled = true;

        const fd = new FormData(form);
        try {
            const res = await fetch(form.action, {
                method: "POST",
                body: fd,
                headers: { "X-Requested-With": "XMLHttpRequest" },
            });
            const data = await res.json();
            if (!res.ok || !data.ok) {
                throw new Error(data.error || `Server returned ${res.status}`);
            }
            renderResults(data);
        } catch (err) {
            showError(err.message || "Inference failed.");
        } finally {
            submitBtn.classList.remove("is-loading");
            submitBtn.disabled = false;
        }
    });

    function renderResults(data) {
        const { results, original_url, preview_data_url, filename, class_names } = data;

        const html = `
            <h2 class="results-title">Prediction Results</h2>
            <div class="result-grid">
                <div class="card image-card">
                    <h3>Uploaded Image</h3>
                    <img class="full-img" src="${original_url}" alt="${escapeHtml(filename)}" />
                    <div class="image-sub">
                        <h4>Preprocessed (28&times;28)</h4>
                        <img class="thumb-img" src="${preview_data_url}" alt="Preprocessed" />
                    </div>
                </div>
                <div class="card consensus-card">
                    <h3>Consensus</h3>
                    <div class="big-digit">${results.consensus.predicted_label}</div>
                    <div class="agreement">
                        ${results.consensus.agreement} / ${results.consensus.total_models} models agree
                        ${results.consensus.unanimous ? '<span class="badge badge-good">Unanimous</span>' : ""}
                    </div>
                    <div class="model-row global-row">
                        <h4>Global Model</h4>
                        ${renderPrediction(results.global, class_names)}
                    </div>
                </div>
            </div>
            <div class="clients-grid">
                ${results.clients.map((c) => `
                    <div class="card client-card">
                        <header class="client-header">
                            <h3>${escapeHtml(c.name)}</h3>
                            <code>${escapeHtml(c.source)}</code>
                        </header>
                        ${renderPrediction(c, class_names, true)}
                    </div>
                `).join("")}
            </div>
        `;

        resultsSection.innerHTML = html;
        resultsSection.hidden = false;
        resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    function renderPrediction(model, classNames, compact) {
        const conf = (model.confidence * 100).toFixed(1);
        const bars = model.probabilities.map((p, idx) => {
            const pct = (p * 100).toFixed(2);
            return `
                <div class="prob-row">
                    <span class="prob-label">${classNames[idx]}</span>
                    <div class="prob-track"><div class="prob-fill" style="width:${pct}%"></div></div>
                    <span class="prob-value">${(p * 100).toFixed(1)}%</span>
                </div>
            `;
        }).join("");
        return `
            <div class="prediction">
                <span class="pred-label">${escapeHtml(model.predicted_label)}</span>
                <span class="pred-conf">${conf}%</span>
            </div>
            <div class="prob-bars${compact ? " compact" : ""}">${bars}</div>
        `;
    }

    function escapeHtml(s) {
        if (s == null) return "";
        return String(s)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#39;");
    }
})();
