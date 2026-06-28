document.addEventListener("DOMContentLoaded", () => {
	const dropZone = document.getElementById("dropZone");
	const fileInput = document.getElementById("fileInput");
	const fileList = document.getElementById("fileList");
	const form = document.getElementById("uploadForm");
	const submitBtn = document.getElementById("submitBtn");
	const btnText = submitBtn.querySelector(".btn-text");
	const spinner = submitBtn.querySelector(".spinner");
	const results = document.getElementById("results");
	const errorBox = document.getElementById("errorBox");
	const errorMsg = document.getElementById("errorMsg");

	let selectedFiles = [];

	// ─── Drop zone interactions ───
	dropZone.addEventListener("click", () => fileInput.click());
	dropZone.addEventListener("dragover", e => {
    	e.preventDefault();
    	dropZone.classList.add("dragover");
	});
	dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
	dropZone.addEventListener("drop", e => {
    	e.preventDefault();
    	dropZone.classList.remove("dragover");
    	handleFiles(e.dataTransfer.files);
	});
	fileInput.addEventListener("change", () => handleFiles(fileInput.files));

	function handleFiles(files) {
    	selectedFiles = [...files];
    	renderFileList();
    	submitBtn.disabled = selectedFiles.length === 0;
	}

	function renderFileList() {
    	fileList.innerHTML = selectedFiles.map(f => {
        	const ext = f.name.split(".").pop().toUpperCase();
        	return `<div class="file-item"><span class="file-ext">.${ext}</span>${f.name}</div>`;
    	}).join("");
	}

	// ─── Submit ───
	form.addEventListener("submit", async e => {
    	e.preventDefault();
    	if (selectedFiles.length === 0) return;

    	// UI → loading state
    	submitBtn.disabled = true;
    	btnText.textContent = "Processing…";
    	spinner.classList.remove("hidden");
    	results.classList.add("hidden");
    	errorBox.classList.add("hidden");

    	
    	const fd = new FormData();
    	selectedFiles.forEach(f => {
    		const ext = f.name.split(".").pop().toLowerCase();
    		if (["hea", "mat", "dat"].includes(ext)) {
        		fd.append("file", f);              
    		}
		});
    	try {
        	const res = await fetch("/predict", { method: "POST", body: fd });
        	const data = await res.json();

        	if (data.error) {
            	showError(data.error);
        	} else {
            	renderResults(data);
        	}
    	} catch (err) {
        	showError(err.message);
    	} finally {
        	submitBtn.disabled = false;
        	btnText.textContent = "🔬 Analyse ECG";
        	spinner.classList.add("hidden");
    	}
	});

	function showError(msg) {
    	errorMsg.textContent = msg;
    	errorBox.classList.remove("hidden");
	}

	// ─── Render results ───
	function renderResults(data) {
    // ─── Metadata ───
    	const metaGrid = document.getElementById("metaGrid");
    	metaGrid.innerHTML = [
        	metaItem("Record ID",       data.record_id),
        	metaItem("Age",             data.age ?? "N/A"),
        	metaItem("Sex",             data.sex === 1 ? "Male" : data.sex === 0 ? "Female" : "N/A"),
        	metaItem("Original Fs",     (data.fs_orig ?? "—") + " Hz"),
        	metaItem("Target Fs",       "500 Hz"),
        	metaItem("Leads",           12),
        	metaItem("Pred (ECG-only)", data.prediction.pred_ecg_only),
        	metaItem("Pred (ECG+Tab)",  data.prediction.pred_ecg_tab),
        	metaItem("Leads repaired",  data.lead_repair?.n_repaired ?? 0),
    	].join("");

    	// ─── Plots ───
    	document.getElementById("rawImg").src   = "data:image/png;base64," + data.plot_raw;
    	document.getElementById("cleanImg").src = "data:image/png;base64," + data.plot_clean;

    	// ─── Predictions: side-by-side bars for both models ───
    	const cont = document.getElementById("predContainer");
    	const classes = Object.keys(data.prediction.ecg_only);   // ["AFIB","GSVT","SB","SR"]

    	let html = `
        	<div class="pred-models">
            	<div class="pred-col">
                	<h3>ECG-only model → <b>${data.prediction.pred_ecg_only}</b></h3>
                	${classes.map(c => predBar(c, data.prediction.ecg_only[c],
                                          c === data.prediction.pred_ecg_only)).join("")}
            	</div>
            	<div class="pred-col">
                	<h3>ECG + Tabular → <b>${data.prediction.pred_ecg_tab}</b></h3>
                	${classes.map(c => predBar(c, data.prediction.ecg_tab[c],
                                          c === data.prediction.pred_ecg_tab)).join("")}
            	</div>
        	</div>
    	`;
    	cont.innerHTML = html;

    	results.classList.remove("hidden");
    	results.scrollIntoView({ behavior: "smooth", block: "start" });
	}

function predBar(cls, prob, isWinner) {
    const pct   = (prob * 100).toFixed(2);
    const color = isWinner ? "var(--green)" : "var(--accent)";
    return `
        <div class="pred-row">
            <span class="pred-label">${cls}</span>
            <div class="pred-bar-bg">
                <div class="pred-bar" style="width:${pct}%; background:${color}"></div>
            </div>
            <span class="pred-pct">${pct}%</span>
            <span class="pred-badge ${isWinner ? 'pos' : 'neg'}">${isWinner ? 'TOP' : ''}</span>
        </div>
    `;
	}

	function metaItem(label, value) {
    	return `
        	<div class="meta-item">
            	<span class="meta-label">${label}</span>
            	<span class="meta-value">${value ?? "—"}</span>
        	</div>
    	`;
	}

	// ─── Tabs ───
	document.querySelectorAll(".tab").forEach(tab => {
    	tab.addEventListener("click", () => {
        	document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
        	tab.classList.add("active");
        	document.getElementById("rawPlot").classList.add("hidden");
        	document.getElementById("cleanPlot").classList.add("hidden");
        	document.getElementById(tab.dataset.target).classList.remove("hidden");
    	});
	});
});


