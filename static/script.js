/* ══════════════════════════════════════════════
   ADAPTIVE LEARN — script.js
   Multi-file upload + enriched UI interactions
══════════════════════════════════════════════ */

let questionCount = 0;

// ── START (multi-file) ──────────────────────
function start() {
    const fileInput = document.getElementById("file");

    if (!fileInput.files.length) {
        alert("Please upload at least one file.");
        return;
    }

    const formData = new FormData();
    // append ALL selected files under the key "files"
    Array.from(fileInput.files).forEach(f => formData.append("files", f));

    document.getElementById("uploadBox").classList.add("hidden");
    document.getElementById("loading").classList.remove("hidden");

    // Animate loader steps while waiting
    const stepTimer = setInterval(() => {
        if (window._advanceStep) window._advanceStep();
    }, 3500);

    fetch("/start", { method: "POST", body: formData })
        .then(r => {
            if (!r.ok) throw new Error("Server error on /start");
            return r.json();
        })
        .then(() => {
            check(stepTimer);
        })
        .catch(err => {
            clearInterval(stepTimer);
            console.error("Start error:", err);
            alert("Failed to start session. Check the server.");
            document.getElementById("loading").classList.add("hidden");
            document.getElementById("uploadBox").classList.remove("hidden");
        });
}


// ── POLL STATUS ────────────────────────────
function check(stepTimer) {
    const intv = setInterval(() => {
        fetch("/status")
            .then(r => r.json())
            .then(data => {
                if (data.ready) {
                    clearInterval(intv);
                    clearInterval(stepTimer);

                    document.getElementById("loading").classList.add("hidden");
                    document.getElementById("qaBox").classList.remove("hidden");

                    questionCount = 1;
                    renderQuestion(data.question, null, null, null);
                }
            })
            .catch(err => console.error("Status error:", err));
    }, 2000);
}


// ── RENDER QUESTION ────────────────────────
function renderQuestion(question, topic, diff, qtype) {
    document.getElementById("question").innerText = question || "Loading…";
    document.getElementById("qCounter").innerText = "QUESTION  " + questionCount;

    // Update meta pills if provided
    if (topic !== null) {
        document.getElementById("metaTopic").innerText = topic || "—";
        document.getElementById("metaDiff").innerText  = (diff  || "—").toUpperCase();
        document.getElementById("metaType").innerText  = (qtype || "—").replace(/_/g," ");
    }
}


// ── SUBMIT ANSWER ──────────────────────────
function submitAnswer() {
    const ans = document.getElementById("answer").value.trim();
    if (!ans) { alert("Please enter an answer."); return; }

    const btn = document.getElementById("submitBtn");
    btn.disabled = true;
    btn.querySelector("span").textContent = "Evaluating…";

    fetch("/submit", {
        method : "POST",
        headers: { "Content-Type": "application/json" },
        body   : JSON.stringify({ answer: ans })
    })
    .then(r => {
        if (!r.ok) throw new Error("Server error on /submit");
        return r.json();
    })
    .then(data => {
        btn.querySelector("span").textContent = "Submit Answer";
        renderResult(data);
    })
    .catch(err => {
        console.error("Submit error:", err);
        btn.disabled = false;
        btn.querySelector("span").textContent = "Submit Answer";
        alert("Submit failed. Check backend.");
    });
}


// ── RENDER RESULT ──────────────────────────
function renderResult(data) {
    const score  = parseFloat(data.score).toFixed(2);
    const reward = parseFloat(data.reward).toFixed(3);
    const ref    = data.reference || "—";
    const feedback = data.feedback

    const scoreClass =
        data.score >= 0.75 ? "good" :
        data.score >= 0.45 ? "warn" : "bad";

    const resultDiv = document.getElementById("result");
    resultDiv.classList.remove("hidden");
    resultDiv.innerHTML = `
        <div class="result-row">
            <div class="result-score-block">
                <div class="score-item">
                    <span class="score-label">Score</span>
                    <span class="score-val ${scoreClass}">${score}</span>
                </div>
                <div class="score-item">
                    <span class="score-label">RL Reward</span>
                    <span class="score-val" style="font-size:1.2rem;color:var(--muted2)">${reward}</span>
                </div>
            </div>
        </div>
        <div class="result-ref-label">Reference Answer</div>
        <div class="result-ref-body">${ref}</div>
        <br>
        <div class="result-ref-label">Feedback</div>
        <div class="result-ref-body">${feedback}</div>
    `;
}


// ── NEXT QUESTION ──────────────────────────
function nextQ() {
    fetch("/next")
        .then(r => r.json())
        .then(data => {
            questionCount++;

            // Reset textarea + result
            document.getElementById("answer").value = "";
            document.getElementById("submitBtn").disabled = true;
            document.getElementById("result").classList.add("hidden");
            document.getElementById("result").innerHTML = "";

            renderQuestion(
                data.question,
                data.topic  ?? null,
                data.diff   ?? null,
                data.qtype  ?? null
            );
        })
        .catch(err => console.error("Next error:", err));
}


// ── QUIT SESSION ───────────────────────────
function quit() {
    fetch("/quit")
        .then(r => r.json())
        .then(() => {
            window.location.href = "/report";
        })
        .catch(err => console.error("Quit error:", err));
}
