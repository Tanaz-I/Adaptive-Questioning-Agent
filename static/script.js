function start() {
    let fileInput = document.getElementById("file");

    if (!fileInput.files.length) {
        alert("Please upload a file");
        return;
    }

    let file = fileInput.files[0];

    let formData = new FormData();
    formData.append("file", file);

    document.getElementById("uploadBox").classList.add("hidden");
    document.getElementById("loading").classList.remove("hidden");

    fetch("/start", { method: "POST", body: formData })
        .then(() => check())
        .catch(err => {
            console.error("Start error:", err);
            alert("Failed to start session");
        });
}


// 🔄 CHECK STATUS
function check() {
    let intv = setInterval(() => {
        fetch("/status")
        .then(r => r.json())
        .then(data => {

            if (data.ready) {
                clearInterval(intv);

                document.getElementById("loading").classList.add("hidden");
                document.getElementById("qaBox").classList.remove("hidden");

                document.getElementById("question").innerText = data.question;
            }
        })
        .catch(err => {
            console.error("Status error:", err);
        });
    }, 2000);
}


// ✅ SUBMIT ANSWER (FIXED)
function submitAnswer() {
    let ans = document.getElementById("answer").value;

    if (!ans.trim()) {
        alert("Please enter an answer");
        return;
    }

    fetch("/submit", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ answer: ans })
    })
    .then(r => {
        if (!r.ok) {
            throw new Error("Server error during submit");
        }
        return r.json();
    })
    .then(data => {

        console.log("Submit response:", data); // DEBUG

        let resultDiv = document.getElementById("result");

        resultDiv.classList.remove("hidden");

        resultDiv.innerHTML = `
            <div style="margin-top:15px; padding:15px; background:#f5f7ff; border-radius:10px;">
                <b>Score:</b> ${data.score}<br>
                <b>Reward:</b> ${data.reward}<br><br>
                <b>Reference Answer:</b><br>
                ${data.reference}
            </div>
        `;
    })
    .catch(err => {
        console.error("Submit error:", err);
        alert("Submit failed. Check backend.");
    });
}


// ➡️ NEXT QUESTION (FIXED)
function nextQ() {
    fetch("/next")
    .then(r => r.json())
    .then(data => {

        document.getElementById("question").innerText = data.question;

        document.getElementById("answer").value = "";

        document.getElementById("result").classList.add("hidden");
        document.getElementById("result").innerHTML = "";
    })
    .catch(err => {
        console.error("Next error:", err);
    });
}


// ❌ QUIT SESSION (IMPROVED)
function quit() {
    fetch("/quit")
    .then(r => r.json())
    .then(data => {

        document.getElementById("qaBox").classList.add("hidden");
        document.getElementById("summary").classList.remove("hidden");

        let html = "<h3>Session Summary</h3><br>";

        data.history.forEach((item, i) => {
            html += `
                <div style="margin-bottom:10px; padding:10px; background:#f5f5f5; border-radius:8px;">
                    <b>Q${i+1}</b><br>
                    Score: ${item.score}<br>
                    Reward: ${item.reward}
                </div>
            `;
        });

        document.getElementById("summary").innerHTML = html;
    })
    .catch(err => {
        console.error("Quit error:", err);
    });
}