const questionInput = document.getElementById("question-input");
const askButton = document.getElementById("ask-button");
const statusEl = document.getElementById("status");
const answerOutput = document.getElementById("answer-output");
const sourcesOutput = document.getElementById("sources-output");

askButton.addEventListener("click", handleAsk);

async function handleAsk() {
    const question = questionInput.value.trim();

    if (!question) {
        statusEl.textContent = "Please enter a question.";
        return;
    }

    setLoading(true);

    try {
        const response = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question }),
        });

        if (!response.ok) {
            throw new Error(`Request failed with status ${response.status}`);
        }

        const data = await response.json();
        renderAnswer(data);
    } catch (error) {
        statusEl.textContent = `Error: ${error.message}`;
    } finally {
        setLoading(false);
    }
}

function renderAnswer(data) {
    answerOutput.innerHTML = marked.parse(data.answer);

    sourcesOutput.innerHTML = "";
    for (const source of data.sources) {
        const li = document.createElement("li");
        li.textContent = `Section ${source.section} — ${source.title} (page ${source.page})`;
        sourcesOutput.appendChild(li);
    }

    statusEl.textContent = "";
}

function setLoading(isLoading) {
    askButton.disabled = isLoading;
    askButton.textContent = isLoading ? "Thinking..." : "Ask";
    statusEl.textContent = isLoading ? "Waiting for response..." : statusEl.textContent;
}