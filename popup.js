// Predefined keywords for COVID-related topics
const keywords = [
    "covid", "vaccine", "pandemic", "coronavirus", "mask",
    "quarantine", "lockdown", "infection", "symptoms", "immunity"
];

// Function to compute TF-IDF relevance score
function calculateRelevance(text, keywords) {
    // Normalize text: lowercase and remove punctuation
    const normalizedText = text.toLowerCase().replace(/[^\w\s]/g, " ");
    const textWords = normalizedText.split(/\s+/);

    // Count word occurrences
    const wordCount = {};
    textWords.forEach(word => {
        wordCount[word] = (wordCount[word] || 0) + 1;
    });

    // Calculate relevance score based on keyword matches
    let score = 0;
    keywords.forEach(keyword => {
        if (wordCount[keyword]) {
            score += wordCount[keyword];
        }
    });

    // Normalize the score
    return score / textWords.length;
}

// Add event listener to classify button
document.getElementById("classify").addEventListener("click", async () => {
    const text = document.getElementById("inputText").value;

    if (!text.trim()) {
        alert("Please paste some text to classify!");
        return;
    }

    // Debug: Log the entered text
    console.log("Entered Text:", text);

    // Calculate relevance score
    const relevanceScore = calculateRelevance(text, keywords);

    // Debug: Log the relevance score
    console.log("Relevance Score:", relevanceScore);

    if (relevanceScore < 0.1) { // Adjust threshold for relevance
        document.getElementById("result").innerText =
            "The text is irrelevant to COVID-19 topics.";
        document.getElementById("confidenceChart").style.display = "none"; // Hide chart
        return;
    }

    // If relevant, send to server for classification
    const serverUrl = "http://127.0.0.1:5000/predict";

    try {
        const response = await fetch(serverUrl, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text }),
        });

        const result = await response.json();
        console.log("Server response:", result);

        if (result.probabilities) {
            document.getElementById("result").innerText =
                `Prediction: ${result.predicted_label}`;
            renderChart(result.probabilities);
        } else {
            document.getElementById("result").innerText =
                "Error: Unable to classify text.";
        }
    } catch (error) {
        console.error("Error:", error);
        document.getElementById("result").innerText =
            "Server error. Please try again later.";
    }
});

// Function to render chart
function renderChart(probabilities) {
    const ctx = document.getElementById("confidenceChart").getContext("2d");

    if (window.confidenceChart) {
        window.confidenceChart.destroy?.();
    }

    window.confidenceChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels: ["True", "False", "Misleading"],
            datasets: [{
                label: "Confidence Scores",
                data: [probabilities.true, probabilities.false, probabilities.misleading],
                backgroundColor: ["#4caf50", "#f44336", "#ff9800"],
                borderWidth: 1,
            }],
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                },
            },
        },
    });

    // Show the chart
    document.getElementById("confidenceChart").style.display = "block";
}
