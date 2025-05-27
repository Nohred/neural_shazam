document.addEventListener("DOMContentLoaded", () => {
    let isListening = false;
    let stream;

    const toggleButton = document.getElementById("toggleButton");
    const statusBar = document.getElementById("status");
    const statusDisplay = document.getElementById("top3");
    const coverImg = document.getElementById("cover");

    async function startMicrophone() {
        stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    }

    async function recognitionCycle() {
        await recordAndPredict();     // 10s grabación + análisis
        await delay(2000);            // espera 2 segundos
        if (isListening) recognitionCycle(); // loop si está activo
    }

    function stopRecognitionLoop() {
        if (stream) stream.getTracks().forEach(track => track.stop());
    }

    function delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    function updateUI(predictions, confidences) {
        let output = "Top 3 canciones:\n";
        predictions.forEach((song, i) => {
            output += `${i + 1}. ${song} (${(confidences[i] * 100).toFixed(1)}%)\n`;
        });
        statusDisplay.textContent = output;

        // Mostrar portada top 1 con fallback
        const filename = `covers/${predictions[0].toLowerCase().replace(/\s+/g, "_")}.jpg`;
        coverImg.onerror = () => {
            coverImg.src = "covers/no_cover.jpg";
        };
        coverImg.src = filename;
        coverImg.style.display = "block";

        // Limpia el estado "Analizando..."
        statusBar.textContent = "";
    }

    async function recordAndPredict() {
        const recorder = new MediaRecorder(stream);
        let chunks = [];

        recorder.ondataavailable = event => {
            if (event.data.size > 0) chunks.push(event.data);
        };

        statusBar.textContent = "Escuchando...";
        recorder.start();

        await delay(10000);
        recorder.stop();

        return new Promise(resolve => {
            recorder.onstop = async () => {
                statusBar.textContent = "Analizando...";

                const blob = new Blob(chunks, { type: 'audio/wav' });
                const formData = new FormData();
                formData.append("file", blob, "recording.wav");

                try {
                    const response = await fetch("/predict", {
                        method: "POST",
                        body: formData
                    });
                    const data = await response.json();
                    if (data.predictions) {
                        updateUI(data.predictions, data.confidences);
                    } else {
                        statusDisplay.textContent = "Error al predecir";
                    }
                } catch (err) {
                    console.error(err);
                    statusDisplay.textContent = "Error de red";
                }

                resolve();
            };
        });
    }

    toggleButton.addEventListener("click", async () => {
        if (!isListening) {
            await startMicrophone();
            isListening = true;
            toggleButton.textContent = "⏹️ Detener reconocimiento";
            recognitionCycle();
        } else {
            stopRecognitionLoop();
            toggleButton.textContent = "▶️ Iniciar reconocimiento";
            isListening = false;
            statusBar.textContent = "Reconocimiento detenido.";

        }
    });
});
