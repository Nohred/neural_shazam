let mediaRecorder;
let audioChunks = [];

const recordButton = document.getElementById("recordButton");
const status = document.getElementById("status");

recordButton.addEventListener("mousedown", async () => {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];

        mediaRecorder.ondataavailable = event => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = () => {
            const blob = new Blob(audioChunks, { type: 'audio/wav' });

            // Mostrar que se terminó la grabación
            status.innerText = "🎧 Procesando audio...";

            // Preparar archivo para envío
            const formData = new FormData();
            formData.append("file", blob, "recording.wav");

            // Enviar al backend
            fetch("/predict", {

                method: "POST",
                body: formData,
            })
            .then(response => response.json())
            .then(data => {
                if (data.predictions) {
                    let resultado = "🔮 Top 3 canciones:\n";
                    data.predictions.forEach((song, i) => {
                        resultado += `${i + 1}. ${song} (Confianza: ${(data.confidences[i] * 100).toFixed(1)}%)\n`;
                    });
                    status.innerText = resultado;
                } else {
                    status.innerText = "❌ Error en la predicción.";
                    console.error(data);
                }
            })
            .catch(err => {
                status.innerText = "❌ Error al contactar el backend.";
                console.error(err);
            });
        };

        mediaRecorder.start();
        status.textContent = "🔴 Grabando...";
    } catch (err) {
        console.error("Error accediendo al micrófono:", err);
        status.textContent = "❌ Error accediendo al micrófono";
    }
});

recordButton.addEventListener("mouseup", () => {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
        mediaRecorder.stop();
    }
});
