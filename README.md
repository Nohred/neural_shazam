# Neural Shazam

Neural Shazam is a web application that simulates the core functionality of Shazam using a convolutional neural network trained on a curated dataset of 50 songs. The app captures live audio through the browser, processes it in real time, and predicts the most likely matching tracks.

- 🔊 **Live audio recognition** directly from the user's microphone.
- 🎧 **Top 3 song predictions** with confidence scores.
- 🖼️ Displays the album cover for the top result (with fallback image support).

Users can press a button to start continuous recognition sessions, where the app listens in 10-second intervals, analyzes the audio, and updates predictions automatically.
