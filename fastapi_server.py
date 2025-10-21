from flask import Flask, request, jsonify, render_template_string
from flask_sock import Sock
import json

app = Flask(__name__)
sock = Sock(app)

# Store connected WebSocket clients
connected_clients = []

# HTML for frontend
html = """
<!DOCTYPE html>
<html>
<head>
    <title>Security System Notifications</title>
    <style>
        body { font-family: Arial; margin: 20px; }
        #notifications { border: 1px solid #ccc; padding: 10px; max-height: 400px; overflow-y: auto; }
        .notification { margin: 5px 0; padding: 5px; background: #f0f0f0; }
    </style>
</head>
<body>
    <h1>Security System Notifications</h1>
    <div id="notifications"></div>
    <script>
        let ws = new WebSocket("ws://localhost:8000/ws");
        
        ws.onopen = () => console.log("Connected to WebSocket");
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            const div = document.createElement("div");
            div.className = "notification";
            div.textContent = `Motion detected at ${data.timestamp}, Face: ${data.face_result}`;
            document.getElementById("notifications").prepend(div);
            alert(`New motion detected at ${data.timestamp}\nFace: ${data.face_result}`);
        };
        ws.onclose = () => {
            console.log("WebSocket closed, reconnecting...");
            setTimeout(() => {
                ws = new WebSocket("ws://localhost:8000/ws");
            }, 5000);
        };
        ws.onerror = (error) => console.error("WebSocket error:", error);
    </script>
</body>
</html>
"""

@app.route("/")
def get():
    return render_template_string(html)

@sock.route("/ws")
def websocket(ws):
    connected_clients.append(ws)
    try:
        while True:
            ws.receive()  # Keep connection alive
    except Exception:
        connected_clients.remove(ws)

@app.route("/notify", methods=["POST"])
def notify():
    try:
        notification = request.get_json()
        if not notification or "timestamp" not in notification or "face_result" not in notification:
            return jsonify({"error": "Invalid notification format"}), 400
        
        # Broadcast to all connected clients
        message = json.dumps(notification)
        disconnected = []
        for client in connected_clients:
            try:
                client.send(message)
            except Exception:
                disconnected.append(client)
        # Remove disconnected clients
        for client in disconnected:
            connected_clients.remove(client)
        
        return jsonify({"status": "notification sent"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)