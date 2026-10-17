const API_BASE = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

export async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: { "Content-Type": "application/json", ...options.headers },
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = body.detail;
    const message = typeof detail === "string" ? detail : detail?.message;
    throw new Error(message || `Error del servidor (${response.status})`);
  }
  return body;
}

export function openEvents(onEvent, onConnectionChange) {
  const socketUrl = API_BASE.replace(/^http/, "ws") + "/ws";
  const socket = new WebSocket(socketUrl);
  socket.onopen = () => onConnectionChange(true);
  socket.onclose = () => onConnectionChange(false);
  socket.onerror = () => onConnectionChange(false);
  socket.onmessage = (message) => {
    try {
      onEvent(JSON.parse(message.data));
    } catch {
      onConnectionChange(false);
    }
  };
  return socket;
}
