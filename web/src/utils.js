export function makeUserId() {
  const token = globalThis.crypto?.randomUUID?.().replaceAll("-", "")
    || Math.random().toString(36).slice(2) + Date.now().toString(36);
  return `EX-${token.slice(0, 10)}`;
}

export function errorText(error) {
  return error instanceof Error ? error.message : "Ocurrió un error inesperado.";
}

export function formatDuration(seconds) {
  const minutes = Math.floor(seconds / 60).toString().padStart(2, "0");
  const remainder = Math.floor(seconds % 60).toString().padStart(2, "0");
  return `${minutes}:${remainder}`;
}
