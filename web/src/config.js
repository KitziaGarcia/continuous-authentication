export const LEVELS = [
  { value: "secundaria", label: "Secundaria", detail: "Comprensión y lectura crítica" },
  { value: "preparatoria", label: "Preparatoria", detail: "Interpretación y argumentos" },
  { value: "universidad", label: "Universidad", detail: "Lectura analítica" },
  { value: "mixto", label: "Mixto", detail: "Preguntas de todos los niveles" },
];

export const DIFFICULTIES = [
  { value: "todas", label: "Variada" },
  { value: "facil", label: "Inicial" },
  { value: "intermedia", label: "Intermedia" },
  { value: "avanzada", label: "Avanzada" },
];

const SIGNAL_COPY = {
  FLAT_SIGNAL: "No llega una lectura clara. Revisa que la banda esté ajustada y que los electrodos estén húmedos.",
  INVALID_VALUES: "La señal contiene valores inválidos. Ajusta la banda y vuelve a intentarlo.",
  MOVEMENT_ARTIFACT: "Detectamos movimiento o mal contacto. Siéntate derecho y permanece quieto unos segundos.",
  FEW_BEATS: "No se detectan suficientes latidos. Ajusta la banda y humedece los electrodos.",
  IMPOSSIBLE_HEART_RATE: "La lectura no parece confiable. Revisa el contacto de la banda y respira con normalidad.",
  IRREGULAR_RHYTHM: "La señal presenta variaciones. Quédate quieto y revisa la posición de la banda.",
  SIGNAL_GAP: "La lectura se interrumpió. Comprueba que la banda siga colocada.",
  POOR_SIGNAL: "La señal no es suficientemente clara. Ajusta la banda y vuelve a intentarlo.",
};

export function humanSignal(codes = []) {
  return codes.map((code) => SIGNAL_COPY[code] || "Revisa la banda y vuelve a intentarlo.");
}
