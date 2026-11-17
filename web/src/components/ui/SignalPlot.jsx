import { useEffect, useRef } from "react";

export const PREVIEW_SAMPLES = 4 * 130;   // 4 s a 130 Hz, igual que PREVIEW_S en el backend

const WIDTH = 600;
const HEIGHT = 240;
const PAD = 22;
const SCALE_SMOOTHING = 0.03;

function percentile(sorted, p) {
  return sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))];
}

// Traza ECG en vivo. Lee las muestras de `bufferRef` y se redibuja en cada cuadro de pantalla sin
// pasar por React. La escala vertical sale de percentiles (ignora picos raros) y se suaviza entre
// cuadros para que la línea no brinque cuando entra o sale un latido de la ventana.
export default function SignalPlot({ bufferRef }) {
  const lineRef = useRef(null);

  useEffect(() => {
    let frame;
    let range = null;
    const draw = () => {
      const samples = bufferRef.current;
      if (samples.length > 1 && lineRef.current) {
        const sorted = [...samples].sort((a, b) => a - b);
        const lo = percentile(sorted, 0.01);
        const hi = percentile(sorted, 0.995);
        range = range
          ? { lo: range.lo + (lo - range.lo) * SCALE_SMOOTHING, hi: range.hi + (hi - range.hi) * SCALE_SMOOTHING }
          : { lo, hi };
        const span = range.hi - range.lo || 1;
        // La señal entra por la derecha: mientras no llena la ventana, se alinea al borde derecho.
        const step = WIDTH / (PREVIEW_SAMPLES - 1);
        const x0 = WIDTH - (samples.length - 1) * step;
        lineRef.current.setAttribute("points", samples.map((value, i) => {
          const norm = Math.max(-0.1, Math.min(1.1, (value - range.lo) / span));
          return `${(x0 + i * step).toFixed(1)},${(HEIGHT - PAD - norm * (HEIGHT - 2 * PAD)).toFixed(1)}`;
        }).join(" "));
      }
      frame = requestAnimationFrame(draw);
    };
    frame = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(frame);
  }, [bufferRef]);

  return (
    <div className="signal-plot">
      <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} preserveAspectRatio="none" role="img" aria-label="Señal ECG en vivo">
        <polyline ref={lineRef} fill="none" strokeWidth="2" strokeLinejoin="round" vectorEffect="non-scaling-stroke" />
      </svg>
      <span className="signal-plot-label">ECG EN VIVO · ÚLTIMOS 4 s</span>
    </div>
  );
}
