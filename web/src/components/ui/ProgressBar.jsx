export default function ProgressBar({ value, className = "" }) {
  const percentage = Math.max(0, Math.min(100, value));

  return (
    <div
      className={`progress-track ${className}`.trim()}
      role="progressbar"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(percentage)}
    >
      <span style={{ width: `${percentage}%` }} />
    </div>
  );
}
