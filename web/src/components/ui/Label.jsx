export function Eyebrow({ children, tone = "default", className = "" }) {
  return (
    <div className={`eyebrow ${className}`.trim()}>
      <span className={`eyebrow-dot ${tone === "default" ? "" : tone}`.trim()} />
      {children}
    </div>
  );
}

export function StepLabel({ children, className = "" }) {
  return <span className={`step-label ${className}`.trim()}>{children}</span>;
}
