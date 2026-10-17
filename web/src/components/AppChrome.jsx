import {
  Activity, AlertCircle, ArrowRight, AudioLines, BadgeCheck, Check, CircleHelp,
  LockKeyhole, Moon, Sun, Wifi, WifiOff, X,
} from "lucide-react";
import Button from "./ui/Button.jsx";
import Card from "./ui/Card.jsx";
import { StepLabel } from "./ui/Label.jsx";

export function AppHeader({ apiStatus, socketConnected, theme, onToggleTheme }) {
  return (
    <header className="topbar">
      <div className="brand" aria-label="Latido">
        <span className="brand-mark"><Activity size={19} strokeWidth={2.5} /></span>
        <span>latido<span className="brand-dot">.</span></span>
      </div>
      <div className="topbar-right">
        <span className="status-pill">
          <span className={`status-dot ${apiStatus?.ok ? "is-up" : "is-down"}`} />
          {apiStatus?.ok ? (apiStatus.mode === "simulate" ? "Modo simulación" : "Backend conectado") : "Backend sin conexión"}
        </span>
        <span className="connection-indicator" title={socketConnected ? "Canal en tiempo real activo" : "Canal en tiempo real desconectado"}>
          {socketConnected ? <Wifi size={16} /> : <WifiOff size={16} />}
        </span>
        <Button variant="icon" onClick={onToggleTheme} aria-label="Cambiar tema">
          {theme === "dark" ? <Sun size={17} /> : <Moon size={17} />}
        </Button>
      </div>
    </header>
  );
}

export function AppFooter() {
  return (
    <footer className="app-footer">
      <span>LATIDO <i>·</i> PROTOTIPO ACADÉMICO</span>
      <span><CircleHelp size={13} /> La autenticación es experimental y puede equivocarse.</span>
    </footer>
  );
}

export function SessionModal({ modal, onClose, onContinue, onFinish, onAbandon }) {
  if (!modal) return null;

  const locked = modal.kind === "locked";
  const persistent = ["recovery", "rechecking", "locked"].includes(modal.kind);
  const warning = ["signal", "enroll-signal", "recheck-signal"].includes(modal.kind);

  return (
    <div className="modal-backdrop" role="presentation">
      <Card as="section" className={`modal-card ${locked ? "modal-danger" : ""}`} role="dialog" aria-modal="true" aria-labelledby="modal-title">
        <Button variant="text" className="modal-close" onClick={onClose} aria-label="Cerrar aviso" disabled={persistent}><X size={17} /></Button>
        <div className={`modal-symbol ${warning ? "modal-symbol-warn" : ""} ${locked ? "modal-symbol-danger" : ""}`}>
          {modal.kind === "enrolled"
            ? <BadgeCheck size={23} />
            : modal.kind === "rechecking"
              ? <AudioLines size={23} className="animate-pulse" />
              : locked
                ? <LockKeyhole size={22} />
                : <AlertCircle size={23} />}
        </div>
        <StepLabel>{modal.kind === "recovery" || modal.kind === "rechecking" ? "CONTROL DE IDENTIDAD" : "AVISO DE SESIÓN"}</StepLabel>
        <h2 id="modal-title">{modal.title}</h2>
        <p>{modal.message}</p>
        {modal.kind === "enrolled" && <Button full onClick={onContinue}>Continuar <ArrowRight size={16} /></Button>}
        {modal.kind === "signal" && <Button full onClick={onClose}>Entendido, revisaré la banda <Check size={16} /></Button>}
        {modal.kind === "recovery" && <Button full onClick={onClose}>Ajustar banda y continuar <ArrowRight size={16} /></Button>}
        {modal.kind === "enroll-signal" && <Button full onClick={onClose}>Entendido <Check size={16} /></Button>}
        {modal.kind === "recheck-signal" && <Button full onClick={onClose}>Ajustar la banda <Check size={16} /></Button>}
        {modal.kind === "error" && <Button full onClick={onClose}>Cerrar</Button>}
        {modal.kind === "finish" && (
          <div className="modal-actions">
            <Button variant="quiet" onClick={onClose}>Seguir revisando</Button>
            <Button onClick={onFinish}>Finalizar examen <Check size={16} /></Button>
          </div>
        )}
        {modal.kind === "abandon" && (
          <div className="modal-actions">
            <Button variant="quiet" onClick={onClose}>Continuar</Button>
            <Button onClick={onAbandon}>Salir</Button>
          </div>
        )}
      </Card>
    </div>
  );
}
