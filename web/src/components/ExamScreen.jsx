import { useState } from "react";
import {
  Activity, AlertCircle, ArrowLeft, ArrowRight, BookOpen, Check, CheckCircle2,
  Clock3, HeartPulse, ShieldCheck, X,
} from "lucide-react";
import { LEVELS } from "../config.js";
import { formatDuration } from "../utils.js";
import Button from "./ui/Button.jsx";
import Card from "./ui/Card.jsx";
import ProgressBar from "./ui/ProgressBar.jsx";

export default function ExamScreen({
  questions, answers, currentIndex, decision, wearerLabel,
  onConfirmWearer, signalBad, elapsed, onAnswer,
  onPrevious, onNext, onFinishRequest, onExitRequest,
}) {
  const [wearerDraft, setWearerDraft] = useState(wearerLabel);
  const currentQuestion = questions[currentIndex];
  const answered = Object.keys(answers).length;
  const progress = questions.length ? ((currentIndex + 1) / questions.length) * 100 : 0;
  const passageIndex = questions
    .filter((question, index) => questions.findIndex((item) => item.passage_id === question.passage_id) === index)
    .findIndex((question) => question.passage_id === currentQuestion.passage_id);
  const displayStatus = decision?.status === "AUTHENTICATED"
    ? "Verificada"
    : decision?.status === "REJECTED"
      ? "Revisión necesaria"
      : decision?.status === "BAD_SIGNAL"
        ? "Señal débil"
        : "Conectando";

  return (
    <section className="exam-layout">
      <div className="exam-topline">
        <div><span className="exam-kicker">EVALUACIÓN DE COMPRENSIÓN</span><h1>Examen <span>simulado</span></h1></div>
        <div className="exam-actions">
          <div className={`auth-chip ${signalBad ? "auth-warning" : decision?.status === "REJECTED" ? "auth-review" : ""}`}>
            {signalBad ? <AlertCircle size={16} /> : decision?.status === "AUTHENTICATED" ? <ShieldCheck size={16} /> : <Activity size={16} />}
            <span>{displayStatus}</span>
          </div>
          <div className="timer-chip"><Clock3 size={15} /> {formatDuration(elapsed)}</div>
          <Button variant="icon" className="exit-button" onClick={onExitRequest} title="Salir del examen"><X size={17} /></Button>
        </div>
      </div>
      <div className="exam-progress">
        <span>Pregunta {String(currentIndex + 1).padStart(2, "0")} <i>de</i> {String(questions.length).padStart(2, "0")}</span>
        <span>{answered} respondidas</span>
        <ProgressBar value={progress} />
      </div>
      <div className="exam-columns">
        <Card as="article" className="reading-card">
          <div className="reading-meta">
            <span className="reading-number">LECTURA 0{passageIndex + 1}</span>
            <span>{LEVELS.find((item) => item.value === currentQuestion.level)?.label}</span>
          </div>
          <h2>{currentQuestion.passage_title}</h2>
          <p>{currentQuestion.passage_text}</p>
          <div className="reading-footer"><BookOpen size={15} /> Lee el texto y responde las preguntas asociadas.</div>
        </Card>
        <Card as="article" className="question-card">
          <div className="question-top">
            <span className="question-index">PREGUNTA {String(currentIndex + 1).padStart(2, "0")}</span>
            <span className="difficulty-badge">{currentQuestion.difficulty}</span>
          </div>
          <h2>{currentQuestion.prompt}</h2>
          <div className="answer-list">
            {Object.entries(currentQuestion.choices).map(([key, choice]) => (
              <button
                key={key}
                className={`answer-option ${answers[currentQuestion.id] === key ? "chosen" : ""}`}
                onClick={() => onAnswer(currentQuestion.id, key)}
                disabled={signalBad}
              >
                <span className="answer-letter">{key}</span><span>{choice}</span>
                {answers[currentQuestion.id] === key && <CheckCircle2 className="answer-check" size={17} />}
              </button>
            ))}
          </div>
          {signalBad && <div className="inline-warning"><AlertCircle size={15} /> Pausamos las respuestas hasta recuperar una señal confiable.</div>}
          <div className="question-footer">
            <Button variant="quiet" onClick={onPrevious} disabled={currentIndex === 0}><ArrowLeft size={16} /> Anterior</Button>
            {currentIndex === questions.length - 1 ? (
              <Button onClick={onFinishRequest} disabled={signalBad}>Finalizar <Check size={16} /></Button>
            ) : (
              <Button onClick={onNext} disabled={signalBad}>Siguiente <ArrowRight size={16} /></Button>
            )}
          </div>
        </Card>
      </div>
      <Card className="monitor-strip">
        <div className="monitor-title"><span className="monitor-pulse"><HeartPulse size={17} /></span><span><b>Autenticación continua</b><small>Señal procesada localmente por el backend</small></span></div>
        <div className="monitor-metric"><small>SIMILITUD</small><b>{decision?.smoothed == null ? "—" : decision.smoothed.toFixed(2)}</b></div>
        <div className="monitor-meter"><span style={{ width: `${Math.max(0, Math.min(100, ((decision?.smoothed ?? 0) + 1) * 50))}%` }} /></div>
        <div className="monitor-caption">{decision?.status === "AUTHENTICATED" ? "Coincide con la muestra inicial" : decision?.status === "BAD_SIGNAL" ? "Sin decisión: señal de baja calidad" : decision?.status === "REJECTED" ? "Se requiere revisar la muestra" : "Esperando la primera lectura"}</div>
        <div className="research-label">
          <label className="research-label-field">
            <span>¿Quién lleva la banda ahora?</span>
            <select value={wearerDraft} onChange={(event) => setWearerDraft(event.target.value)}>
              <option value="original">Participante original — genuine</option>
              <option value="impostor">Otra persona — impostor</option>
              <option value="unknown">No confirmado — unknown</option>
            </select>
          </label>
          <Button
            variant="quiet"
            onClick={() => onConfirmWearer(wearerDraft)}
          >
            Aplicar etiqueta
          </Button>
          <small>
            {wearerLabel === "original"
              ? "Lecturas nuevas: genuine (misma persona que se enroló)"
              : wearerLabel === "impostor"
                ? "Lecturas nuevas: impostor (persona distinta)"
                : "Lecturas nuevas: unknown hasta confirmar"}
          </small>
        </div>
      </Card>
    </section>
  );
}
