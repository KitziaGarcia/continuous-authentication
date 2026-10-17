import {
  Activity, AlertCircle, ArrowLeft, ArrowRight, BookOpen, Check, CheckCircle2,
  Clock3, HeartPulse, ShieldCheck, X,
} from "lucide-react";
import { LEVELS } from "../config.js";
import { formatDuration } from "../utils.js";

export default function ExamScreen({
  questions, answers, currentIndex, decision, signalBad, elapsed, onAnswer,
  onPrevious, onNext, onFinishRequest, onExitRequest,
}) {
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
          <button className="icon-button exit-button" onClick={onExitRequest} title="Salir del examen"><X size={17} /></button>
        </div>
      </div>
      <div className="exam-progress">
        <span>Pregunta {String(currentIndex + 1).padStart(2, "0")} <i>de</i> {String(questions.length).padStart(2, "0")}</span>
        <span>{answered} respondidas</span>
        <div className="progress-track"><span style={{ width: `${progress}%` }} /></div>
      </div>
      <div className="exam-columns">
        <article className="reading-card panel">
          <div className="reading-meta">
            <span className="reading-number">LECTURA 0{passageIndex + 1}</span>
            <span>{LEVELS.find((item) => item.value === currentQuestion.level)?.label}</span>
          </div>
          <h2>{currentQuestion.passage_title}</h2>
          <p>{currentQuestion.passage_text}</p>
          <div className="reading-footer"><BookOpen size={15} /> Lee el texto y responde las preguntas asociadas.</div>
        </article>
        <article className="question-card panel">
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
            <button className="button button-quiet" onClick={onPrevious} disabled={currentIndex === 0}><ArrowLeft size={16} /> Anterior</button>
            {currentIndex === questions.length - 1 ? (
              <button className="button button-primary" onClick={onFinishRequest} disabled={signalBad}>Finalizar <Check size={16} /></button>
            ) : (
              <button className="button button-primary" onClick={onNext} disabled={signalBad}>Siguiente <ArrowRight size={16} /></button>
            )}
          </div>
        </article>
      </div>
      <div className="monitor-strip panel">
        <div className="monitor-title"><span className="monitor-pulse"><HeartPulse size={17} /></span><span><b>Autenticación continua</b><small>Señal procesada localmente por el backend</small></span></div>
        <div className="monitor-metric"><small>SIMILITUD</small><b>{decision?.smoothed == null ? "—" : decision.smoothed.toFixed(2)}</b></div>
        <div className="monitor-meter"><span style={{ width: `${Math.max(0, Math.min(100, ((decision?.smoothed ?? 0) + 1) * 50))}%` }} /></div>
        <div className="monitor-caption">{decision?.status === "AUTHENTICATED" ? "Coincide con la muestra inicial" : decision?.status === "BAD_SIGNAL" ? "Sin decisión: señal de baja calidad" : decision?.status === "REJECTED" ? "Se requiere revisar la muestra" : "Esperando la primera lectura"}</div>
      </div>
    </section>
  );
}
