import {
  AlertCircle, ArrowRight, AudioLines, BadgeCheck, BookOpen, Check, ChevronDown, Fingerprint,
  HeartPulse, LockKeyhole, Radio,
} from "lucide-react";
import { DIFFICULTIES, LEVELS } from "../config.js";

function questionOptionsFor(level, difficulty) {
  if (level === "mixto") {
    return difficulty === "todas" ? [10, 20, 30] : difficulty === "facil" ? [3, 6] : [6, 12];
  }
  return difficulty === "todas" ? [5, 10] : difficulty === "facil" ? [2] : [2, 4];
}

function defaultQuestionCount(level, difficulty) {
  if (level === "mixto") return difficulty === "todas" ? 20 : 6;
  return difficulty === "todas" ? 10 : difficulty === "facil" ? 2 : 4;
}

export function WelcomeScreen({
  level, difficulty, questionCount, apiAvailable, onLevelChange,
  onDifficultyChange, onQuestionCountChange, onBegin,
}) {
  const questionOptions = questionOptionsFor(level, difficulty);

  function changeLevel(nextLevel) {
    onLevelChange(nextLevel);
    onQuestionCountChange(defaultQuestionCount(nextLevel, difficulty));
  }

  function changeDifficulty(nextDifficulty) {
    onDifficultyChange(nextDifficulty);
    onQuestionCountChange(defaultQuestionCount(level, nextDifficulty));
  }

  return (
    <section className="welcome-layout">
      <div className="welcome-copy">
        <div className="eyebrow"><span className="eyebrow-dot" /> EXPERIENCIA DE INVESTIGACIÓN</div>
        <h1>Una evaluación<br />con <span>identidad continua.</span></h1>
        <p className="lead">Resuelve un examen breve de comprensión lectora mientras una banda ECG verifica que la señal siga correspondiendo a la misma sesión.</p>
        <div className="feature-row">
          <div><span className="feature-icon"><Fingerprint size={18} /></span><span><b>Sin nombre personal</b><small>Solo una clave anónima de sesión</small></span></div>
          <div><span className="feature-icon"><Radio size={18} /></span><span><b>Revisión en vivo</b><small>Señal y estado durante el examen</small></span></div>
        </div>
      </div>
      <section className="setup-card panel">
        <div className="card-heading">
          <div><span className="step-label">CONFIGURAR SESIÓN</span><h2>Tu simulación</h2></div>
          <span className="setup-icon"><BookOpen size={20} /></span>
        </div>
        <label className="field-label" htmlFor="level">Nivel del examen</label>
        <div className="level-grid" id="level">
          {LEVELS.map((item) => (
            <button key={item.value} className={`level-option ${level === item.value ? "selected" : ""}`} onClick={() => changeLevel(item.value)}>
              <span>{item.label}</span><small>{item.detail}</small>
            </button>
          ))}
        </div>
        <div className="field-row">
          <label className="field-block">
            <span className="field-label">Dificultad</span>
            <span className="select-wrap">
              <select value={difficulty} onChange={(event) => changeDifficulty(event.target.value)}>
                {DIFFICULTIES.map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}
              </select><ChevronDown size={15} />
            </span>
          </label>
          <label className="field-block">
            <span className="field-label">Preguntas</span>
            <span className="select-wrap">
              <select value={questionCount} onChange={(event) => onQuestionCountChange(Number(event.target.value))}>
                {questionOptions.map((count) => <option key={count} value={count}>{count} preguntas</option>)}
              </select><ChevronDown size={15} />
            </span>
          </label>
        </div>
        <div className="session-note">
          <LockKeyhole size={16} />
          <span>La plataforma usa una clave temporal y no almacena tus respuestas en una base de datos.</span>
        </div>
        <button className="button button-primary button-full" onClick={onBegin} disabled={!apiAvailable}>
          Preparar autenticación <ArrowRight size={17} />
        </button>
        <p className="small-disclaimer">Prototipo académico. No es un sistema de seguridad ni determina resultados de alto impacto.</p>
      </section>
      <div className="welcome-bottom"><span>POLAR H10 · ECG 130 HZ</span><span>PLATAFORMA DE PRUEBA</span></div>
    </section>
  );
}

export function EnrollmentScreen({
  userId, enrolled, backendPhase, enrollStage, enrollProgress,
  apiAvailable, socketConnected, busy, onStartEnrollment, onBeginExam, onBack,
}) {
  return (
    <section className="center-stage">
      <div className="stage-heading">
        <div className="eyebrow"><span className="eyebrow-dot" /> PASO 01 / AUTENTICACIÓN INICIAL</div>
        <h1>Prepara tu <span>muestra.</span></h1>
        <p>Primero crearemos una referencia temporal de tu señal. No buscamos tu nombre ni tu identidad civil.</p>
      </div>
      <div className="enroll-card panel">
        <div className="sensor-illustration"><div className="pulse-ring"><HeartPulse size={30} /></div><span className="pulse-line" /></div>
        <div className="enroll-copy">
          <span className="step-label">CLAVE DE ESTA SESIÓN</span>
          <div className="session-key">
            {userId}
            <button className="copy-key" onClick={() => navigator.clipboard?.writeText(userId)} aria-label="Copiar clave">
              <Check size={14} />
            </button>
          </div>
          <h2>{enrolled ? "Tu muestra está lista" : backendPhase === "enrolling" ? "Registrando señal" : "Coloca y ajusta la banda"}</h2>
          <p>{enrolled ? "La referencia se conservará solo para esta sesión de examen." : "Humedece los electrodos, coloca la banda firmemente y siéntate en una posición cómoda."}</p>
          <div className="enroll-progress">
            <div className="progress-label"><span>{enrollStage === "settling" ? "Estabilizando la señal" : "Capturando muestra"}</span><span>{enrollProgress}%</span></div>
            <div className="progress-track"><span style={{ width: `${enrollProgress}%` }} /></div>
            <div className="progress-foot"><span>Tiempo aproximado: 65 segundos</span><span>{socketConnected ? "Canal en vivo activo" : "Conectando canal…"}</span></div>
          </div>
          {enrolled ? (
            <button className="button button-primary button-full" onClick={onBeginExam} disabled={busy}>
              Iniciar examen <ArrowRight size={17} />
            </button>
          ) : (
            <button className="button button-primary button-full" onClick={onStartEnrollment} disabled={busy || !apiAvailable || !socketConnected}>
              {busy ? <><AudioLines size={17} className="animate-pulse" /> Esperando señal…</> : <><Radio size={17} /> Iniciar captura de 1 minuto</>}
            </button>
          )}
          <button className="text-button back-link" onClick={onBack}>Volver a configuración</button>
        </div>
      </div>
    </section>
  );
}

export function ReadyScreen({ userId, questionCount, busy, onBeginExam }) {
  return (
    <section className="center-stage">
      <div className="ready-card panel">
        <div className="success-emblem"><BadgeCheck size={32} /></div>
        <div className="eyebrow"><span className="eyebrow-dot" /> IDENTIDAD DE SESIÓN PREPARADA</div>
        <h1>Todo listo para <span>comenzar.</span></h1>
        <p>La lectura inicial quedó asociada a <b>{userId}</b>. Durante el examen volveremos a comprobar la señal.</p>
        <div className="ready-summary">
          <span><BookOpen size={17} /> {questionCount} preguntas de comprensión lectora</span>
          <span><HeartPulse size={17} /> Verificación ECG continua</span>
          <span><LockKeyhole size={17} /> Sin historial personal persistente</span>
        </div>
        <button className="button button-primary button-full" onClick={onBeginExam} disabled={busy}>
          Entrar al examen <ArrowRight size={17} />
        </button>
      </div>
    </section>
  );
}

export function RecoveryScreen({ recheckProgress, busy, onRetry, onAbandon }) {
  const progress = recheckProgress
    ? (recheckProgress.elapsed_s / recheckProgress.total_s) * 100
    : 0;

  return (
    <section className="center-stage">
      <div className="recovery-card panel">
        <div className="warning-emblem"><AlertCircle size={29} /></div>
        <div className="eyebrow"><span className="eyebrow-dot amber" /> VERIFICACIÓN ADICIONAL</div>
        <h1>Vamos a confirmar<br /><span>la señal.</span></h1>
        <p>La sesión está pausada. Comprueba que la banda esté bien colocada, humedece los electrodos si hace falta y mantente quieto.</p>
        {recheckProgress && (
          <div className="enroll-progress recovery-progress">
            <div className="progress-label"><span>{recheckProgress.stage === "settling" ? "Estabilizando" : "Leyendo señal"}</span><span>{Math.round(progress)}%</span></div>
            <div className="progress-track"><span style={{ width: `${progress}%` }} /></div>
          </div>
        )}
        <button className="button button-primary button-full" onClick={onRetry} disabled={busy || Boolean(recheckProgress)}>
          {recheckProgress ? "Tomando muestra…" : "Ya ajusté la banda — comprobar"} <ArrowRight size={17} />
        </button>
        <button className="text-button back-link" onClick={onAbandon}>Salir de la simulación</button>
      </div>
    </section>
  );
}

export function LockedScreen({ onHome }) {
  return (
    <section className="center-stage">
      <div className="locked-card panel">
        <div className="locked-emblem"><LockKeyhole size={29} /></div>
        <div className="eyebrow"><span className="eyebrow-dot red" /> SESIÓN DETENIDA</div>
        <h1>La verificación<br /><span>no fue superada.</span></h1>
        <p>La muestra adicional no coincidió con la referencia inicial. El examen simulado se detuvo y las respuestas no se calificaron.</p>
        <button className="button button-primary button-full" onClick={onHome}>
          Volver al inicio <ArrowRight size={17} />
        </button>
      </div>
    </section>
  );
}

export function ResultsScreen({ result, onHome }) {
  return (
    <section className="center-stage">
      <div className="result-card panel">
        <div className="success-emblem"><BadgeCheck size={31} /></div>
        <div className="eyebrow"><span className="eyebrow-dot" /> SIMULACIÓN COMPLETADA</div>
        <h1>Buen trabajo<span>.</span></h1>
        <p>Terminaste el examen de comprensión lectora.</p>
        <div className="score-number">{result.percentage}<small>%</small></div>
        <div className="score-caption">{result.correct} correctas de {result.total}</div>
        <button className="button button-primary button-full" onClick={onHome}>
          Volver al inicio <ArrowRight size={17} />
        </button>
      </div>
    </section>
  );
}
