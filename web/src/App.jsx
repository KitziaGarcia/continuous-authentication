import { useCallback, useEffect, useRef, useState } from "react";
import { openEvents, request } from "./api.js";
import { AppFooter, AppHeader, SessionModal } from "./components/AppChrome.jsx";
import ExamScreen from "./components/ExamScreen.jsx";
import {
  EnrollmentScreen, LockedScreen, ReadyScreen, RecoveryScreen,
  ResultsScreen, WelcomeScreen,
} from "./components/SessionScreens.jsx";
import { humanSignal } from "./config.js";
import { errorText, makeUserId } from "./utils.js";

export default function App() {
  const [theme, setTheme] = useState(() => localStorage.getItem("latido-theme") || "light");
  const [flow, setFlow] = useState("welcome");
  const flowRef = useRef(flow);
  flowRef.current = flow;
  const [userId] = useState(makeUserId);
  const userIdRef = useRef(userId);
  const [apiStatus, setApiStatus] = useState(null);
  const [socketConnected, setSocketConnected] = useState(false);
  const [backendPhase, setBackendPhase] = useState("idle");
  const [level, setLevel] = useState("mixto");
  const [difficulty, setDifficulty] = useState("todas");
  const [questionCount, setQuestionCount] = useState(20);
  const [authMode, setAuthMode] = useState("test");
  const authModeRef = useRef(authMode);
  authModeRef.current = authMode;
  const [enrollProgress, setEnrollProgress] = useState(0);
  const [enrollStage, setEnrollStage] = useState("settling");
  const [enrolled, setEnrolled] = useState(false);
  const [questions, setQuestions] = useState([]);
  const [wearerLabel, setWearerLabel] = useState("unknown");
  const wearerLabelRef = useRef("unknown");
  const [answers, setAnswers] = useState({});
  const [currentIndex, setCurrentIndex] = useState(0);
  const questionsRef = useRef(questions);
  questionsRef.current = questions;
  const currentIndexRef = useRef(currentIndex);
  currentIndexRef.current = currentIndex;
  const metricReadingsRef = useRef([]);
  const metricsCollectionActiveRef = useRef(false);
  const examTerminatingRef = useRef(false);
  const [decision, setDecision] = useState(null);
  const [signalBad, setSignalBad] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [recheckProgress, setRecheckProgress] = useState(null);
  const [lockedMessage, setLockedMessage] = useState("");
  const [result, setResult] = useState(null);
  const [modal, setModal] = useState(null);
  const modalRef = useRef(modal);
  modalRef.current = modal;
  const recoveryTriggered = useRef(false);
  const badSignalWarned = useRef(false);
  const socketRef = useRef(null);
  const eventHandlerRef = useRef(null);

  const showError = useCallback((title, error) => {
    setModal({ kind: "error", title, message: errorText(error) });
  }, []);

  const startMonitoring = useCallback(async () => {
    for (let attempt = 0; attempt < 5; attempt += 1) {
      try {
        await request("/monitor/start", {
          method: "POST",
          body: JSON.stringify({ user_id: userIdRef.current }),
        });
        recoveryTriggered.current = false;
        setDecision(null);
        setSignalBad(false);
        setModal(null);
        setFlow("exam");
        return;
      } catch (error) {
        if (!error.message.includes("operación en curso") || attempt === 4) {
          showError("No se pudo iniciar la vigilancia", error);
          return;
        }
        await new Promise((resolve) => window.setTimeout(resolve, 150));
      }
    }
  }, [showError]);

  const handleEvent = useCallback((event) => {
    if (event.type === "state") {
      setBackendPhase(event.phase);
      if (event.monitor) setDecision(event.monitor);
      return;
    }
    if (event.type === "enroll_progress") {
      const total = event.stage === "settling" ? 65 : 60;
      const completed = event.stage === "settling" ? event.elapsed_s : 5 + event.elapsed_s;
      setEnrollProgress(Math.min(100, Math.round((completed / total) * 100)));
      setEnrollStage(event.stage);
      if (event.signal_ok === false && flowRef.current === "enroll") {
        setModal((current) => current?.kind === "enroll-signal" ? current : {
          kind: "enroll-signal",
          title: "Ajusta la banda",
          message: humanSignal(event.codes).join(" "),
        });
      } else if (event.signal_ok === true && modalRef.current?.kind === "enroll-signal") {
        setModal(null);
      }
      return;
    }
    if (event.type === "enroll_result") {
      if (event.ok) {
        setEnrolled(true);
        setFlow("ready");
        setModal({ kind: "enrolled", title: "Huella lista", message: "La muestra quedó registrada para esta sesión." });
      } else {
        setEnrolled(false);
        setFlow("enroll");
        setModal({
          kind: "enroll-signal",
          title: "No pudimos guardar la muestra",
          message: (event.messages || humanSignal(event.codes)).join(" "),
        });
      }
      return;
    }
    if (event.type === "recheck_progress") {
      setRecheckProgress(event);
      return;
    }
    if (event.type === "recheck_result") {
      setRecheckProgress(null);
      if (event.kind === "verified") {
        void startMonitoring();
      } else if (event.kind === "signal") {
        setSignalBad(true);
        setModal({
          kind: "recheck-signal",
          title: "La lectura todavía no es clara",
          message: (event.messages || humanSignal(event.codes)).join(" "),
        });
      } else {
        void terminateExamInterrupted("La muestra de verificación no coincidió con la huella inicial. El examen se finalizó y las respuestas no se calificaron.");
      }
      return;
    }
    if (event.type === "decision") {
      setDecision(event);
      if (metricsCollectionActiveRef.current) {
        const question = questionsRef.current[currentIndexRef.current];
        metricReadingsRef.current.push({
          time_seconds: event.t,
          question_number: question ? currentIndexRef.current + 1 : null,
          question_id: question?.id || "",
          question_text: question?.prompt || "",
          wearer_label: wearerLabelRef.current,
          similarity: event.score,
          smoothed_similarity: event.smoothed,
          threshold: event.threshold,
          status: event.status,
          heart_rate_bpm: event.hr_bpm,
        });
      }
      const isBad = event.status === "BAD_SIGNAL";
      setSignalBad(isBad);
      if (isBad && event.bad_signal_streak >= 3 && !badSignalWarned.current) {
        badSignalWarned.current = true;
        if (flowRef.current === "exam") {
          setModal({
            kind: "signal",
            title: "Revisa la banda",
            message: humanSignal(event.codes).join(" "),
          });
        }
      } else if (!isBad) {
        badSignalWarned.current = false;
        if (modalRef.current?.kind === "signal") setModal(null);
      }
      if (event.status === "REJECTED" && flowRef.current === "exam") {
        if (authModeRef.current === "metrics") {
          if (modalRef.current?.kind === "low-match-warning") setModal(null);
        } else if (event.rejection_streak >= 3 && !recoveryTriggered.current) {
          recoveryTriggered.current = true;
          void terminateExamInterrupted("Detectamos una coincidencia muy baja en tres verificaciones consecutivas. Por seguridad, finalizamos el examen y las respuestas no se calificarán. Revisa que la banda esté bien colocada y tenga buen contacto con la piel.");
        } else if (event.rejection_streak === 1 || event.rejection_streak === 2) {
          setModal({
            kind: "low-match-warning",
            title: event.rejection_streak === 1 ? "Coincidencia baja" : "Segunda alerta de coincidencia",
            message: event.rejection_streak === 1
              ? "La lectura tuvo una coincidencia baja con la muestra inicial. Revisa que la banda siga firme y en buen contacto con la piel. El examen continúa por ahora."
              : "La siguiente lectura también tuvo una coincidencia baja. Revisa y ajusta la banda ahora; si una tercera verificación consecutiva sigue sin coincidir, la sesión finalizará.",
          });
        }
      } else if (
        event.status !== "REJECTED"
        && modalRef.current?.kind === "low-match-warning"
      ) {
        setModal(null);
      }
      return;
    }
    if (event.type === "error") {
      if (event.code === "CONNECTION_LOST" && flowRef.current === "exam") {
        recoveryTriggered.current = true;
        setFlow("recovery");
        setModal({
          kind: "recheck-signal",
          title: "Se perdió la conexión",
          message: "Revisa que la banda esté puesta y que el Bluetooth siga activo. Después podremos volver a comprobar la señal.",
        });
      } else {
        setModal({ kind: "error", title: "Problema con la banda", message: event.message });
      }
    }
  }, [showError, startMonitoring]);
  eventHandlerRef.current = handleEvent;

  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark");
    localStorage.setItem("latido-theme", theme);
  }, [theme]);

  useEffect(() => {
    let active = true;
    let retry;
    let healthRetry;
    const checkHealth = () => {
      request("/health")
        .then((health) => active && setApiStatus(health))
        .catch(() => active && setApiStatus({ ok: false }));
    };
    const connect = () => {
      if (!active) return;
      const socket = openEvents(
        (event) => eventHandlerRef.current?.(event),
        (connected) => active && setSocketConnected(connected),
      );
      socketRef.current = socket;
      socket.onclose = () => {
        if (!active) return;
        setSocketConnected(false);
        retry = window.setTimeout(connect, 1800);
      };
    };
    checkHealth();
    healthRetry = window.setInterval(checkHealth, 3000);
    connect();
    return () => {
      active = false;
      window.clearTimeout(retry);
      window.clearInterval(healthRetry);
      socketRef.current?.close();
    };
  }, []);

  useEffect(() => {
    if (flow !== "exam") return undefined;
    const interval = window.setInterval(() => setElapsed((seconds) => seconds + 1), 1000);
    return () => window.clearInterval(interval);
  }, [flow]);

  async function beginEnrollment() {
    if (!apiStatus?.ok || !socketConnected) {
      showError("Backend no disponible", new Error("Inicia primero el servidor de autenticación y comprueba su conexión."));
      return;
    }
    setEnrollProgress(0);
    setEnrollStage("settling");
    setEnrolled(false);
    recoveryTriggered.current = false;
    setModal(null);
    setFlow("enroll");
    try {
      await request("/enroll/start", {
        method: "POST",
        body: JSON.stringify({ user_id: userId }),
      });
    } catch (error) {
      setFlow("welcome");
      showError("No se pudo iniciar el registro", error);
    }
  }

  async function beginExam() {
    try {
      const params = new URLSearchParams({
        level,
        difficulty,
        count: String(questionCount),
      });
      const data = await request(`/exam/questions?${params}`);
      if (!data.questions.length) throw new Error("No hay preguntas para esa combinación.");
      setQuestions(data.questions);
      metricReadingsRef.current = [];
      metricsCollectionActiveRef.current = true;
      examTerminatingRef.current = false;
      wearerLabelRef.current = "unknown";
      setWearerLabel("unknown");
      setAnswers({});
      setCurrentIndex(0);
      setElapsed(0);
      setLockedMessage("");
      await startMonitoring();
    } catch (error) {
      showError("No se pudo preparar el examen", error);
    }
  }

  async function retryRecheck() {
    setRecheckProgress({ stage: "settling", elapsed_s: 0, total_s: 3 });
    setModal({
      kind: "rechecking",
      title: "Tomando una nueva muestra",
      message: "Mantente quieto mientras comparamos la señal con la huella inicial.",
    });
    try {
      await request("/monitor/recheck", {
        method: "POST",
        body: JSON.stringify({ user_id: userId }),
      });
    } catch (error) {
      setRecheckProgress(null);
      showError("No se pudo iniciar la revalidación", error);
    }
  }

  async function finishExam() {
    if (examTerminatingRef.current) return;
    examTerminatingRef.current = true;
    const failures = [];
    try {
      await request("/stop", { method: "POST" });
    } catch (error) {
      failures.push(`No se pudo detener la vigilancia: ${errorText(error)}`);
    } finally {
      metricsCollectionActiveRef.current = false;
    }
    try {
      await request("/exam/metrics", {
        method: "POST",
        body: JSON.stringify({
          outcome: "completed",
          readings: metricReadingsRef.current,
        }),
      });
    } catch (error) {
      failures.push(`No se pudieron guardar las métricas: ${errorText(error)}`);
    }
    try {
      const grade = await request("/exam/grade", {
        method: "POST",
        body: JSON.stringify({
          answers: questions.map((question) => ({
            question_id: question.id,
            answer: answers[question.id] || null,
          })),
        }),
      });
      setResult(grade);
      setModal(null);
      setFlow("results");
      if (failures.length) showError("El examen terminó con avisos", new Error(failures.join(" ")));
    } catch (error) {
      const message = failures.length
        ? `${errorText(error)} ${failures.join(" ")}`
        : errorText(error);
      showError("No se pudo cerrar el examen", new Error(message));
    }
  }

  async function terminateExamInterrupted(message) {
    if (examTerminatingRef.current) return;
    examTerminatingRef.current = true;
    setFlow("locked");
    setLockedMessage(message);
    setModal(null);

    let stopError = null;
    try {
      await request("/stop", { method: "POST" });
    } catch (error) {
      stopError = error;
    } finally {
      metricsCollectionActiveRef.current = false;
    }
    let metricsError = null;
    try {
      await request("/exam/metrics", {
        method: "POST",
        body: JSON.stringify({
          outcome: "interrupted",
          readings: metricReadingsRef.current,
        }),
      });
    } catch (error) {
      metricsError = error;
    }
    if (metricsError) {
      setLockedMessage((current) => `${current} No se pudieron guardar las métricas: ${errorText(metricsError)}`);
    }
    if (stopError) showError("No se pudo detener la vigilancia", stopError);
  }

  async function abandonExam() {
    const failures = [];
    if (metricsCollectionActiveRef.current && !examTerminatingRef.current) {
      examTerminatingRef.current = true;
      try {
        await request("/stop", { method: "POST" });
      } catch (error) {
        failures.push(`No se pudo detener la vigilancia: ${errorText(error)}`);
      } finally {
        metricsCollectionActiveRef.current = false;
      }
      try {
        await request("/exam/metrics", {
          method: "POST",
          body: JSON.stringify({
            outcome: "interrupted",
            readings: metricReadingsRef.current,
          }),
        });
      } catch (error) {
        failures.push(`No se pudieron guardar las métricas: ${errorText(error)}`);
      }
    } else {
      try {
        await request("/stop", { method: "POST" });
      } catch (error) {
        failures.push(`No se pudo detener la vigilancia: ${errorText(error)}`);
      }
    }
    setFlow("welcome");
    setDecision(null);
    setModal(null);
    if (failures.length) showError("No se pudo completar el cierre", new Error(failures.join(" ")));
  }

  function goHome() {
    setFlow("welcome");
    setEnrolled(false);
    setDecision(null);
    setResult(null);
    setModal(null);
  }

  function nextQuestion() {
    if (currentIndex < questions.length - 1) setCurrentIndex((index) => index + 1);
  }

  function previousQuestion() {
    if (currentIndex > 0) setCurrentIndex((index) => index - 1);
  }

  const busy = ["connecting", "enrolling", "monitoring", "rechecking"].includes(backendPhase);
  const examInProgress = flow === "exam";

  return (
    <div className="app-shell min-h-screen">
      <AppHeader
        apiStatus={apiStatus}
        socketConnected={socketConnected}
        theme={theme}
        onToggleTheme={() => setTheme((current) => current === "dark" ? "light" : "dark")}
      />
      <main className={`main-content ${examInProgress ? "exam-main" : ""}`}>
        {flow === "welcome" && (
          <WelcomeScreen
            level={level}
            difficulty={difficulty}
            questionCount={questionCount}
            authMode={authMode}
            apiAvailable={apiStatus?.ok}
            onLevelChange={setLevel}
            onDifficultyChange={setDifficulty}
            onQuestionCountChange={setQuestionCount}
            onAuthModeChange={setAuthMode}
            onBegin={() => setFlow("enroll")}
          />
        )}
        {flow === "enroll" && (
          <EnrollmentScreen
            userId={userId}
            enrolled={enrolled}
            backendPhase={backendPhase}
            enrollStage={enrollStage}
            enrollProgress={enrollProgress}
            apiAvailable={apiStatus?.ok}
            socketConnected={socketConnected}
            busy={busy}
            onStartEnrollment={() => void beginEnrollment()}
            onBeginExam={() => void beginExam()}
            onBack={() => {
              void request("/stop", { method: "POST" }).catch(() => {});
              setFlow("welcome");
              setModal(null);
            }}
          />
        )}
        {flow === "ready" && (
          <ReadyScreen
            userId={userId}
            questionCount={questionCount}
            busy={busy}
            onBeginExam={() => void beginExam()}
          />
        )}
        {flow === "exam" && questions[currentIndex] && (
          <ExamScreen
            questions={questions}
            answers={answers}
            currentIndex={currentIndex}
            decision={decision}
            authMode={authMode}
            wearerLabel={wearerLabel}
            onConfirmWearer={(label) => {
              if (!["original", "impostor", "unknown"].includes(label)) {
                showError("Etiqueta de participante inválida", new Error("Selecciona original, impostor o no confirmado."));
                return;
              }
              wearerLabelRef.current = label;
              setWearerLabel(label);
            }}
            signalBad={signalBad}
            elapsed={elapsed}
            onAnswer={(questionId, answer) => setAnswers((current) => ({ ...current, [questionId]: answer }))}
            onPrevious={previousQuestion}
            onNext={nextQuestion}
            onFinishRequest={() => setModal({
              kind: "finish",
              title: "Terminar simulación",
              message: `Has respondido ${Object.keys(answers).length} de ${questions.length} preguntas. ¿Quieres finalizar y ver tu resultado?`,
            })}
            onExitRequest={() => setModal({
              kind: "abandon",
              title: "¿Salir del examen?",
              message: "Tu sesión se detendrá y no se guardará una calificación.",
            })}
          />
        )}
        {flow === "recovery" && (
          <RecoveryScreen
            recheckProgress={recheckProgress}
            busy={busy}
            onRetry={() => void retryRecheck()}
            onAbandon={() => void abandonExam()}
          />
        )}
        {flow === "locked" && (
          <LockedScreen message={lockedMessage} onHome={() => {
            setModal(null);
            setFlow("welcome");
            setEnrolled(false);
            setLockedMessage("");
          }} />
        )}
        {flow === "results" && result && <ResultsScreen result={result} onHome={goHome} />}
      </main>
      <AppFooter />
      <SessionModal
        modal={modal}
        onClose={() => setModal(null)}
        onContinue={() => {
          setModal(null);
          setFlow("ready");
        }}
        onFinish={() => void finishExam()}
        onAbandon={() => void abandonExam()}
      />
    </div>
  );
}
