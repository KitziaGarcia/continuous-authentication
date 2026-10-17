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
  const [enrollProgress, setEnrollProgress] = useState(0);
  const [enrollStage, setEnrollStage] = useState("settling");
  const [enrolled, setEnrolled] = useState(false);
  const [questions, setQuestions] = useState([]);
  const [answers, setAnswers] = useState({});
  const [currentIndex, setCurrentIndex] = useState(0);
  const [decision, setDecision] = useState(null);
  const [signalBad, setSignalBad] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [recheckProgress, setRecheckProgress] = useState(null);
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
        setFlow("locked");
        setModal({
          kind: "locked",
          title: "Examen pausado",
          message: "La nueva muestra no coincidió con la huella inicial. La simulación se detuvo.",
        });
        void request("/stop", { method: "POST" }).catch(() => {});
      }
      return;
    }
    if (event.type === "decision") {
      setDecision(event);
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
      if (
        event.status === "REJECTED"
        && event.rejection_streak >= 3
        && flowRef.current === "exam"
        && !recoveryTriggered.current
      ) {
        recoveryTriggered.current = true;
        setFlow("recovery");
        setModal({
          kind: "recovery",
          title: "Confirma tu señal",
          message: "No pudimos confirmar la huella en tres lecturas seguidas. Ajusta la banda y permanece quieto; tomaremos una nueva muestra antes de continuar.",
        });
        void request("/stop", { method: "POST" }).catch((error) => {
          showError("No se pudo pausar la vigilancia", error);
        });
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
      setAnswers({});
      setCurrentIndex(0);
      setElapsed(0);
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
    try {
      await request("/stop", { method: "POST" });
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
    } catch (error) {
      showError("No se pudo cerrar el examen", error);
    }
  }

  async function abandonExam() {
    await request("/stop", { method: "POST" }).catch(() => {});
    setFlow("welcome");
    setDecision(null);
    setModal(null);
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
            apiAvailable={apiStatus?.ok}
            onLevelChange={setLevel}
            onDifficultyChange={setDifficulty}
            onQuestionCountChange={setQuestionCount}
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
          <LockedScreen onHome={() => {
            setModal(null);
            setFlow("welcome");
            setEnrolled(false);
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
