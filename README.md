🎥 **Video explicativo del proyecto:** [Ver en YouTube](https://www.youtube.com/watch?v=eBG2DaoQMBk)

# CliniAI v2 — Chatbot de Citas Médicas

> **Proyecto Final — AI Engineering**
>
> Sistema de IA en producción para la búsqueda de médicos y el agendamiento de
> citas médicas sobre la plataforma **Cliniweb**, construido con arquitectura
> CAG + RAG + agentes, evaluación con métricas objetivas y despliegue
> dockerizado.
>
> **Integrantes:** _<completar nombres del equipo>_
>
> **Rama de entrega:** `finalproject-<iniciales>` (ej.: `finalproject-JRP`)

---

## 1. Dominio y problema que resuelve

**Dominio:** salud — agendamiento de citas médicas en Panamá sobre la API real
de Cliniweb (datos reales de médicos, especialidades, localidades y horarios).

**Problema:** agendar una cita hoy exige que el paciente sepa de antemano qué
especialista necesita, navegue perfiles, compare localidades y horarios, y
complete formularios. CliniAI v2 lo resuelve con una conversación en español:

- El paciente **describe sus síntomas** en lenguaje natural y el sistema
  sugiere médicos adecuados (enrutamiento, **no** diagnóstico ni triaje).
- Responde **preguntas sobre el médico** (formación, seguros, servicios)
  fundamentadas en su perfil real vía RAG.
- Guía el **embudo completo de reserva**: médico → localidad → fecha/hora →
  datos del paciente → confirmación con URL de reserva.

**Datos reales:** perfiles públicos de médicos, localidades y disponibilidad
consumidos en vivo desde la API de Cliniweb (`app/services/cliniweb.py`).
Por seguridad, la reserva final apunta siempre al host de pruebas
(`testers.cliniweb.com`).

---

## 2. Arquitectura del sistema

```
Paciente (web UI / API REST)
        │
        ▼
FastAPI  POST /chat  (app/api/chat.py)
        │
        ├── CAG: caché exacta + semántica (app/cag/) ── hit → respuesta inmediata
        │
        ▼
Supervisor (app/agents/supervisor.py) ── enruta cada turno con privilegios (privileges.py)
        │
        ▼
LangGraph StateGraph (app/graph/graph.py)
        │
        ├── intent_node          clasifica: reservar / info / saludo
        ├── suggest_doctor_node  síntomas → especialidad → búsqueda de médicos
        ├── doctor_info_node     Q&A fundamentado en el perfil (RAG)
        ├── location_node        selección de localidad (Instructor)
        ├── datetime_node        fechas en lenguaje natural (Instructor)
        ├── fetch_slots_node     disponibilidad real vía httpx async
        ├── collect_node         nombre / síntomas / email (Instructor)
        └── confirm_node         URL de reserva, cierre de sesión
        │
        ├── RAG (app/rag/): chunking → ingest → vector store → retriever → quality
        ├── LLM (app/services/llm.py): LiteLLM con modelo primario + fallback
        └── Sesiones (app/services/session_store.py): Redis con degradación a memoria
```

### Decisiones técnicas justificadas

| Decisión | Justificación |
|---|---|
| **LangGraph** para el flujo conversacional | Grafo de estados explícito y depurable; sustituye una máquina de estados frágil basada en strings centinela (v1). |
| **Instructor + Pydantic** para extracción | Salidas del LLM validadas por esquema (`DoctorChoice`, `SymptomExtraction`…), sin regex ni parseo manual. |
| **LiteLLM** con fallback de modelo | Resiliencia: si el modelo primario falla, se conmuta automáticamente al de respaldo. |
| **CAG en dos niveles** (exacta + semántica) | Consultas repetidas o parecidas no pagan latencia ni coste de LLM. |
| **RAG por médico bajo demanda** | Cada perfil se ingesta al ser seleccionado: el corpus siempre está fresco y acotado al contexto de la sesión. |
| **httpx async** en toda la capa de I/O | Sin bloqueo del event loop de FastAPI bajo concurrencia. |
| **Redis con degradación a memoria** | Las sesiones sobreviven reinicios; si Redis cae, el sistema sigue funcionando. |
| **Supervisor con privilegios** | Un agente orquestador decide el nodo por turno; los privilegios acotan qué puede hacer cada uno (seguridad por diseño). |
| **structlog + Logfire** | Observabilidad estructurada y trazas de cada turno en producción. |

---

## 3. Componentes

### CAG — `app/cag/`
- `exact.py`: caché exacta por normalización de la consulta.
- `semantic.py`: caché semántica por similitud de embeddings; respuestas ya
  generadas se reutilizan para preguntas equivalentes.

### RAG — `app/rag/`
- `chunking.py`: troceado del perfil del médico en fragmentos semánticos.
- `ingest.py`: ingesta bajo demanda al seleccionar un médico
  (ver `_select_doctor` en `app/graph/node_suggest_doctor.py`).
- `store.py`: almacén vectorial en memoria (embeddings con numpy).
- `retriever.py`: recuperación top-k por similitud para `doctor_info_node`.
- `quality.py`: control de calidad del contexto recuperado (grounding).

### Agentes — `app/agents/`
- `supervisor.py`: agente supervisor que enruta cada turno al nodo adecuado
  del grafo según el estado de la conversación.
- `privileges.py`: matriz de privilegios por nodo — qué acciones puede
  ejecutar cada agente (defensa frente a desvíos del LLM).

### Evaluación — `evals/`
Métricas objetivas, no intuición:
- `eval_intent.py`: precisión de la clasificación de intención.
- `eval_dates.py`: exactitud del parseo de fechas en lenguaje natural.
- `eval_grounding.py`: fidelidad de las respuestas al contexto recuperado
  (anti-alucinación).
- `run_all.py`: ejecuta la suite completa y genera informes en
  `evals/reports/`.

```powershell
.venv\Scripts\python -m evals.run_all
```

### Despliegue — `docker/`
- `Dockerfile`: build multi-etapa (uv → imagen slim), usuario no-root,
  dependencias **congeladas** con `uv.lock` (`uv sync --frozen`), healthcheck
  contra `/health`, **cero secretos en la imagen**.
- `docker-compose.yml`: app + Redis; las claves se inyectan solo en runtime.
- Detalles y garantías de seguridad: [`docker/README.md`](docker/README.md).

---

## 4. Cómo levantar el sistema

### Opción A — Docker (recomendada)

```powershell
cd docker
Copy-Item runtime.env.example runtime.env   # rellenar OPENAI_API_KEY y CLINIWEB_API_KEY
docker compose up --build
```

- UI: http://localhost:8001/
- Salud: http://localhost:8001/health
- Docs OpenAPI: http://localhost:8001/docs

### Opción B — Local (desarrollo)

```powershell
uv sync                        # o: pip install -e .
Copy-Item .env.example .env    # rellenar OPENAI_API_KEY
.\run.ps1
```

### Ejemplo de petición

```json
POST http://localhost:8001/chat
{
  "session_id": "user-abc-123",
  "message": "Hola, tengo dolor de cabeza frecuente, ¿con quién puedo agendar?"
}
```

### Tests

```powershell
.venv\Scripts\python -m pytest
```

---

## 5. Limitaciones conocidas y próximos pasos

**Limitaciones:**
- El almacén vectorial es **en memoria**: el índice RAG no persiste entre
  reinicios (se re-ingesta al seleccionar médico, por lo que el impacto es
  bajo pero existe).
- El asistente **no realiza triaje ni diagnóstico**; solo enruta hacia
  especialidades disponibles en la cuenta configurada.
- La reserva final apunta deliberadamente al **entorno de pruebas** de
  Cliniweb (regla de seguridad del proyecto).
- Cobertura de especialidades limitada al catálogo de la cuenta
  (`_AVAILABLE_SPECIALTIES` — sin Medicina General; se usa Medicina Interna
  como fallback para adultos).
- Un solo idioma (español).

**Próximos pasos:**
- Vector store persistente (pgvector / Qdrant) y checkpointer de LangGraph
  sobre almacenamiento duradero.
- Ampliar la suite de evaluación (extracción de elección de médico,
  robustez ante adversarial prompts) e integrarla en CI.
- Streaming de respuestas (SSE) en la UI.
- Métricas de coste/latencia por turno exportadas a dashboards de Logfire.
- Soporte multi-idioma.

---

## Anexo: mejoras respecto a la v1

| v1 | v2 |
|---|---|
| Strings centinela + regex | Instructor + Pydantic (extracción estructurada) |
| LangChain v0 `LLMChain` | LiteLLM + enrutamiento con fallback |
| `requests` bloqueante | `httpx` asíncrono |
| Sin capa API | FastAPI `/chat` |
| Claves hardcodeadas | `pydantic-settings` + inyección en runtime |
| `print` / logger a fichero | `structlog` + Logfire |
| Sin tests ni evals | pytest + suite de evaluación con métricas |
