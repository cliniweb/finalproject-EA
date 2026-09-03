"""Cliente de la API de Cliniweb — totalmente asíncrono, sin llamadas bloqueantes.

Este módulo es el ÚNICO lugar del código que habla con Cliniweb.
Según el documento oficial de integración (tecnica.pdf) solo se permiten
exactamente tres APIs de SOLO LECTURA de la *plataforma de pacientes*:

    1. GET /api/doctores                       → search_doctors_by_text()
    2. GET /api/perfiles-publicos/{nickname}   → fetch_doctor()
    3. GET /api/citas/horarios/disponibles     → fetch_available_slots()

La API real de agendamiento aún no existe — build_booking_url() apunta a un
host FALSO deliberado (ver CLINIWEB_BOOKING_BASE) hasta que Cliniweb la publique.

Toda petición debe llevar una API Key en los headers HTTP (también según el
documento); ver _auth_headers().
"""

from __future__ import annotations

import urllib.parse

import httpx
import structlog

from app.config import get_settings
from app.services.console import say

log = structlog.get_logger()


def _auth_headers() -> dict:
    """Header de API Key requerido en toda petición a Cliniweb (según el doc).

    El NOMBRE del header es configurable (CLINIWEB_API_KEY_HEADER, por defecto
    "X-Api-Key") porque Cliniweb envía la especificación exacta del header
    junto con la credencial por un canal seguro aparte. Si no hay clave
    configurada (p. ej. tests locales) no se envía ningún header de
    autenticación en lugar de enviar uno vacío.
    """
    settings = get_settings()
    if settings.CLINIWEB_API_KEY:
        return {settings.CLINIWEB_API_KEY_HEADER: settings.CLINIWEB_API_KEY}
    return {}


async def fetch_doctor(doctor_id: str) -> dict:
    """Descarga y devuelve el perfil público completo del médico desde Cliniweb.

    API #2 del documento de integración.

    ``doctor_id`` es el *nickname* del perfil (p. ej. "doctoracuddy") — se
    obtiene del campo `url` de los resultados de búsqueda quitando la barra
    diagonal inicial (ver simplify_search_results). `nombreCuenta` debe ser
    siempre la cuenta configurada (minimed-administracion).

    La respuesta contiene el arreglo `localidades` con los IDs dinámicos que
    necesita la API de horarios — se extraen con extract_localidades().
    """
    settings = get_settings()
    url = (
        f"{settings.CLINIWEB_API_BASE}/perfiles-publicos/{doctor_id}"
        f"?lenguaje={settings.CLINIWEB_LANGUAGE}&nombreCuenta={settings.CLINIWEB_ACCOUNT}"
    )
    log.info("cliniweb_fetch_doctor", url=url)
    say(f"🌐 Cliniweb API #2: descargando perfil público de '{doctor_id}'…")
    async with httpx.AsyncClient(timeout=15, headers=_auth_headers()) as client:
        resp = await client.get(url)
        log.debug(
            "cliniweb_fetch_doctor_response",
            status=resp.status_code,
            elapsed_ms=int(resp.elapsed.total_seconds() * 1000),
            content_length=len(resp.content),
        )
        say(
            f"🌐 Cliniweb API #2: respuesta {resp.status_code} en "
            f"{int(resp.elapsed.total_seconds() * 1000)} ms ({len(resp.content)} bytes)"
        )
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict):
            # La API responde el cuerpo literal `null` (200 OK) cuando el perfil
            # no existe o no pertenece a la cuenta configurada.
            raise ValueError(
                f"perfil '{doctor_id}' no encontrado en la cuenta "
                f"'{settings.CLINIWEB_ACCOUNT}' (respuesta null)"
            )
        return data


async def search_profiles_by_concept(tipo_concepto: str, concepto: str) -> list[dict]:
    """Busca perfiles públicos de doctores por concepto clínico.

    El buscador /api/doctores devuelve *conceptos clínicos* (especialidad,
    diagnóstico, procedimiento...) cuyo campo `url` tiene la forma
    "/{tipoConcepto}/{concepto}" (p. ej. "/especialidad/medicina-general").
    Los doctores reales se obtienen resolviendo ese concepto contra
    GET /api/perfiles-publicos/{tipoConcepto}/{concepto}, que devuelve un
    objeto con el arreglo `perfiles` (idPersona, nombre, sexo, especialidad,
    rutaNavegacion = nickname del perfil público).

    Nota: `seed` es un parámetro OBLIGATORIO de la ruta (solo ordena los
    resultados); sin él la API responde 404.
    """
    settings = get_settings()
    params = {
        "seed": "1",                                # obligatorio; solo afecta el orden
        "lenguaje": settings.CLINIWEB_LANGUAGE,
        "nombreCuenta": settings.CLINIWEB_ACCOUNT,  # OJO: dominioEmpresa se ignora aquí
                                                    # y devuelve doctores fuera de la cuenta
        "tipoPerfil": 1,                            # Doctor = 1
    }
    url = (
        f"{settings.CLINIWEB_API_BASE}/perfiles-publicos/"
        f"{urllib.parse.quote(tipo_concepto)}/{urllib.parse.quote(concepto)}"
        f"?{urllib.parse.urlencode(params)}"
    )
    log.info("cliniweb_search_profiles_by_concept", url=url)
    say(f"🌐 Cliniweb API #2b: buscando perfiles por concepto '{tipo_concepto}/{concepto}'…")
    async with httpx.AsyncClient(timeout=15, headers=_auth_headers()) as client:
        resp = await client.get(url)
        log.debug(
            "cliniweb_search_profiles_response",
            status=resp.status_code,
            elapsed_ms=int(resp.elapsed.total_seconds() * 1000),
            content_length=len(resp.content),
        )
        resp.raise_for_status()
        data = resp.json()
        # El endpoint SP-backed devuelve {nresul, medicos: [...], localidades: [...]}
        # (ResultadosDeBusquedaMicroservicioPerfilPublicoModel); las rutas
        # "esprofiles" devuelven {perfiles: [...]}. Aceptar ambas formas.
        perfiles = None
        if isinstance(data, dict):
            perfiles = data.get("medicos") or data.get("perfiles")
        perfiles = perfiles if isinstance(perfiles, list) else []
        say(f"🌐 Cliniweb API #2b: {len(perfiles)} perfil(es) para '{tipo_concepto}/{concepto}'")
        return perfiles


def simplify_profile_results(perfiles: list[dict]) -> list[dict]:
    """Mapea los resultados de /perfiles-publicos/{tipo}/{concepto} al mismo
    formato de sugerencias que simplify_search_results.

    El endpoint devuelve `medicos` con campos abreviados (MedicoModel del
    gateway): nom=nombre, sx=sexo, nav=ruta de navegación (nickname),
    esp=especialidad, idpersona. Las rutas "esprofiles" usan los nombres
    largos (nombre, sexo, rutaNavegacion, especialidad, idPersona); se
    aceptan ambas formas. doctor_id = nickname ("/doctoracuddy" -> "doctoracuddy").
    """
    simplified = []
    for p in perfiles:
        nombre = p.get("nom") or p.get("nombre") or ""
        nav = p.get("nav") or p.get("rutaNavegacion") or ""
        nickname = nav.strip("/")
        if not nombre or not nickname or "/" in nickname:
            log.debug(
                "cliniweb_profile_result_dropped",
                nombre=nombre,
                nav=nav,
                idPersona=p.get("idpersona") or p.get("idPersona"),
                keys=sorted(p.keys()),
            )
            continue
        simplified.append(
            {
                "idpersona": str(p.get("idpersona") or p.get("idPersona") or ""),
                "doctor_id": nickname,
                "nombre": nombre,
                "especialidad": p.get("esp") or p.get("especialidad") or "",
                "sexo": p.get("sx") or p.get("sexo") or "",
            }
        )
    return simplified


def extract_concepts(resultados: list[dict]) -> list[dict]:
    """Extrae los conceptos clínicos resolubles de los resultados de /api/doctores.

    Devuelve dicts {"tipo": ..., "concepto": ..., "nombre": ..., "clase": ...}
    a partir de urls "/{tipoConcepto}/{concepto}", ordenados por prioridad de
    clase (especialidades primero — son las que mejor mapean a doctores).
    """
    _CLASS_PRIORITY = {"Especialidad": 0, "Área de enfoque": 1, "Diagnóstico": 2}
    concepts = []
    for m in resultados:
        clean_url = (m.get("url") or "").strip("/")
        parts = clean_url.split("/")
        if len(parts) == 2 and all(parts) and not int(m.get("idPersona") or 0):
            concepts.append(
                {
                    "tipo": parts[0],
                    "concepto": parts[1],
                    "nombre": m.get("nombre", ""),
                    "clase": m.get("clase", ""),
                }
            )
    concepts.sort(key=lambda c: _CLASS_PRIORITY.get(c["clase"], 9))
    return concepts


async def search_doctors_by_text(text: str) -> list[dict]:
    """Búsqueda de doctores en el endpoint oficial de pacientes (/api/doctores).

    API #1 del documento de integración.

    - ``textoBusqueda`` es flexible: acepta nombres de doctores, especialidades
      médicas o áreas de experiencia ("Cardiología", "Luisa Cuddy", ...).
    - ``tipoPerfil=1`` es fijo según el documento.
    - ``dominioEmpresa`` debe mantenerse siempre en la cuenta configurada.

    Devuelve el arreglo JSON crudo; usar simplify_search_results() para reducir
    cada resultado a los campos que el LLM/paciente realmente necesita.
    """
    settings = get_settings()
    params = {
        "localizacion": settings.CLINIWEB_LANGUAGE,   # idioma de la interfaz (es)
        "textoBusqueda": text,                        # texto libre: nombre/especialidad
        "tipoPerfil": 1,                              # fijo según el documento
        "dominioEmpresa": settings.CLINIWEB_ACCOUNT,  # siempre minimed-administracion
    }
    url = f"{settings.CLINIWEB_API_BASE}/doctores?{urllib.parse.urlencode(params)}"
    log.info("cliniweb_search_doctors", url=url)
    say(f"🌐 Cliniweb API #1: buscando doctores con texto '{text}'…")
    async with httpx.AsyncClient(timeout=15, headers=_auth_headers()) as client:
        resp = await client.get(url)
        log.debug(
            "cliniweb_search_doctors_response",
            status=resp.status_code,
            elapsed_ms=int(resp.elapsed.total_seconds() * 1000),
            content_length=len(resp.content),
        )
        resp.raise_for_status()
        data = resp.json()
        # El endpoint devuelve un arreglo JSON plano de profesionales; cualquier
        # otra cosa (objeto de error, null) se trata como "sin resultados".
        medicos = data if isinstance(data, list) else []
        log.debug("cliniweb_search_doctors_parsed", result_count=len(medicos))
        say(f"🌐 Cliniweb API #1: {len(medicos)} doctor(es) encontrado(s) para '{text}'")
        return medicos


def simplify_search_results(medicos: list[dict]) -> list[dict]:
    """Conserva solo los campos que el LLM/paciente necesita de /api/doctores.

    Cada resultado crudo trae ~20 campos (imágenes, prioridades, empresas
    relacionadas...); pasar todo eso al LLM desperdicia tokens y genera
    confusión.

    El campo `url` trae una barra diagonal inicial (p. ej. "/doctoracuddy")
    que debe eliminarse para obtener el nickname que usa el endpoint de
    perfil público (documento de integración, sección 3.1).

    El buscador también devuelve entradas que NO son doctores (p. ej.
    especialidades con url "/especialidad/medicina-general"); en el modelo del
    gateway (ConceptoClinicoModel) solo las personas traen idPersona > 0, ese
    es el discriminador fiable. El endpoint de perfil público responde 404
    para las entradas de concepto, por lo que se descartan.
    """
    def _is_doctor(m: dict) -> bool:
        clean_url = (m.get("url") or "").strip("/")
        try:
            id_persona = int(m.get("idPersona") or 0)
        except (TypeError, ValueError):
            id_persona = 0
        return (
            bool(m.get("nombre"))
            and bool(clean_url)
            and not clean_url.startswith("especialidad/")
            and id_persona > 0
        )

    simplified = []
    for m in medicos:
        if _is_doctor(m):
            simplified.append(
                {
                    "idpersona": str(m.get("idPersona", "")),      # id numérico de la persona
                    "doctor_id": (m.get("url") or "").strip("/"),  # "/doctoracuddy" -> "doctoracuddy"
                    "nombre": m.get("nombre", ""),                 # nombre para mostrar
                    "especialidad": m.get("descripcion", ""),      # texto de especialidad
                    "sexo": m.get("sexo", ""),                     # "F"/"M" -> prefijo Dra./Dr.
                }
            )
        else:
            # Detalle completo del descarte para poder diagnosticar respuestas inesperadas.
            log.debug(
                "cliniweb_search_result_dropped_detail",
                nombre=m.get("nombre"),
                url=m.get("url"),
                idPersona=m.get("idPersona"),
                clase=m.get("clase"),
                claseId=m.get("claseId"),
                descripcion=m.get("descripcion"),
                keys=sorted(m.keys()),
            )
    dropped = len(medicos) - len(simplified)
    if dropped:
        log.debug(
            "cliniweb_search_results_dropped",
            dropped=dropped,
            reason="missing nombre/url or non-doctor entry",
            total_raw=len(medicos),
        )
    return simplified


async def fetch_available_slots(
    date_start: str,
    date_end: str,
    empresa_id: str,
    responsable_id: str,
    localidad_id: str,
) -> list[dict]:
    """Consulta los horarios de cita disponibles para un rango de fechas y localidad.

    API #3 del documento de integración.

    Los tres IDs NUNCA deben ingresarse manualmente — se mapean de la
    respuesta del perfil público según la localidad que eligió el paciente:
        empresa_id     <- localidades[i].idEmpresa
        responsable_id <- localidades[i].localidad.idPersona
        localidad_id   <- localidades[i].localidad.id
    (ver extract_localidades y node_fetch_slots).

    fechaInicio/fechaFin son un rango inclusivo según el documento (su propio
    ejemplo abarca 2026-08-13..2026-08-19).
    """
    settings = get_settings()
    url = (
        f"{settings.CLINIWEB_API_BASE}/citas/horarios/disponibles"
        f"?idEmpresa={empresa_id}"
        f"&idResponsableServicio={responsable_id}"
        f"&fechaInicio={date_start}"
        f"&fechaFin={date_end}"
        f"&idLocalidad={localidad_id}"
    )
    log.info("cliniweb_fetch_slots", url=url)
    say(
        f"🌐 Cliniweb API #3: consultando horarios ({date_start}..{date_end}, "
        f"empresa={empresa_id}, localidad={localidad_id})…"
    )
    async with httpx.AsyncClient(timeout=15, headers=_auth_headers()) as client:
        resp = await client.get(url)
        log.debug(
            "cliniweb_fetch_slots_response",
            status=resp.status_code,
            elapsed_ms=int(resp.elapsed.total_seconds() * 1000),
            content_length=len(resp.content),
        )
        resp.raise_for_status()
        data = resp.json()
        # Arreglo JSON plano de turnos; cualquier otra cosa significa "sin disponibilidad".
        raw = data if isinstance(data, list) else []
        # La API devuelve entradas anidadas por localidad:
        #   {"idLocalidad": ..., "turnos": [{"fechaHoraInicio": ..., "duracionEnMinutos": ...}]}
        # Se aplanan a un slot por turno con claves uniformes para el resto del código.
        slots: list[dict] = []
        for entry in raw:
            turnos = entry.get("turnos") if isinstance(entry, dict) else None
            if isinstance(turnos, list):
                for turno in turnos:
                    inicio = turno.get("fechaHoraInicio")
                    if inicio:
                        slots.append(
                            {
                                "fechaHora": inicio,
                                "duracionEnMinutos": turno.get("duracionEnMinutos"),
                                "idLocalidad": entry.get("idLocalidad"),
                            }
                        )
            elif isinstance(entry, dict) and (entry.get("fechaHora") or entry.get("fecha")):
                slots.append(entry)  # formato plano legado
        log.debug("cliniweb_fetch_slots_parsed", slot_count=len(slots))
        if slots:
            say(f"🌐 Cliniweb API #3: {len(slots)} turno(s) disponible(s)")
        else:
            say("🌐 Cliniweb API #3: SIN disponibilidad en ese rango de fechas")
        return slots


def build_booking_url(
    patient_name: str,
    symptoms: str,
    email: str,
    slot_datetime: str,
    localidad_id: str,
) -> str:
    """Construye la URL de reserva de cita de Cliniweb.

    IMPORTANTE: la API real de agendamiento aún no está publicada.
    CLINIWEB_BOOKING_BASE apunta deliberadamente a un host FALSO
    (testers.cliniweb.com) para que jamás se pueda crear una cita real desde
    este entorno; run.ps1 y check_env.ps1 lo verifican. Solo el nodo `confirm`
    puede llamar esta función (registro de privilegios mínimos) y únicamente
    tras la confirmación HITL del paciente.
    """
    settings = get_settings()
    # Los parámetros de una letra replican el formato legado del enlace de reserva.
    params = {
        "u": patient_name,    # usuario / nombre del paciente
        "p": symptoms,        # padecimiento / motivo de consulta
        "x": email,           # correo de contacto
        "t": slot_datetime,   # turno elegido "yyyy-MM-dd HH:mm"
        "l": localidad_id,    # id de la sucursal/localidad
    }
    query = urllib.parse.urlencode(params)
    url = f"{settings.CLINIWEB_BOOKING_BASE}?{query}"
    log.info("cliniweb_booking_url", url=url)
    return url


def extract_localidades(doctor_data: dict) -> list[dict]:
    """Extrae del perfil los IDs por localidad que necesita la API de horarios.

    Según el documento de integración:
    - idEmpresa vive en la raíz de cada entrada de localidades[]
    - idPersona (el responsable del servicio) vive dentro de .localidad
    - id (la sucursal/localidad) vive dentro de .localidad
    Ejemplo de entrada del perfil (estructura real, datos ficticios):

        {
          "idEmpresa": 39965,              <- id de la empresa (nivel raíz)
          "localidad": {
            "idPersona": 115986,           <- responsable del servicio
            "id": 10420023,                <- id de la sucursal/localidad
            "nombre": "Videoconsulta"
          }
        }
    """
    return [
        {
            "id": str(loc["localidad"]["id"]),                  # idLocalidad para la API de horarios
            "nombre": loc["localidad"]["nombre"],               # se muestra al paciente
            "idEmpresa": str(loc.get("idEmpresa", "")),         # idEmpresa para la API de horarios
            "idPersona": str(loc["localidad"].get("idPersona", "")),  # idResponsableServicio
        }
        for loc in doctor_data.get("localidades", [])
        # Entradas sin id+nombre no se pueden ofrecer al paciente — se omiten.
        if loc.get("localidad", {}).get("id") and loc.get("localidad", {}).get("nombre")
    ]


def simplify_doctor_data(data: dict) -> dict:
    """Elimina campos internos/ruido antes de enviar el perfil al LLM.

    El perfil crudo es grande (fotos, píxeles de tracking, ids internos,
    discriminadores de tipo) — nada de eso ayuda al LLM a responder preguntas
    del paciente y encarece cada prompt. La limpieza es recursiva:

    - descarta valores vacíos (None, "", [], {})
    - descarta claves que empiezan/terminan con un token de ruido
      (id*, *foto, pixel*, tipo*)

    NOTA: los IDs de localidad necesarios para la reserva NO se toman de este
    dict simplificado — se preservan por separado con extract_localidades().
    """
    _NOISE = {"id", "foto", "pixel", "tipo"}

    def _clean(obj):
        if isinstance(obj, dict):
            return {
                k: _clean(v)
                for k, v in obj.items()
                if v not in (None, "", [], {})
                and not any(k.lower().startswith(n) or k.lower().endswith(n) for n in _NOISE)
            }
        if isinstance(obj, list):
            return [_clean(i) for i in obj if i not in (None, "", [], {})]
        return obj

    return _clean(data)
