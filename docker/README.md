🎥 **Video explicativo del proyecto:** [Ver en YouTube](https://www.youtube.com/watch?v=eBG2DaoQMBk)

# CliniAI v2 — Despliegue con Docker

Todo lo necesario para ejecutar CliniAI v2 en contenedores. **No se almacenan
secretos en esta carpeta, en el repositorio ni en la imagen construida** — la
clave de OpenAI se inyecta exclusivamente en tiempo de ejecución.

## Contenido

| Archivo | Propósito |
|---|---|
| `Dockerfile` | Build multi-etapa (builder con uv → runtime slim sin root) |
| `docker-compose.yml` | App + Redis, claves inyectadas desde el entorno del host / `runtime.env` |
| `runtime.env.example` | Plantilla para secretos locales (copiar a `runtime.env`, ignorado por git) |

## Inicio rápido

Desde esta carpeta `docker/`:

```powershell
# Opción A: clave desde la shell del host
$env:OPENAI_API_KEY = "sk-proj-..."
docker compose up --build

# Opción B: clave desde un archivo env local
Copy-Item runtime.env.example runtime.env
# ...editar runtime.env y completar OPENAI_API_KEY / CLINIWEB_API_KEY...
docker compose up --build
```

Abrir http://localhost:8001/ — sonda de salud en `/health`.

## Garantías de seguridad

- El `Dockerfile` nunca usa `COPY .env`, `ARG` ni `ENV` para claves; compose
  falla de inmediato (sintaxis `:?`) si falta `OPENAI_API_KEY` al arrancar.
- `.dockerignore` (raíz del repo y contexto de build) excluye los archivos
  `.env*`, por lo que un archivo de claves extraviado nunca puede terminar en
  una capa de la imagen.
- `runtime.env` está ignorado por git; solo se versiona `runtime.env.example`
  (con valores vacíos).
- El contenedor se ejecuta con el usuario sin privilegios `cliniai`.
- `CLINIWEB_BOOKING_BASE` está fijado al host de pruebas falso — no se pueden
  crear citas reales desde este despliegue.

## Build independiente (sin compose)

```powershell
# Desde la raíz del proyecto:
docker build -f docker/Dockerfile -t cliniai_v2 .
docker run --rm -p 8001:8001 -e OPENAI_API_KEY=$env:OPENAI_API_KEY cliniai_v2
```
