#!/usr/bin/env bash
# Levné nasazení ytindex na Google Cloud (odhad < 1 USD/měsíc):
#   Cloud Run service (web + API + MCP, škáluje na 0) + Cloud Run Job (indexace) + SQLite index v Cloud Storage.
# Žádný databázový server. Předpoklad: přihlášený gcloud (uživatel nebo servisní účet s rolí Owner/Editor).
#
#   GEMINI_API_KEY=... CHANNEL="https://youtube.com/@kanal" ./deploy/deploy.sh
#
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${REGION:-europe-west1}"
SERVICE="${SERVICE:-ytindex}"
BUCKET="${BUCKET:-${PROJECT_ID}-ytindex}"
CHANNEL="${CHANNEL:?Nastav CHANNEL (handle/ID/URL kanálu)}"
GEMINI_API_KEY="${GEMINI_API_KEY:-}"           # povinný jen při prvním nasazení; ROTATE_GEMINI_KEY=1 uloží novou verzi
YOUTUBE_API_KEY="${YOUTUBE_API_KEY:-}"
YTINDEX_LIMIT="${YTINDEX_LIMIT:-}"          # např. 10 = indexovat jen prvních N videí (test)
MEMORY="${MEMORY:-1Gi}"                      # celý kanál (~100k chunků s vektory) → 2Gi
REPO="ytindex"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE}:$(git rev-parse --short HEAD 2>/dev/null || date +%s)"
GCS_DB="gs://${BUCKET}/index.sqlite"

echo "▶ Projekt ${PROJECT_ID}, region ${REGION}"
gcloud config set project "$PROJECT_ID" >/dev/null

echo "▶ Zapínám API"
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com \
  secretmanager.googleapis.com storage.googleapis.com generativelanguage.googleapis.com --quiet

echo "▶ Bucket ${GCS_DB}"
gcloud storage buckets describe "gs://${BUCKET}" >/dev/null 2>&1 || \
  gcloud storage buckets create "gs://${BUCKET}" --location "$REGION" --uniform-bucket-level-access --quiet

echo "▶ Artifact Registry"
gcloud artifacts repositories describe "$REPO" --location "$REGION" >/dev/null 2>&1 || \
  gcloud artifacts repositories create "$REPO" --repository-format docker --location "$REGION" --quiet

echo "▶ Secret s Gemini klíčem"
if ! gcloud secrets describe ytindex-gemini-key >/dev/null 2>&1 && [ -z "$GEMINI_API_KEY" ]; then
  echo "Chybí GEMINI_API_KEY (secret ještě neexistuje)." >&2; exit 2
fi
if gcloud secrets describe ytindex-gemini-key >/dev/null 2>&1; then
  if [ "${ROTATE_GEMINI_KEY:-0}" = "1" ]; then
    printf '%s' "$GEMINI_API_KEY" | gcloud secrets versions add ytindex-gemini-key --data-file=- --quiet
  fi
else
  printf '%s' "$GEMINI_API_KEY" | gcloud secrets create ytindex-gemini-key --data-file=- --quiet
fi
PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format 'value(projectNumber)')"
SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
gcloud secrets add-iam-policy-binding ytindex-gemini-key --member "serviceAccount:${SA}" \
  --role roles/secretmanager.secretAccessor --quiet >/dev/null
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" --member "serviceAccount:${SA}" \
  --role roles/storage.objectAdmin --quiet >/dev/null

echo "▶ Build image (Cloud Build)"
gcloud builds submit --tag "$IMAGE" --quiet .

ENV_VARS="YTINDEX_CHANNEL=${CHANNEL},YTINDEX_GCS_DB=${GCS_DB}"
[ -n "$YOUTUBE_API_KEY" ] && ENV_VARS="${ENV_VARS},YOUTUBE_API_KEY=${YOUTUBE_API_KEY}"
[ -n "$YTINDEX_LIMIT" ] && ENV_VARS="${ENV_VARS},YTINDEX_LIMIT=${YTINDEX_LIMIT}"
SECRETS="GEMINI_API_KEY=ytindex-gemini-key:latest"

echo "▶ Cloud Run Job: indexace"
JOB_ARGS=(--region "$REGION" --image "$IMAGE" --set-env-vars "$ENV_VARS" --set-secrets "$SECRETS"
          --task-timeout 24h --memory "$MEMORY" --cpu 1 --max-retries 0 --command ytindex --args cloud-ingest --quiet)
if gcloud run jobs describe "${SERVICE}-ingest" --region "$REGION" >/dev/null 2>&1; then
  gcloud run jobs update "${SERVICE}-ingest" "${JOB_ARGS[@]}"
else
  gcloud run jobs create "${SERVICE}-ingest" "${JOB_ARGS[@]}"
fi

echo "▶ Cloud Run service: web + API + MCP"
gcloud run deploy "$SERVICE" --region "$REGION" --image "$IMAGE" --allow-unauthenticated \
  --set-env-vars "$ENV_VARS" --set-secrets "$SECRETS" \
  --memory "$MEMORY" --cpu 1 --min-instances 0 --max-instances 2 --concurrency 40 --port 8000 \
  --command ytindex --args cloud-serve --quiet

URL="$(gcloud run services describe "$SERVICE" --region "$REGION" --format 'value(status.url)')"
cat <<MSG

✔ Hotovo.
  Web + API:   ${URL}          (OpenAPI: ${URL}/docs)
  MCP server:  ${URL}/mcp      (Streamable HTTP – přidej do Claude jako vzdálený MCP)
  Index:       ${GCS_DB}

Indexace v cloudu (Gemini):   gcloud run jobs execute ${SERVICE}-ingest --region ${REGION} --wait
Nahrání lokálního indexu:     gcloud storage cp data/index.sqlite ${GCS_DB}
Nové instance služby si po nahrání stáhnou aktuální index (služba škáluje na 0; vynutit lze
  gcloud run services update ${SERVICE} --region ${REGION} --update-env-vars INDEX_VERSION=\$(date +%s)).
MSG
