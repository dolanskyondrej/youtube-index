#!/usr/bin/env bash
# Nasazení ytindex na Google Cloud: Cloud Run (web + API + MCP), Cloud Run Job (indexace), Cloud SQL Postgres (pgvector).
# Předpoklad: `gcloud auth login` + nastavený projekt. Spouštět z kořene repa. Idempotentní (dá se pouštět opakovaně).
#
#   GEMINI_API_KEY=... CHANNEL="https://youtube.com/@kanal" ./deploy/deploy.sh
#
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${REGION:-europe-west1}"
SERVICE="${SERVICE:-ytindex}"
SQL_INSTANCE="${SQL_INSTANCE:-ytindex-pg}"
SQL_TIER="${SQL_TIER:-db-f1-micro}"          # nejlevnější (~8 USD/měs.); pro 1 500 h videa stačí
DB_NAME="${DB_NAME:-ytindex}"
DB_USER="${DB_USER:-ytindex}"
CHANNEL="${CHANNEL:?Nastav CHANNEL (handle/ID/URL kanálu)}"
GEMINI_API_KEY="${GEMINI_API_KEY:?Nastav GEMINI_API_KEY}"
YOUTUBE_API_KEY="${YOUTUBE_API_KEY:-}"
REPO="ytindex"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE}:$(git rev-parse --short HEAD 2>/dev/null || date +%s)"

echo "▶ Projekt ${PROJECT_ID}, region ${REGION}"
gcloud config set project "$PROJECT_ID" >/dev/null

echo "▶ Zapínám API"
gcloud services enable run.googleapis.com sqladmin.googleapis.com artifactregistry.googleapis.com \
  cloudbuild.googleapis.com secretmanager.googleapis.com generativelanguage.googleapis.com \
  youtube.googleapis.com --quiet

echo "▶ Artifact Registry"
gcloud artifacts repositories describe "$REPO" --location "$REGION" >/dev/null 2>&1 || \
  gcloud artifacts repositories create "$REPO" --repository-format docker --location "$REGION" --quiet

echo "▶ Cloud SQL (PostgreSQL 16 + pgvector)"
if ! gcloud sql instances describe "$SQL_INSTANCE" >/dev/null 2>&1; then
  gcloud sql instances create "$SQL_INSTANCE" --database-version POSTGRES_16 --tier "$SQL_TIER" \
    --region "$REGION" --edition ENTERPRISE --storage-size 10GB --storage-auto-increase --quiet
fi
DB_PASS="$(gcloud secrets versions access latest --secret ytindex-db-pass 2>/dev/null || true)"
if [ -z "$DB_PASS" ]; then
  DB_PASS="$(openssl rand -base64 24 | tr -d '/+=' )"
  printf '%s' "$DB_PASS" | gcloud secrets create ytindex-db-pass --data-file=- --quiet
fi
gcloud sql users describe "$DB_USER" --instance "$SQL_INSTANCE" >/dev/null 2>&1 || \
  gcloud sql users create "$DB_USER" --instance "$SQL_INSTANCE" --password "$DB_PASS" --quiet
gcloud sql databases describe "$DB_NAME" --instance "$SQL_INSTANCE" >/dev/null 2>&1 || \
  gcloud sql databases create "$DB_NAME" --instance "$SQL_INSTANCE" --quiet
CONN="$(gcloud sql instances describe "$SQL_INSTANCE" --format 'value(connectionName)')"
# vector + unaccent jsou na Cloud SQL povolená rozšíření; CREATE EXTENSION dělá aplikace při startu (uživatel musí mít práva)
gcloud sql users set-password "$DB_USER" --instance "$SQL_INSTANCE" --password "$DB_PASS" --quiet >/dev/null

echo "▶ Secrets"
printf '%s' "$GEMINI_API_KEY" | (gcloud secrets create ytindex-gemini-key --data-file=- --quiet 2>/dev/null || \
  gcloud secrets versions add ytindex-gemini-key --data-file=- --quiet)
DATABASE_URL="postgresql://${DB_USER}:${DB_PASS}@/${DB_NAME}?host=/cloudsql/${CONN}"
printf '%s' "$DATABASE_URL" | (gcloud secrets create ytindex-database-url --data-file=- --quiet 2>/dev/null || \
  gcloud secrets versions add ytindex-database-url --data-file=- --quiet)
PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format 'value(projectNumber)')"
SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
for s in ytindex-gemini-key ytindex-database-url; do
  gcloud secrets add-iam-policy-binding "$s" --member "serviceAccount:${SA}" --role roles/secretmanager.secretAccessor --quiet >/dev/null
done
gcloud projects add-iam-policy-binding "$PROJECT_ID" --member "serviceAccount:${SA}" --role roles/cloudsql.client --quiet >/dev/null

echo "▶ Build image (Cloud Build)"
gcloud builds submit --tag "$IMAGE" --quiet .

ENV_VARS="YTINDEX_CHANNEL=${CHANNEL},YTINDEX_CONFIG=/app/ytindex.toml"
[ -n "$YOUTUBE_API_KEY" ] && ENV_VARS="${ENV_VARS},YOUTUBE_API_KEY=${YOUTUBE_API_KEY}"
SECRETS="GEMINI_API_KEY=ytindex-gemini-key:latest,DATABASE_URL=ytindex-database-url:latest"

echo "▶ Cloud Run Job: indexace (init + ingest + embed)"
if gcloud run jobs describe "${SERVICE}-ingest" --region "$REGION" >/dev/null 2>&1; then
  gcloud run jobs update "${SERVICE}-ingest" --region "$REGION" --image "$IMAGE" \
    --set-env-vars "$ENV_VARS" --set-secrets "$SECRETS" --set-cloudsql-instances "$CONN" \
    --task-timeout 24h --memory 1Gi --cpu 1 --max-retries 1 --command ytindex --args cloud-ingest --quiet
else
  gcloud run jobs create "${SERVICE}-ingest" --region "$REGION" --image "$IMAGE" \
    --set-env-vars "$ENV_VARS" --set-secrets "$SECRETS" --set-cloudsql-instances "$CONN" \
    --task-timeout 24h --memory 1Gi --cpu 1 --max-retries 1 --command ytindex --args cloud-ingest --quiet
fi

echo "▶ Cloud Run service: web + API + MCP"
gcloud run deploy "$SERVICE" --region "$REGION" --image "$IMAGE" --allow-unauthenticated \
  --set-env-vars "$ENV_VARS" --set-secrets "$SECRETS" --add-cloudsql-instances "$CONN" \
  --memory 512Mi --cpu 1 --min-instances 0 --max-instances 3 --port 8000 \
  --command ytindex --args cloud-serve --quiet

URL="$(gcloud run services describe "$SERVICE" --region "$REGION" --format 'value(status.url)')"
cat <<MSG

✔ Hotovo.
  Web + API:   ${URL}          (OpenAPI: ${URL}/docs)
  MCP server:  ${URL}/mcp      (Streamable HTTP – přidej do Claude jako vzdálený MCP)

Indexace (test na 10 videích):
  gcloud run jobs execute ${SERVICE}-ingest --region ${REGION} --args cloud-ingest,--limit,10 --wait
Celý kanál:
  gcloud run jobs execute ${SERVICE}-ingest --region ${REGION} --wait
Pravidelný doběh nových videí (každý den v 04:00):
  gcloud scheduler jobs create http ${SERVICE}-ingest-daily --location ${REGION} --schedule "0 4 * * *" \\
    --uri "https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${PROJECT_ID}/jobs/${SERVICE}-ingest:run" \\
    --http-method POST --oauth-service-account-email ${SA}
MSG
