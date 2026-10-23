# Nasazení bez lokálního gcloud (service account)

Pokud nechceš spouštět `deploy/deploy.sh` ze svého počítače, vytvoř servisní účet a jeho klíč předej do prostředí
(Claude Code → Environment → Secrets jako `GOOGLE_APPLICATION_CREDENTIALS_JSON`), nikdy ne do repa.

```bash
PROJECT=youtube-index-512512
gcloud iam service-accounts create ytindex-deployer --project $PROJECT
for role in roles/run.admin roles/cloudsql.admin roles/artifactregistry.admin roles/cloudbuild.builds.editor \
            roles/secretmanager.admin roles/serviceusage.serviceUsageAdmin roles/iam.serviceAccountUser \
            roles/resourcemanager.projectIamAdmin roles/storage.admin; do
  gcloud projects add-iam-policy-binding $PROJECT --member serviceAccount:ytindex-deployer@$PROJECT.iam.gserviceaccount.com --role $role --quiet >/dev/null
done
gcloud iam service-accounts keys create ytindex-deployer.json --iam-account ytindex-deployer@$PROJECT.iam.gserviceaccount.com
```

V Cloud Console totéž: IAM → Service accounts → Create → přidat role výše → Keys → Add key (JSON).

Použití: `gcloud auth activate-service-account --key-file ytindex-deployer.json && ./deploy/deploy.sh`.
Po nasazení klíč smaž (`gcloud iam service-accounts keys delete …`).
