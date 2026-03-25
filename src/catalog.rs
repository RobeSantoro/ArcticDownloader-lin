use crate::{
    config::{default_catalog_endpoint, ConfigStore},
    env_flags::prefer_local_catalog,
    model::{LoraDefinition, ModelCatalog, ModelVariant, ResolvedModel, WorkflowDefinition},
    vram::VramTier,
};
use anyhow::{Context, Result};
use log::{info, warn};
use reqwest::{header, Client, StatusCode};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::{Arc, RwLock},
    time::Duration,
};

const BUNDLED_CATALOG: &str = include_str!("../data/catalog.json");
const CACHED_CATALOG_FILE: &str = "catalog.json";

#[derive(Debug)]
pub struct CatalogService {
    catalog: RwLock<ModelCatalog>,
    config: Arc<ConfigStore>,
}

impl CatalogService {
    pub fn new(config: Arc<ConfigStore>) -> Result<Self> {
        let catalog = if prefer_local_catalog() {
            resolve_catalog()
                .or_else(|| load_cached_catalog(&config))
                .unwrap_or_else(|| {
                    serde_json::from_str(BUNDLED_CATALOG).expect("valid bundled JSON")
                })
        } else {
            load_cached_catalog(&config)
                .or_else(resolve_catalog)
                .unwrap_or_else(|| {
                    serde_json::from_str(BUNDLED_CATALOG).expect("valid bundled JSON")
                })
        };
        info!(
            "Catalog initialised with {} models ({} LoRAs, {} workflows).",
            catalog.models.len(),
            catalog.loras.len(),
            catalog.workflows.len()
        );
        Ok(Self {
            catalog: RwLock::new(catalog),
            config,
        })
    }

    pub fn catalog_snapshot(&self) -> ModelCatalog {
        self.catalog.read().expect("catalog poisoned").clone()
    }

    pub fn variants_for_tier(&self, model_id: &str, tier: VramTier) -> Vec<ModelVariant> {
        let catalog = self.catalog_snapshot();
        catalog
            .models
            .into_iter()
            .find(|m| m.id == model_id)
            .map(|master| master.variants_for_tier(tier))
            .unwrap_or_default()
    }

    pub fn resolve_variant(&self, model_id: &str, variant_id: &str) -> Option<ResolvedModel> {
        let catalog = self.catalog_snapshot();
        let master = catalog.models.into_iter().find(|m| m.id == model_id)?;
        let variant = master
            .variants
            .iter()
            .find(|variant| variant.id == variant_id)?
            .clone();
        Some(ResolvedModel { master, variant })
    }

    pub fn loras(&self) -> Vec<LoraDefinition> {
        self.catalog_snapshot().loras
    }

    pub fn lora_families(&self) -> Vec<String> {
        self.catalog_snapshot().lora_families()
    }

    pub fn find_lora(&self, id: &str) -> Option<LoraDefinition> {
        self.catalog_snapshot().find_lora(id)
    }

    pub fn workflows(&self) -> Vec<WorkflowDefinition> {
        self.catalog_snapshot().workflows
    }

    pub fn workflow_families(&self) -> Vec<String> {
        self.catalog_snapshot().workflow_families()
    }

    pub fn find_workflow(&self, id: &str) -> Option<WorkflowDefinition> {
        self.catalog_snapshot().find_workflow(id)
    }

    pub async fn refresh_from_remote(&self) -> Result<bool> {
        let settings = self.config.settings();
        let endpoint = settings
            .catalog_endpoint
            .clone()
            .or_else(default_catalog_endpoint);

        let Some(url) = endpoint else {
            info!("No remote catalog endpoint configured; using bundled data.");
            return Ok(false);
        };

        let client = Client::builder()
            .user_agent(format!(
                "ArcticDownloader/{} ({})",
                env!("CARGO_PKG_VERSION"),
                env!("CARGO_PKG_NAME")
            ))
            .timeout(Duration::from_secs(10))
            .build()
            .context("failed to build HTTP client for catalog refresh")?;

        let mut request = client.get(&url);
        if let Some(etag) = settings
            .last_catalog_etag
            .as_deref()
            .filter(|etag| !etag.is_empty())
        {
            request = request.header(header::IF_NONE_MATCH, etag);
        }

        info!("Refreshing catalog from {url}");
        let mut response = request
            .send()
            .await
            .with_context(|| format!("failed to fetch remote catalog from {url}"))?;

        match response.status() {
            StatusCode::NOT_MODIFIED => {
                // Migration safety: older app versions could cache a lossy catalog
                // missing newly added sections/fields (e.g. `workflows`, `youtube_url`).
                // If we detect
                // that case, force one full fetch without ETag.
                if !cached_catalog_contains_key(&self.cached_catalog_path(), "workflows")
                    || !cached_catalog_contains_key(&self.cached_catalog_path(), "youtube_url")
                {
                    info!(
                        "Catalog cache is missing required workflow keys; forcing a full remote fetch."
                    );
                    response = client
                        .get(&url)
                        .send()
                        .await
                        .with_context(|| format!("failed forced catalog fetch from {url}"))?;
                    if response.status() != StatusCode::OK {
                        warn!(
                            "Forced catalog fetch skipped: server returned {} ({:?})",
                            response.status().as_u16(),
                            response.status()
                        );
                        return Ok(false);
                    }
                } else {
                    info!("Remote catalog is up to date (HTTP 304).");
                    return Ok(false);
                }
            }
            StatusCode::OK => {}
            status => {
                warn!(
                    "Catalog refresh skipped: server returned {} ({:?})",
                    status.as_u16(),
                    status
                );
                return Ok(false);
            }
        }

        let etag = response
            .headers()
            .get(header::ETAG)
            .and_then(|value| value.to_str().ok())
            .map(|value| value.to_string());

        let bytes = response
            .bytes()
            .await
            .context("failed to read remote catalog body")?;
        let catalog: ModelCatalog = serde_json::from_slice(&bytes)
            .with_context(|| format!("failed to parse remote catalog JSON from {url}"))?;

        self.persist_catalog(&catalog)?;

        {
            let mut guard = self.catalog.write().expect("catalog poisoned for write");
            *guard = catalog;
        }

        self.config.update_settings(|settings| {
            settings.last_catalog_etag = etag.clone();
        })?;

        info!("Catalog updated from remote source.");
        Ok(true)
    }

    fn persist_catalog(&self, catalog: &ModelCatalog) -> Result<()> {
        let path = self.cached_catalog_path();
        let data = serde_json::to_vec_pretty(catalog)?;
        fs::write(&path, data)
            .with_context(|| format!("failed to write cached catalog to {path:?}"))?;
        Ok(())
    }

    fn cached_catalog_path(&self) -> PathBuf {
        self.config.cache_path().join(CACHED_CATALOG_FILE)
    }
}

fn resolve_catalog() -> Option<ModelCatalog> {
    for path in catalog_candidate_paths() {
        if let Some(catalog) = load_catalog_from_path(&path) {
            return Some(catalog);
        }
    }
    warn!("Falling back to bundled catalog data.");
    None
}

fn load_catalog_from_path(path: &Path) -> Option<ModelCatalog> {
    match fs::read_to_string(path) {
        Ok(contents) => match serde_json::from_str::<ModelCatalog>(&contents) {
            Ok(parsed) => {
                info!("Loaded catalog from {:?}", path);
                Some(parsed)
            }
            Err(err) => {
                warn!("Failed to parse catalog at {:?}: {err}", path);
                None
            }
        },
        Err(err) => {
            warn!("Failed to read catalog at {:?}: {err}", path);
            None
        }
    }
}

fn catalog_candidate_paths() -> Vec<PathBuf> {
    let mut candidates = Vec::new();

    if let Ok(custom) = std::env::var("ARCTIC_CATALOG_PATH") {
        candidates.push(PathBuf::from(custom));
    }

    if let Ok(cwd) = std::env::current_dir() {
        candidates.push(cwd.join("data/catalog.json"));
    }

    if let Some(exe_dir) = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(PathBuf::from))
    {
        candidates.push(exe_dir.join("data/catalog.json"));
        if let Some(parent) = exe_dir.parent() {
            candidates.push(parent.join("data/catalog.json"));
            if let Some(grand) = parent.parent() {
                candidates.push(grand.join("data/catalog.json"));
            }
        }
    }

    candidates.retain(|p| p.exists());
    candidates
}

fn load_cached_catalog(config: &ConfigStore) -> Option<ModelCatalog> {
    let path = cached_catalog_path(config);
    if path.exists() {
        load_catalog_from_path(&path)
    } else {
        None
    }
}

fn cached_catalog_path(config: &ConfigStore) -> PathBuf {
    config.cache_path().join(CACHED_CATALOG_FILE)
}

fn cached_catalog_contains_key(path: &Path, key: &str) -> bool {
    let Ok(text) = fs::read_to_string(path) else {
        return false;
    };
    let needle = format!("\"{key}\"");
    text.contains(&needle)
}
