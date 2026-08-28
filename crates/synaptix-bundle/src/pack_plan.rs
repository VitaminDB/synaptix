//! Распознавание модели на диске и план её упаковки в `.syn`.
//!
//! Модуль отвечает на два вопроса, которые раньше приходилось решать
//! пользователю руками в форме:
//!
//! * [`scan_collection`] — «что в этой папке вообще есть?»: готовые `.syn`
//!   и кандидаты на упаковку (каталоги моделей, одиночные `.safetensors`).
//! * [`scan_source`] — «из чего состоит эта модель и как её паковать?»:
//!   компоненты, вспомогательные файлы и догадки о метаданных.
//!
//! [`PackPlan`] — редактируемый результат: UI показывает его пользователю,
//! тот при желании правит, после чего [`PackPlan::into_builder`] превращает
//! план в [`BundleBuilder`]. `syn-pack` строит тот же план из аргументов
//! командной строки, поэтому CLI и GUI пакуют одинаково — расхождению
//! правил взяться неоткуда.
//!
//! Всё чтение здесь дешёвое: `fs::metadata`, `config.json` (килобайты) и
//! safetensors-заголовки через [`crate::inspect`]. Веса не читаются.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use crate::builder::BundleBuilder;
use crate::cdir::FileTag;
use crate::error::{Error, Result};
use crate::inspect;

/// Каталоги, которых не должно быть в бандле: кеш загрузчика HuggingFace
/// умеет весить гигабайты недокачанных шардов.
const SKIP_DIRS: &[&str] = &[".cache", ".git", "__pycache__", ".ipynb_checkpoints"];

/// Мусорные файлы и следы прерванных загрузок.
const SKIP_SUFFIXES: &[&str] =
    &[".incomplete", ".lock", ".metadata", ".swp", ".part", ".part.meta", ".download", ".tmp"];
const SKIP_NAMES: &[&str] = &[".DS_Store", ".gitattributes"];

/// Суффиксы «той же модели в другой точности». Такие наборы — альтернатива
/// основному, а не добавка к нему: включив оба, пользователь получил бы
/// бандл двойного размера с дублирующимися именами тензоров.
const PRECISION_VARIANTS: &[&str] = &[".fp16", ".fp32", ".bf16", ".fp8", ".f16", ".f32", ".int8"];

// ── Результат скана каталога ───────────────────────────────────────────────

/// Что нашлось в папке: либо уже собранный бандл, либо кандидат на упаковку.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FoundItem {
    /// Готовый `.syn`.
    Bundle { path: PathBuf, name: String, bytes: u64 },
    /// Модель, которую можно упаковать.
    Source(SourceCandidate),
}

impl FoundItem {
    pub fn path(&self) -> &Path {
        match self {
            FoundItem::Bundle { path, .. } => path,
            FoundItem::Source(c) => &c.path,
        }
    }
    pub fn name(&self) -> &str {
        match self {
            FoundItem::Bundle { name, .. } => name,
            FoundItem::Source(c) => &c.name,
        }
    }
    pub fn bytes(&self) -> u64 {
        match self {
            FoundItem::Bundle { bytes, .. } => *bytes,
            FoundItem::Source(c) => c.bytes,
        }
    }
}

/// Лёгкая карточка модели для списка: ровно то, что нужно нарисовать, без
/// построения полного плана.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceCandidate {
    /// Каталог модели или одиночный `.safetensors`.
    pub path: PathBuf,
    pub name: String,
    /// Сумма весов всех шардов.
    pub bytes: u64,
    pub shard_count: usize,
    pub component_count: usize,
    /// `model_type` из `config.json`, если он лежит на виду. Пусто —
    /// значит определится позже, при построении плана.
    pub arch: String,
}

/// Перечислить содержимое папки: `.syn`-пакеты и кандидаты на упаковку.
/// Спускается на один уровень — закладка на `~/models` показывает каждую
/// модель внутри, а не пустой список.
pub fn scan_collection(dir: &Path) -> Result<Vec<FoundItem>> {
    let mut out: Vec<FoundItem> = Vec::new();
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)?
        .flatten()
        .map(|e| e.path())
        .collect();
    entries.sort();

    // Сама папка — модель? Тогда она и есть единственный кандидат, а внутрь
    // спускаться нельзя: подкаталоги — её компоненты, а не соседние модели.
    if let Some(c) = candidate_for_dir(dir) {
        out.push(FoundItem::Source(c));
    }
    let self_is_model = !out.is_empty();

    for path in entries {
        if is_skipped_path(&path) {
            continue;
        }
        if path.is_file() {
            if has_ext(&path, "syn") {
                let bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                out.push(FoundItem::Bundle {
                    name: file_stem(&path),
                    path,
                    bytes,
                });
            } else if !self_is_model && has_ext(&path, "safetensors") {
                // Одиночный чекпойнт рядом с другими — самостоятельная модель.
                // Шарды одного набора (`*-00001-of-00012`) сюда не попадают:
                // их забирает `candidate_for_dir` выше.
                let bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                out.push(FoundItem::Source(SourceCandidate {
                    name: file_stem(&path),
                    path,
                    bytes,
                    shard_count: 1,
                    component_count: 1,
                    arch: String::new(),
                }));
            }
        } else if path.is_dir() && !self_is_model {
            if let Some(c) = candidate_for_dir(&path) {
                out.push(FoundItem::Source(c));
                continue;
            }
            // Раскладка HuggingFace — `<организация>/<модель>`: сами модели
            // лежат на два уровня вглубь (`~/models/Qwen/Qwen3.8-Flash-Next`).
            // Ещё глубже не идём: это уже не каталог моделей, а файловая
            // система вообще.
            let Ok(inner) = std::fs::read_dir(&path) else { continue };
            let mut children: Vec<PathBuf> = inner
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.is_dir() && !is_skipped_path(p))
                .collect();
            children.sort();
            for child in children {
                if let Some(c) = candidate_for_dir(&child) {
                    out.push(FoundItem::Source(c));
                }
            }
        }
    }
    Ok(out)
}

/// Кандидат для каталога, если внутри есть веса. `None` — обычная папка
/// или, наоборот, склад из нескольких самостоятельных моделей: такие
/// перечисляются по отдельности.
fn candidate_for_dir(dir: &Path) -> Option<SourceCandidate> {
    let groups = shard_groups_in(dir);
    let subs = sub_components(dir);
    if groups.is_empty() && subs.is_empty() {
        return None;
    }
    // Что связывает подкаталоги в одну модель: манифест пайплайна или общий
    // конфиг в корне. Без них `~/models` с десятком моделей внутри выглядел
    // бы одной гигантской моделью со странными компонентами.
    let bound_together =
        dir.join("model_index.json").exists() || dir.join("config.json").exists();
    if groups.is_empty() && !bound_together {
        return None;
    }
    // Несколько независимых наборов в корне и никакого манифеста — это
    // склад чекпойнтов (как `ltx2.3_v1.1`), а не одна модель.
    if groups.len() > 1 && !bound_together {
        return None;
    }
    let mut bytes = 0u64;
    let mut shard_count = 0usize;
    for g in groups.iter().chain(subs.iter()) {
        bytes += g.bytes;
        shard_count += g.paths.len();
    }
    let component_count = if groups.is_empty() { subs.len() } else { groups.len() + subs.len() };
    Some(SourceCandidate {
        name: file_name(dir),
        arch: read_config_model_type(dir).unwrap_or_default(),
        path: dir.to_path_buf(),
        bytes,
        shard_count,
        component_count: component_count.max(1),
    })
}

// ── План упаковки ──────────────────────────────────────────────────────────

/// Один tensors-чанк будущего бандла.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanComponent {
    /// Суффикс чанка: `main`, `transformer`, `video_vae`.
    pub name: String,
    /// Шарды в порядке склейки.
    pub paths: Vec<PathBuf>,
    /// Префикс имён тензоров; пусто — без пространства имён.
    pub prefix: String,
    pub bytes: u64,
    /// Снятый флаг — компонент в бандл не попадёт. Так помечаются, например,
    /// однофайловые дубликаты весов из подкаталогов diffusers-пайплайна.
    pub enabled: bool,
    /// Почему компонент выключен или чем он примечателен.
    pub note: String,
}

/// Вспомогательный файл (config.json, tokenizer.json, …).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanFile {
    /// Путь внутри бандла — относительный, с сохранением подкаталогов.
    pub rel: String,
    pub path: PathBuf,
    pub bytes: u64,
    pub tag: FileTag,
    pub enabled: bool,
}

/// Откуда взялась догадка — UI показывает это подписью у поля.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Guess {
    /// `config.json` → `model_type`.
    ConfigJson,
    /// `model_index.json` → `_class_name`.
    ModelIndex,
    /// `__metadata__` внутри safetensors.
    SafetensorsMeta,
    /// Имя файла или каталога.
    Name,
    /// Имена тензоров (например, LoRA по `lora_A`/`lora_B`).
    TensorNames,
    /// Не определилось — поле пустое, пусть заполняет человек.
    Unknown,
}

/// Метаданные будущего бандла с пометками, что откуда взялось.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GuessedMeta {
    pub id: String,
    pub version: String,
    pub arch: String,
    pub purpose: String,
    pub arch_from: Guess,
    pub purpose_from: Guess,
    pub version_from: Guess,
}

/// Раскладка источника — влияет только на подсказки в UI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceKind {
    /// Один `.safetensors`.
    SingleFile,
    /// Каталог с весами в корне (обычная раскладка HuggingFace).
    FlatDir,
    /// Каталог, где веса разложены по подпапкам (diffusers-пайплайн).
    MultiComponentDir,
}

/// Полный план: что паковать, во что и с какими метаданными.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackPlan {
    pub root: PathBuf,
    pub kind: SourceKind,
    pub components: Vec<PlanComponent>,
    pub aux: Vec<PlanFile>,
    pub meta: GuessedMeta,
    /// Что стоит показать пользователю до запуска упаковки.
    pub warnings: Vec<String>,
}

impl PackPlan {
    /// Разобрать каталог модели или одиночный `.safetensors`.
    pub fn scan(path: &Path) -> Result<PackPlan> {
        scan_source(path)
    }

    /// Суммарный payload включённых компонентов и файлов — по нему считается
    /// и требуемое место, и подпись прогресса.
    pub fn payload_bytes(&self) -> u64 {
        let t: u64 = self.components.iter().filter(|c| c.enabled).map(|c| c.bytes).sum();
        let f: u64 = self.aux.iter().filter(|f| f.enabled).map(|f| f.bytes).sum();
        t.saturating_add(f)
    }

    /// Самый крупный компонент: во время записи рядом с `out.syn.tmp` живёт
    /// промежуточный `tensors_stage*.tmp` такого размера.
    pub fn max_component_bytes(&self) -> u64 {
        self.components
            .iter()
            .filter(|c| c.enabled)
            .map(|c| c.bytes)
            .max()
            .unwrap_or(0)
    }

    /// Сколько места нужно на разделе с результатом: payload + пик стейджа
    /// + запас на cdir и выравнивание.
    pub fn required_space(&self) -> u64 {
        self.payload_bytes()
            .saturating_add(self.max_component_bytes())
            .saturating_add(64 << 20)
    }

    pub fn item_count(&self) -> usize {
        self.components.iter().filter(|c| c.enabled).count()
            + self.aux.iter().filter(|f| f.enabled).count()
    }

    /// Путь по умолчанию: `<dir>/<id>.syn`.
    pub fn suggested_out(&self, dir: &Path) -> PathBuf {
        dir.join(format!("{}.syn", self.meta.id))
    }

    /// Собрать [`BundleBuilder`] по плану. Выключенные элементы пропускаются.
    pub fn into_builder(self) -> Result<BundleBuilder> {
        let mut b = BundleBuilder::new(&self.meta.id, &self.meta.version);
        if !self.meta.arch.is_empty() {
            b = b.arch(&self.meta.arch);
        }
        if !self.meta.purpose.is_empty() {
            b = b.purpose(&self.meta.purpose);
        }
        let mut any = false;
        for c in self.components.iter().filter(|c| c.enabled) {
            if c.paths.is_empty() {
                return Err(Error::Safetensors(format!(
                    "компонент `{}`: не задано ни одного шарда",
                    c.name
                )));
            }
            any = true;
            let prefix = if c.prefix.is_empty() { None } else { Some(c.prefix.as_str()) };
            if let Some(p) = prefix {
                b = b.component(&c.name, p);
            }
            b = b.add_safetensors_component(&c.name, c.paths.clone(), prefix);
        }
        if !any {
            return Err(Error::Safetensors(
                "в плане нет ни одного включённого компонента".into(),
            ));
        }
        for f in self.aux.iter().filter(|f| f.enabled) {
            b = b.add_file_path(&f.rel, &f.path, f.tag)?;
        }
        Ok(b)
    }
}

/// Разобрать один источник — каталог модели или одиночный `.safetensors`.
pub fn scan_source(path: &Path) -> Result<PackPlan> {
    if path.is_file() {
        return scan_single_file(path);
    }
    if !path.is_dir() {
        return Err(Error::FileNotFound(path.display().to_string()));
    }
    scan_dir(path)
}

fn scan_single_file(path: &Path) -> Result<PackPlan> {
    if !has_ext(path, "safetensors") {
        return Err(Error::Safetensors(format!(
            "{}: ожидался каталог модели или .safetensors",
            path.display()
        )));
    }
    let bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    let components = vec![PlanComponent {
        name: "main".into(),
        paths: vec![path.to_path_buf()],
        prefix: String::new(),
        bytes,
        enabled: true,
        note: String::new(),
    }];
    // Соседние файлы не трогаем: рядом могут лежать чужие модели, и
    // сгрести их в бандл было бы сюрпризом.
    let meta = guess_meta(path, &file_stem(path), &components, &[]);
    Ok(PackPlan {
        root: path.to_path_buf(),
        kind: SourceKind::SingleFile,
        components,
        aux: Vec::new(),
        meta,
        warnings: Vec::new(),
    })
}

fn scan_dir(dir: &Path) -> Result<PackPlan> {
    let root_groups = shard_groups_in(dir);
    let subs = sub_components(dir);
    let has_model_index = dir.join("model_index.json").exists();
    let mut warnings: Vec<String> = Vec::new();

    if root_groups.is_empty() && subs.is_empty() {
        return Err(Error::Safetensors(format!(
            "в {} не найдено .safetensors",
            dir.display()
        )));
    }

    let mut components: Vec<PlanComponent> = Vec::new();
    let kind;
    let primary_root = root_groups.iter().filter(|g| !g.variant).count();
    if !subs.is_empty() && (has_model_index || root_groups.is_empty()) {
        // Пайплайн diffusers: веса живут в подпапках. Одиночные файлы в
        // корне — обычно вторая раскладка тех же весов; включать их значит
        // удвоить бандл, поэтому кладём выключенными.
        kind = SourceKind::MultiComponentDir;
        for g in subs {
            components.push(component_from_group(g, true, String::new()));
        }
        for g in root_groups {
            let note = "тот же чекпойнт одним файлом — дубликат весов из подкаталогов".to_string();
            components.push(component_from_group(g, false, note));
        }
    } else {
        kind = SourceKind::FlatDir;
        // Несколько самостоятельных наборов и никакого пайплайна — скорее
        // всего это склад чекпойнтов. Варианты точности сюда не считаются:
        // `model.safetensors` рядом с `model.fp16.safetensors` — одна модель.
        if primary_root > 1 {
            warnings.push(format!(
                "в каталоге {primary_root} независимых набора весов — проверьте состав, возможно это разные модели"
            ));
        }
        for g in root_groups {
            components.push(component_from_group(g, true, String::new()));
        }
        for g in subs {
            components.push(component_from_group(g, true, String::new()));
        }
    }

    sort_components(&mut components);
    let aux = aux_files(dir)?;
    let meta = guess_meta(dir, &file_name(dir), &components, &aux);
    Ok(PackPlan {
        root: dir.to_path_buf(),
        kind,
        components,
        aux,
        meta,
        warnings,
    })
}

/// Главный компонент — первым. Порядок важен не только глазу: по первому
/// включённому компоненту определяются архитектура и назначение бандла, и
/// брать их у CLIP-энкодера вместо трансформера было бы неверно.
fn sort_components(components: &mut [PlanComponent]) {
    components.sort_by_key(|c| {
        let rank = match c.name.as_str() {
            "main" => 0,
            "transformer" | "unet" | "dit" | "lm" => 1,
            _ => 2,
        };
        (!c.enabled, rank, c.name.clone())
    });
}

/// `enabled` — верхняя граница: вариант другой точности выключен всегда,
/// как бы его ни звал вызывающий.
fn component_from_group(g: ShardGroup, enabled: bool, note: String) -> PlanComponent {
    let (enabled, note) = if g.variant {
        (
            false,
            "та же модель в другой точности — включите вместо основного набора, если нужна именно она"
                .to_string(),
        )
    } else {
        (enabled, note)
    };
    PlanComponent {
        name: g.name,
        paths: g.paths,
        prefix: String::new(),
        bytes: g.bytes,
        enabled,
        note,
    }
}

// ── Поиск шардов ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
struct ShardGroup {
    name: String,
    paths: Vec<PathBuf>,
    bytes: u64,
    /// Набор той же модели в другой точности — по умолчанию не пакуется.
    variant: bool,
}

/// Наборы safetensors в одном каталоге (без рекурсии).
///
/// Порядок разбора: `*.safetensors.index.json` (HF-шардинг) → имена вида
/// `foo-00001-of-00012.safetensors` → одиночные файлы. Каждый набор — свой
/// компонент, поэтому папка с семью независимыми чекпойнтами LTX не
/// склеивается в один гигантский `tensors:main`.
fn shard_groups_in(dir: &Path) -> Vec<ShardGroup> {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut files: Vec<PathBuf> = Vec::new();
    let mut indexes: Vec<PathBuf> = Vec::new();
    for e in rd.flatten() {
        let p = e.path();
        if !p.is_file() || is_skipped_path(&p) {
            continue;
        }
        let name = file_name(&p);
        if name.ends_with(".safetensors.index.json") {
            indexes.push(p);
        } else if has_ext(&p, "safetensors") {
            files.push(p);
        }
    }
    files.sort();
    indexes.sort();

    let mut groups: Vec<ShardGroup> = Vec::new();
    let mut consumed: BTreeSet<PathBuf> = BTreeSet::new();

    for idx in &indexes {
        let stem = file_name(idx).trim_end_matches(".safetensors.index.json").to_string();
        let Some(paths) = shards_from_index(idx, dir) else { continue };
        let paths: Vec<PathBuf> = paths.into_iter().filter(|p| p.is_file()).collect();
        if paths.is_empty() {
            continue;
        }
        for p in &paths {
            consumed.insert(p.clone());
        }
        groups.push(ShardGroup {
            name: component_name_for(&stem),
            bytes: sum_bytes(&paths),
            paths,
            variant: is_precision_variant(&stem),
        });
    }

    // `foo-00001-of-00012.safetensors` без index.json.
    let mut by_prefix: BTreeMap<String, Vec<PathBuf>> = BTreeMap::new();
    let mut loose: Vec<PathBuf> = Vec::new();
    for p in files {
        if consumed.contains(&p) {
            continue;
        }
        match shard_prefix(&file_name(&p)) {
            Some(prefix) => by_prefix.entry(prefix).or_default().push(p),
            None => loose.push(p),
        }
    }
    for (prefix, mut paths) in by_prefix {
        paths.sort();
        groups.push(ShardGroup {
            name: component_name_for(&prefix),
            bytes: sum_bytes(&paths),
            paths,
            variant: is_precision_variant(&prefix),
        });
    }
    for p in loose {
        let stem = file_stem(&p);
        groups.push(ShardGroup {
            name: component_name_for(&stem),
            bytes: sum_bytes(std::slice::from_ref(&p)),
            paths: vec![p],
            variant: is_precision_variant(&stem),
        });
    }
    // Альтернативой можно быть только чему-то. Если в каталоге лежит один
    // `diffusion_pytorch_model.fp16.safetensors` и больше ничего — это и
    // есть модель (так распространяется fp16-репозиторий SDXL), а вовсе не
    // запасная точность, которую надо выключить.
    if !groups.is_empty() && groups.iter().all(|g| g.variant) {
        for g in &mut groups {
            g.variant = false;
        }
    }
    groups
}

/// Компоненты из подкаталогов. `video_vae/source/model.safetensors` даёт
/// компонент `video_vae`: спускаемся до папки с весами, но имя берём
/// верхнее — по нему модель и адресуется в загрузчиках.
fn sub_components(dir: &Path) -> Vec<ShardGroup> {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut subdirs: Vec<PathBuf> = rd
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir() && !is_skipped_path(p))
        .collect();
    subdirs.sort();

    let mut out: Vec<ShardGroup> = Vec::new();
    for sub in subdirs {
        let name = file_name(&sub);
        let mut groups = shard_groups_in(&sub);
        if groups.is_empty() {
            // Один уровень вглубь: `video_vae/source/`.
            let Ok(inner_rd) = std::fs::read_dir(&sub) else { continue };
            let mut inner: Vec<PathBuf> = inner_rd
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.is_dir() && !is_skipped_path(p))
                .collect();
            inner.sort();
            for i in inner {
                let g = shard_groups_in(&i);
                if !g.is_empty() {
                    groups = g;
                    break;
                }
            }
        }
        match groups.len() {
            0 => {}
            1 => {
                let mut g = groups.pop().unwrap();
                g.name = component_name_for(&name);
                out.push(g);
            }
            _ => {
                // Несколько наборов в одной подпапке. Обычный случай —
                // `vae/diffusion_pytorch_model.safetensors` рядом с
                // `…fp16.safetensors`: основной набор берёт имя папки, а
                // альтернативные точности остаются вариантами.
                let primary_count = groups.iter().filter(|g| !g.variant).count();
                for g in groups {
                    let inner = g.name.clone();
                    let takes_dir_name = primary_count == 1 && !g.variant;
                    let full = if takes_dir_name {
                        component_name_for(&name)
                    } else {
                        component_name_for(&format!("{name}_{inner}"))
                    };
                    out.push(ShardGroup { name: full, ..g });
                }
            }
        }
    }
    out
}

fn shards_from_index(index: &Path, dir: &Path) -> Option<Vec<PathBuf>> {
    let raw = std::fs::read(index).ok()?;
    let v: serde_json::Value = serde_json::from_slice(&raw).ok()?;
    let map = v.get("weight_map")?.as_object()?;
    let files: BTreeSet<PathBuf> = map
        .values()
        .filter_map(|f| f.as_str())
        .map(|n| dir.join(n))
        .collect();
    if files.is_empty() {
        None
    } else {
        Some(files.into_iter().collect())
    }
}

/// `model-00003-of-00012.safetensors` → `Some("model")`.
fn shard_prefix(name: &str) -> Option<String> {
    let stem = name.strip_suffix(".safetensors")?;
    let (head, tail) = stem.rsplit_once("-of-")?;
    if !tail.bytes().all(|b| b.is_ascii_digit()) || tail.is_empty() {
        return None;
    }
    let (prefix, num) = head.rsplit_once('-')?;
    if !num.bytes().all(|b| b.is_ascii_digit()) || num.is_empty() || prefix.is_empty() {
        return None;
    }
    Some(prefix.to_string())
}

/// Каноническое имя компонента: `model`/`diffusion_pytorch_model` — это
/// «основной» набор, он зовётся `main` (так его и ищут загрузчики).
fn component_name_for(raw: &str) -> String {
    match raw {
        "model" | "diffusion_pytorch_model" | "pytorch_model" => "main".to_string(),
        other => other.to_string(),
    }
}

fn sum_bytes(paths: &[PathBuf]) -> u64 {
    paths
        .iter()
        .filter_map(|p| std::fs::metadata(p).ok())
        .map(|m| m.len())
        .fold(0u64, |a, b| a.saturating_add(b))
}

// ── Вспомогательные файлы ──────────────────────────────────────────────────

/// Рекурсивный обход каталога: всё, кроме весов и мусора, попадает в бандл
/// файловыми чанками под своим относительным путём. Именно этого не хватало
/// GUI: без вложенных `transformer/config.json` и `tokenizer/*` раскладку
/// MiniMax-H3 нельзя собрать вообще.
pub fn aux_files(root: &Path) -> Result<Vec<PlanFile>> {
    let mut out: Vec<PlanFile> = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let rd = match std::fs::read_dir(&dir) {
            Ok(rd) => rd,
            Err(_) => continue,
        };
        for e in rd.flatten() {
            let p = e.path();
            if is_skipped_path(&p) {
                continue;
            }
            if p.is_dir() {
                stack.push(p);
                continue;
            }
            if !p.is_file() {
                continue;
            }
            let name = file_name(&p);
            if has_ext(&p, "safetensors") || name.ends_with(".safetensors.index.json") {
                continue;
            }
            let Ok(rel) = p.strip_prefix(root) else { continue };
            let rel = rel.to_string_lossy().replace('\\', "/");
            let bytes = std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
            out.push(PlanFile {
                tag: tag_for(&rel),
                rel,
                path: p,
                bytes,
                enabled: true,
            });
        }
    }
    out.sort_by(|a, b| a.rel.cmp(&b.rel));
    Ok(out)
}

/// Назначение файла внутри бандла. Загрузчики читают только `Inference`,
/// поэтому README и картинки не должны им мешать.
pub fn tag_for(rel: &str) -> FileTag {
    let lower = rel.to_ascii_lowercase();
    if lower.starts_with("examples/") || lower.contains("/examples/") {
        return FileTag::Example;
    }
    if lower.contains("readme") || lower.contains("license") || lower.ends_with(".md") {
        return FileTag::Doc;
    }
    let asset = [
        ".png", ".jpg", ".jpeg", ".webp", ".gif", ".mp4", ".mov", ".wav", ".mp3", ".flac",
    ];
    if asset.iter().any(|e| lower.ends_with(e)) {
        return FileTag::Asset;
    }
    FileTag::Inference
}

// ── Догадки о метаданных ───────────────────────────────────────────────────

/// Определить id/version/arch/purpose. Ничего не выдумывает: если признака
/// нет, поле остаётся пустым с пометкой [`Guess::Unknown`] — пусть лучше
/// человек впишет, чем бандл получит неверную архитектуру.
fn guess_meta(
    root: &Path,
    raw_name: &str,
    components: &[PlanComponent],
    aux: &[PlanFile],
) -> GuessedMeta {
    let (id, version, version_from) = split_name_version(raw_name);

    // Порядок источников: корневой `config.json` (обычная модель) →
    // `model_index.json` (пайплайн diffusers) → конфиг главного компонента →
    // `__metadata__` самих весов. Без второго шага у FLUX архитектурой
    // становился `clip_text_model` из `text_encoder/config.json` — то есть
    // энкодер вместо пайплайна.
    let root_config = read_root_json(root, "config.json");
    let model_index = read_root_json(root, "model_index.json");
    let primary = components.iter().find(|c| c.enabled).map(|c| c.name.as_str());
    let nested_config = read_component_json(aux, primary, "config.json");
    let config = root_config.clone().or_else(|| nested_config.clone());

    let mut arch = String::new();
    let mut arch_from = Guess::Unknown;
    if let Some(v) = root_config
        .as_ref()
        .and_then(|v| v.get("model_type"))
        .and_then(|v| v.as_str())
    {
        arch = v.to_string();
        arch_from = Guess::ConfigJson;
    } else if let Some(v) = model_index
        .as_ref()
        .and_then(|v| v.get("_class_name"))
        .and_then(|v| v.as_str())
    {
        arch = v.to_string();
        arch_from = Guess::ModelIndex;
    } else if let Some(v) = nested_config
        .as_ref()
        .and_then(|v| v.get("model_type"))
        .and_then(|v| v.as_str())
    {
        arch = v.to_string();
        arch_from = Guess::ConfigJson;
    } else if let Some(v) = arch_from_safetensors_meta(components) {
        arch = v;
        arch_from = Guess::SafetensorsMeta;
    }

    let names = first_component_tensor_names(components);
    let (purpose, purpose_from) = guess_purpose(&id, &arch, config.as_ref(), model_index.as_ref(), &names);

    GuessedMeta { id, version, arch, purpose, arch_from, purpose_from, version_from }
}

/// `ltx-2.3-22b-distilled-1.1` → id тот же, версия `1.1.0`. Хвост вида
/// `-1.1` / `-v2.0.1` — это версия сборки; `2.3` и `22b` внутри имени
/// версией не считаются, поэтому смотрим только на конец строки.
fn split_name_version(raw: &str) -> (String, String, Guess) {
    let id = normalize_id(raw);
    let Some((_, tail)) = id.rsplit_once('-') else {
        return (id, "1.0.0".to_string(), Guess::Unknown);
    };
    let t = tail.strip_prefix('v').unwrap_or(tail);
    let parts: Vec<&str> = t.split('.').collect();
    let numeric = parts.len() >= 2
        && parts.len() <= 3
        && parts.iter().all(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()));
    if !numeric {
        return (id, "1.0.0".to_string(), Guess::Unknown);
    }
    let mut v: Vec<String> = parts.iter().map(|s| s.to_string()).collect();
    while v.len() < 3 {
        v.push("0".to_string());
    }
    (id, v.join("."), Guess::Name)
}

/// Имя в стиле остальных бандлов пользователя: нижний регистр, пробелы и
/// слэши — в дефис.
pub fn normalize_id(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut prev_dash = false;
    for ch in raw.trim().chars() {
        let c = if ch.is_whitespace() || ch == '/' || ch == '\\' {
            '-'
        } else {
            ch.to_ascii_lowercase()
        };
        if c == '-' {
            if prev_dash {
                continue;
            }
            prev_dash = true;
        } else {
            prev_dash = false;
        }
        out.push(c);
    }
    out.trim_matches('-').to_string()
}

fn guess_purpose(
    id: &str,
    arch: &str,
    config: Option<&serde_json::Value>,
    model_index: Option<&serde_json::Value>,
    tensor_names: &[String],
) -> (String, Guess) {
    // LoRA виднее всего по самим тензорам — имя файла врёт чаще.
    if tensor_names
        .iter()
        .any(|n| n.contains("lora_A") || n.contains("lora_B") || n.contains("lora_down"))
    {
        return ("lora".to_string(), Guess::TensorNames);
    }

    // Манифест пайплайна описывает бандл целиком и потому сильнее, чем
    // `architectures` вложенного конфига: у FLUX там CLIPTextModel, и без
    // этой проверки картиночный пайплайн получал назначение «embed».
    if let Some(class) = model_index.and_then(|v| v.get("_class_name")).and_then(|v| v.as_str()) {
        let c = class.to_ascii_lowercase();
        if c.contains("video") {
            return ("video".to_string(), Guess::ModelIndex);
        }
        if c.contains("audio") || c.contains("music") {
            return ("music".to_string(), Guess::ModelIndex);
        }
        if c.contains("pipeline") {
            return ("image".to_string(), Guess::ModelIndex);
        }
    }

    let architectures: Vec<String> = config
        .and_then(|v| v.get("architectures"))
        .and_then(|a| a.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_str()).map(|s| s.to_string()).collect())
        .unwrap_or_default();
    let arch_str = architectures.join(" ");
    let has_vision = config.map(|v| v.get("vision_config").is_some()).unwrap_or(false);

    if !arch_str.is_empty() {
        if arch_str.contains("ForCTC")
            || arch_str.contains("ForSpeechSeq2Seq")
            || arch_str.contains("Whisper")
        {
            return ("asr".to_string(), Guess::ConfigJson);
        }
        if arch_str.contains("ForSequenceClassification") {
            return ("rerank".to_string(), Guess::ConfigJson);
        }
        if arch_str.contains("ForCausalLM") || arch_str.contains("ForConditionalGeneration") {
            let p = if has_vision { "multimodal_llm" } else { "text-generation" };
            return (p.to_string(), Guess::ConfigJson);
        }
        if arch_str.contains("EncoderModel") {
            return ("text-encoder".to_string(), Guess::ConfigJson);
        }
        if arch_str.ends_with("Model") {
            return ("embed".to_string(), Guess::ConfigJson);
        }
    }

    // Последний рубеж — имя. Тут мы уже не знаем ничего наверняка, поэтому
    // ловим только однозначные слова.
    let n = format!("{id} {arch}").to_ascii_lowercase();
    for (needle, purpose) in [
        ("upscaler", "upscaler"),
        ("lora", "lora"),
        ("vae", "vae"),
        ("rerank", "rerank"),
        ("embedding", "embed"),
        ("text-encoder", "text-encoder"),
        ("text_encoder", "text-encoder"),
    ] {
        if n.contains(needle) {
            return (purpose.to_string(), Guess::Name);
        }
    }
    (String::new(), Guess::Unknown)
}

/// `__metadata__["config"]` одиночного чекпойнта: так подписан LTX, у
/// которого `config.json` рядом нет вовсе.
fn arch_from_safetensors_meta(components: &[PlanComponent]) -> Option<String> {
    let first = components.iter().find(|c| c.enabled)?;
    let path = first.paths.first()?;
    let meta = inspect::read_metadata_file(path).ok()?;
    for key in ["model_type", "arch", "architecture"] {
        if let Some(v) = meta.get(key) {
            if !v.is_empty() {
                return Some(v.clone());
            }
        }
    }
    let cfg = meta.get("config")?;
    let v: serde_json::Value = serde_json::from_str(cfg).ok()?;
    for key in ["model_type", "_class_name"] {
        if let Some(s) = v.get(key).and_then(|x| x.as_str()) {
            if !s.is_empty() {
                return Some(s.to_string());
            }
        }
    }
    None
}

fn first_component_tensor_names(components: &[PlanComponent]) -> Vec<String> {
    let Some(first) = components.iter().find(|c| c.enabled) else {
        return Vec::new();
    };
    let Some(path) = first.paths.first() else {
        return Vec::new();
    };
    inspect::read_header_file(path)
        .map(|v| v.into_iter().map(|t| t.name).collect())
        .unwrap_or_default()
}

fn read_root_json(root: &Path, name: &str) -> Option<serde_json::Value> {
    let path = root.join(name);
    if !path.is_file() {
        return None;
    }
    let raw = std::fs::read(&path).ok()?;
    serde_json::from_slice(&raw).ok()
}

/// Конфиг главного компонента: `transformer/config.json` у пайплайна. Берём
/// именно его, а не первый попавшийся — иначе архитектурой модели окажется
/// архитектура её текстового энкодера.
fn read_component_json(
    aux: &[PlanFile],
    component: Option<&str>,
    name: &str,
) -> Option<serde_json::Value> {
    let want = component.map(|c| format!("{c}/{name}"));
    let nested = aux
        .iter()
        .find(|f| want.as_deref().is_some_and(|w| f.rel == w))
        .or_else(|| aux.iter().find(|f| f.rel.ends_with(&format!("/{name}"))))?;
    let raw = std::fs::read(&nested.path).ok()?;
    serde_json::from_slice(&raw).ok()
}

fn read_config_model_type(dir: &Path) -> Option<String> {
    let raw = std::fs::read(dir.join("config.json")).ok()?;
    let v: serde_json::Value = serde_json::from_slice(&raw).ok()?;
    v.get("model_type").and_then(|x| x.as_str()).map(str::to_string)
}

// ── Мелочи ─────────────────────────────────────────────────────────────────

fn is_skipped_path(p: &Path) -> bool {
    let Some(name) = p.file_name().and_then(|s| s.to_str()) else {
        return true;
    };
    if p.is_dir() {
        return SKIP_DIRS.contains(&name);
    }
    if SKIP_NAMES.contains(&name) || SKIP_SUFFIXES.iter().any(|s| name.ends_with(s)) {
        return true;
    }
    // Всё, что дописано после `.safetensors`, — след загрузчика
    // (`model-00008-of-00131.safetensors.part.meta`). Исключение —
    // штатный индекс шардов.
    if let Some(rest) = name.split_once(".safetensors.").map(|(_, r)| r) {
        return rest != "index.json";
    }
    false
}

/// Имя набора вида `diffusion_pytorch_model.fp16` — та же модель в другой
/// точности.
fn is_precision_variant(stem: &str) -> bool {
    let lower = stem.to_ascii_lowercase();
    PRECISION_VARIANTS.iter().any(|v| lower.ends_with(v) || lower.contains(&format!("{v}.")))
}

fn has_ext(p: &Path, ext: &str) -> bool {
    p.extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case(ext))
        .unwrap_or(false)
}

fn file_name(p: &Path) -> String {
    p.file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| p.display().to_string())
}

fn file_stem(p: &Path) -> String {
    p.file_stem()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| p.display().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stream::{safetensors_header, StDtype, StreamTensor};

    /// Минимальный валидный safetensors с одним тензором заданного имени.
    fn write_st(path: &Path, tensor: &str, bytes_per: usize) {
        let plan = vec![StreamTensor {
            name: tensor.to_string(),
            dtype: StDtype::U8,
            shape: vec![bytes_per],
        }];
        let mut buf = safetensors_header(&plan, 64).unwrap();
        buf.extend(std::iter::repeat(0u8).take(bytes_per));
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(path, buf).unwrap();
    }

    #[test]
    fn single_file_needs_no_staging_directory() {
        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("ltx-2.3-upscaler-1.1.safetensors");
        write_st(&f, "blocks.0.attn.qkv.weight", 128);

        let plan = PackPlan::scan(&f).unwrap();
        assert_eq!(plan.kind, SourceKind::SingleFile);
        assert_eq!(plan.components.len(), 1);
        assert_eq!(plan.components[0].name, "main");
        assert_eq!(plan.components[0].paths, vec![f]);
        // Версия вынута из хвоста имени, id остался узнаваемым.
        assert_eq!(plan.meta.id, "ltx-2.3-upscaler-1.1");
        assert_eq!(plan.meta.version, "1.1.0");
        // Соседей не подхватываем — их просто нет в плане.
        assert!(plan.aux.is_empty());
    }

    #[test]
    fn flat_hf_dir_gives_one_component_and_nested_aux() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("Qwen3.8-Flash-Next");
        write_st(&root.join("model-00001-of-00002.safetensors"), "model.embed_tokens.weight", 64);
        write_st(&root.join("model-00002-of-00002.safetensors"), "lm_head.weight", 64);
        std::fs::write(
            root.join("config.json"),
            br#"{"model_type":"qwen3_5","architectures":["Qwen3NextForCausalLM"]}"#,
        )
        .unwrap();
        std::fs::write(root.join("tokenizer.json"), b"{}").unwrap();
        std::fs::create_dir_all(root.join("chat")).unwrap();
        std::fs::write(root.join("chat/template.jinja"), b"x").unwrap();
        std::fs::write(root.join("README.md"), b"doc").unwrap();
        // Мусор загрузчика в бандл попасть не должен.
        std::fs::create_dir_all(root.join(".cache")).unwrap();
        std::fs::write(root.join(".cache/huge.incomplete"), b"garbage").unwrap();

        let plan = PackPlan::scan(&root).unwrap();
        assert_eq!(plan.kind, SourceKind::FlatDir);
        assert_eq!(plan.components.len(), 1);
        assert_eq!(plan.components[0].name, "main");
        assert_eq!(plan.components[0].paths.len(), 2, "оба шарда в одном компоненте");

        assert_eq!(plan.meta.arch, "qwen3_5");
        assert_eq!(plan.meta.arch_from, Guess::ConfigJson);
        assert_eq!(plan.meta.purpose, "text-generation");
        assert_eq!(plan.meta.id, "qwen3.8-flash-next");

        let rels: Vec<&str> = plan.aux.iter().map(|f| f.rel.as_str()).collect();
        assert!(rels.contains(&"config.json"));
        assert!(rels.contains(&"chat/template.jinja"), "вложенные пути сохраняются: {rels:?}");
        assert!(!rels.iter().any(|r| r.contains(".cache")), "кеш не пакуем: {rels:?}");
        let readme = plan.aux.iter().find(|f| f.rel == "README.md").unwrap();
        assert_eq!(readme.tag, FileTag::Doc);
    }

    #[test]
    fn diffusers_layout_splits_into_components_and_skips_duplicate_root_file() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("FLUX.1-dev");
        std::fs::write(root.join("model_index.json").as_path(), b"{}").ok();
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("model_index.json"), br#"{"_class_name":"FluxPipeline"}"#).unwrap();
        write_st(&root.join("flux1-dev.safetensors"), "single.weight", 512);
        write_st(&root.join("transformer/diffusion_pytorch_model.safetensors"), "blocks.0.attn.qkv.weight", 256);
        write_st(&root.join("vae/diffusion_pytorch_model.safetensors"), "decoder.conv_in.weight", 128);
        std::fs::write(root.join("transformer/config.json"), b"{}").unwrap();

        let plan = PackPlan::scan(&root).unwrap();
        assert_eq!(plan.kind, SourceKind::MultiComponentDir);
        let names: Vec<&str> = plan.components.iter().map(|c| c.name.as_str()).collect();
        assert!(names.contains(&"transformer"), "{names:?}");
        assert!(names.contains(&"vae"), "{names:?}");

        let root_file = plan
            .components
            .iter()
            .find(|c| c.name == "flux1-dev")
            .expect("одиночный файл в корне остаётся в плане");
        assert!(!root_file.enabled, "но выключенным — иначе бандл удвоится");
        assert!(!root_file.note.is_empty());

        // Размер считается только по включённому — выключенный дубликат
        // (самый крупный файл в раскладке) в него не входит.
        let enabled: u64 = plan.components.iter().filter(|c| c.enabled).map(|c| c.bytes).sum();
        let aux: u64 = plan.aux.iter().map(|f| f.bytes).sum();
        assert_eq!(plan.payload_bytes(), enabled + aux);
        assert!(root_file.bytes > 0);
        assert!(plan.payload_bytes() < enabled + aux + root_file.bytes);
        assert!(plan.aux.iter().any(|f| f.rel == "transformer/config.json"));
    }

    #[test]
    fn nested_component_dir_keeps_the_outer_name() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("FL2VA");
        write_st(&root.join("transformer/model.safetensors"), "blocks.0.attn.qkv.weight", 64);
        write_st(&root.join("video_vae/source/model.safetensors"), "decoder.conv_in.weight", 64);
        std::fs::write(root.join("model_index.json"), b"{}").unwrap();

        let plan = PackPlan::scan(&root).unwrap();
        let names: Vec<&str> = plan.components.iter().map(|c| c.name.as_str()).collect();
        assert!(names.contains(&"video_vae"), "имя берётся верхнее: {names:?}");
    }

    #[test]
    fn collection_lists_bundles_and_packable_models_side_by_side() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::write(root.join("qwen3.8-27b.syn"), b"stub").unwrap();
        write_st(&root.join("Qwen3.8-Flash-Next/model.safetensors"), "lm_head.weight", 64);
        std::fs::write(root.join("Qwen3.8-Flash-Next/config.json"), br#"{"model_type":"qwen3_5"}"#).unwrap();
        std::fs::create_dir_all(root.join("notes")).unwrap();
        std::fs::write(root.join("notes/todo.txt"), b"x").unwrap();

        let items = scan_collection(root).unwrap();
        let bundles: Vec<&FoundItem> = items.iter().filter(|i| matches!(i, FoundItem::Bundle { .. })).collect();
        let sources: Vec<&FoundItem> = items.iter().filter(|i| matches!(i, FoundItem::Source(_))).collect();
        assert_eq!(bundles.len(), 1);
        assert_eq!(bundles[0].name(), "qwen3.8-27b");
        assert_eq!(sources.len(), 1, "папка без весов кандидатом не считается");
        assert_eq!(sources[0].name(), "Qwen3.8-Flash-Next");
        if let FoundItem::Source(c) = sources[0] {
            assert_eq!(c.arch, "qwen3_5");
        }
    }

    #[test]
    fn checkpoint_folder_lists_each_file_separately() {
        // `ltx2.3_v1.1`: несколько независимых чекпойнтов в одной папке —
        // склеивать их в один бандл нельзя.
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("ltx2.3_v1.1");
        write_st(&root.join("ltx-2.3-22b-dev.safetensors"), "blocks.0.attn.qkv.weight", 64);
        write_st(&root.join("ltx-2.3-22b-distilled-1.1.safetensors"), "blocks.0.attn.qkv.weight", 64);
        write_st(&root.join("ltx-2.3-upscaler.safetensors"), "blocks.0.attn.qkv.weight", 64);

        let items = scan_collection(&root).unwrap();
        assert_eq!(items.len(), 3, "три отдельные модели: {items:?}");
        assert!(items.iter().all(|i| matches!(i, FoundItem::Source(_))));
        // И каждая пакуется сама по себе.
        let plan = PackPlan::scan(items[0].path()).unwrap();
        assert_eq!(plan.kind, SourceKind::SingleFile);
    }

    /// Раскладка SDXL: в `vae/` лежат обе точности — пакуем полную, fp16
    /// оставляем выключенным. В `unet/` только fp16 — и это сама модель,
    /// выключать нечего.
    #[test]
    fn precision_variants_are_dropped_only_when_there_is_an_alternative() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("sdxl");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("model_index.json"), br#"{"_class_name":"StableDiffusionXLPipeline"}"#).unwrap();
        write_st(&root.join("unet/diffusion_pytorch_model.fp16.safetensors"), "blocks.0.attn.qkv.weight", 256);
        write_st(&root.join("vae/diffusion_pytorch_model.safetensors"), "decoder.conv_in.weight", 128);
        write_st(&root.join("vae/diffusion_pytorch_model.fp16.safetensors"), "decoder.conv_in.weight", 64);

        let plan = PackPlan::scan(&root).unwrap();
        let by = |n: &str| plan.components.iter().find(|c| c.name == n).cloned();
        assert!(by("unet").expect("unet есть").enabled, "единственный набор — он и есть модель");
        assert!(by("vae").expect("vae есть").enabled);
        let fp16 = by("vae_diffusion_pytorch_model.fp16").expect("вариант виден в списке");
        assert!(!fp16.enabled, "дубликат другой точности выключен");
        assert!(fp16.note.contains("точности"));
        assert!(plan.warnings.is_empty(), "это не склад чекпойнтов: {:?}", plan.warnings);
    }

    /// Огрызки докачки HuggingFace (`*.safetensors.part.meta`) в бандл не
    /// попадают, а штатный индекс шардов — не огрызок.
    #[test]
    fn downloader_leftovers_never_reach_the_bundle() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("m");
        write_st(&root.join("model.safetensors"), "lm_head.weight", 64);
        std::fs::write(root.join("config.json"), br#"{"model_type":"qwen3"}"#).unwrap();
        std::fs::write(root.join("model-00008-of-00131.safetensors.part.meta"), b"x").unwrap();
        std::fs::write(root.join("model-00008-of-00131.safetensors.part"), b"x").unwrap();
        std::fs::write(root.join("shard.incomplete"), b"x").unwrap();

        let plan = PackPlan::scan(&root).unwrap();
        let rels: Vec<&str> = plan.aux.iter().map(|f| f.rel.as_str()).collect();
        assert_eq!(rels, vec!["config.json"], "лишнего не собрали: {rels:?}");
        assert!(!is_skipped_path(&root.join("model.safetensors.index.json")));
    }

    #[test]
    fn lora_is_recognised_by_tensor_names() {
        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("ic-lora-lipdub-0.9.safetensors");
        write_st(&f, "transformer_blocks.0.attn1.to_k.lora_A.weight", 64);
        let plan = PackPlan::scan(&f).unwrap();
        assert_eq!(plan.meta.purpose, "lora");
        assert_eq!(plan.meta.purpose_from, Guess::TensorNames);
        assert_eq!(plan.meta.version, "0.9.0");
    }

    #[test]
    fn plan_builds_a_bundle_end_to_end() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("tiny-model");
        write_st(&root.join("model.safetensors"), "model.embed_tokens.weight", 128);
        std::fs::write(root.join("config.json"), br#"{"model_type":"qwen3"}"#).unwrap();
        std::fs::write(root.join("nested/extra.json"), b"{}").ok();
        std::fs::create_dir_all(root.join("nested")).unwrap();
        std::fs::write(root.join("nested/extra.json"), b"{}").unwrap();

        let plan = PackPlan::scan(&root).unwrap();
        let out = dir.path().join("tiny-model.syn");
        assert_eq!(plan.suggested_out(dir.path()), out);
        assert!(plan.required_space() > plan.payload_bytes());
        let items = plan.item_count();
        plan.into_builder().unwrap().write(&out).unwrap();

        let b = crate::Bundle::open(&out).unwrap();
        assert_eq!(b.meta().id, "tiny-model");
        assert_eq!(b.meta().arch, "qwen3");
        assert_eq!(String::from_utf8_lossy(&b.read_file("nested/extra.json").unwrap()), "{}");
        let tensors = inspect::read_header_slice(b.tensors_slice_named("main").unwrap()).unwrap();
        assert_eq!(tensors.len(), 1);
        assert_eq!(items, 3, "1 компонент + config.json + nested/extra.json");
    }

    #[test]
    fn version_suffix_is_only_taken_from_the_tail() {
        assert_eq!(split_name_version("qwen3.8-27b").1, "1.0.0");
        assert_eq!(split_name_version("ltx-2.3-22b-distilled-1.1").1, "1.1.0");
        assert_eq!(split_name_version("model-v2.0.1").1, "2.0.1");
        assert_eq!(split_name_version("gigaam-v3").1, "1.0.0", "v3 без точки — не версия");
        assert_eq!(split_name_version("Qwen3.8 Flash Next").0, "qwen3.8-flash-next");
    }

    #[test]
    fn shard_prefix_matches_only_the_hf_pattern() {
        assert_eq!(shard_prefix("model-00001-of-00012.safetensors").as_deref(), Some("model"));
        assert_eq!(
            shard_prefix("diffusion_pytorch_model-00001-of-00003.safetensors").as_deref(),
            Some("diffusion_pytorch_model")
        );
        assert_eq!(shard_prefix("ltx-2.3-22b-dev.safetensors"), None);
        assert_eq!(shard_prefix("model.safetensors"), None);
    }
}
