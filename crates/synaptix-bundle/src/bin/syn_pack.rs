//! syn-pack — упаковать каталог модели (или одиночный `.safetensors`) в `.syn`.
//!
//! Раскладка распознаётся автоматически ([`synaptix_bundle::pack_plan`]):
//! одиночный `model.safetensors`, шарды по `model.safetensors.index.json`,
//! подкаталоги diffusers-пайплайна, вложенные конфиги и токенайзеры. Всё,
//! что не задано флагами, берётся из плана — обычно достаточно `syn-pack
//! <dir> -o <out.syn>`.
//!
//! Тот же план строит страница пакетов в synthos, поэтому CLI и GUI дают
//! байт в байт одинаковый бандл.
//!
//! Usage:
//!     syn-pack <input> -o <output.syn> [--id NAME] [--version VER] [--arch A]
//!              [--purpose P] [--prefix P] [--dry-run]
//!     syn-pack -o <output.syn> --component <name>:<dir>[:<prefix>] … [--files <aux_dir>]

use std::path::{Path, PathBuf};

use synaptix_bundle::pack_plan::{
    self, Guess, PackPlan, PlanComponent, PlanFile, SourceKind,
};
use synaptix_bundle::{resolve_safetensors_in_dir, Result};

struct Component {
    name: String,
    dir: PathBuf,
    prefix: Option<String>,
}

fn parse_component(s: &str) -> anyhow::Result<Component> {
    let parts: Vec<&str> = s.splitn(3, ':').collect();
    if parts.len() < 2 {
        anyhow::bail!("--component ожидает <name>:<dir>[:<prefix>], получено {s:?}");
    }
    // Пустой префикс (`name:dir:`) — то же, что его отсутствие.
    let prefix = parts.get(2).filter(|p| !p.is_empty()).map(|p| p.to_string());
    Ok(Component {
        name: parts[0].to_string(),
        dir: PathBuf::from(parts[1]),
        prefix,
    })
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        usage();
        return Ok(());
    }

    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut id: Option<String> = None;
    let mut version: Option<String> = None;
    let mut arch: Option<String> = None;
    let mut purpose: Option<String> = None;
    let mut prefix: Option<String> = None;
    let mut components: Vec<Component> = Vec::new();
    let mut files_dir: Option<PathBuf> = None;
    let mut sha256 = false;
    let mut blake3 = false;
    let mut dry_run = false;

    let mut it = args.into_iter();
    while let Some(a) = it.next() {
        let mut need = |flag: &str| -> anyhow::Result<String> {
            it.next().ok_or_else(|| anyhow::anyhow!("{flag} требует значения"))
        };
        match a.as_str() {
            "-o" | "--output" => output = Some(PathBuf::from(need("-o")?)),
            "--id" => id = Some(need("--id")?),
            "--version" => version = Some(need("--version")?),
            "--arch" => arch = Some(need("--arch")?),
            "--purpose" => purpose = Some(need("--purpose")?),
            "--prefix" => prefix = Some(need("--prefix")?),
            "--component" => components.push(parse_component(&need("--component")?)?),
            "--files" => files_dir = Some(PathBuf::from(need("--files")?)),
            "--sha256" => sha256 = true,
            "--blake3" => blake3 = true,
            "--dry-run" => dry_run = true,
            _ if input.is_none() && !a.starts_with('-') => input = Some(PathBuf::from(a)),
            other => anyhow::bail!("неизвестный аргумент: {other}"),
        }
    }

    let output = output.ok_or_else(|| anyhow::anyhow!("не задан -o <output.syn>"))?;

    let mut plan = if components.is_empty() {
        let input = input
            .ok_or_else(|| anyhow::anyhow!("укажите каталог/файл модели или --component"))?;
        PackPlan::scan(&input)?
    } else {
        if input.is_some() {
            anyhow::bail!("--component несовместим с позиционным <input>");
        }
        manual_plan(&components, files_dir.as_deref())?
    };

    // Флаги перекрывают догадки.
    if let Some(v) = id {
        plan.meta.id = v;
    }
    if let Some(v) = version {
        plan.meta.version = v;
    }
    if let Some(v) = arch {
        plan.meta.arch = v;
        plan.meta.arch_from = Guess::Unknown;
    }
    if let Some(v) = purpose {
        plan.meta.purpose = v;
        plan.meta.purpose_from = Guess::Unknown;
    }
    if let Some(p) = prefix {
        // Одиночный режим: префикс относится к главному компоненту.
        if let Some(c) = plan.components.iter_mut().find(|c| c.enabled) {
            c.prefix = p;
        }
    }

    print_plan(&plan, &output);
    if dry_run {
        eprintln!("\n--dry-run: ничего не записано");
        return Ok(());
    }
    check_space(&plan, &output)?;

    // `mut` нужен только при включённых фичах контрольных сумм.
    #[allow(unused_mut)]
    let mut b = plan.into_builder()?;
    if sha256 {
        #[cfg(feature = "sha256")]
        {
            b = b.with_sha256(true);
        }
        #[cfg(not(feature = "sha256"))]
        anyhow::bail!("--sha256 требует сборки с `--features sha256`");
    }
    if blake3 {
        #[cfg(feature = "blake3")]
        {
            b = b.with_blake3(true);
        }
        #[cfg(not(feature = "blake3"))]
        anyhow::bail!("--blake3 требует сборки с `--features blake3`");
    }

    eprintln!("\nупаковка → {}", output.display());
    b.write(&output).map_err(into_anyhow)?;
    let size = std::fs::metadata(&output)?.len();
    eprintln!("готово: {} ({})", output.display(), human(size));
    Ok(())
}

/// План из явных `--component`: состав задал человек, догадки не нужны.
fn manual_plan(components: &[Component], files_dir: Option<&Path>) -> anyhow::Result<PackPlan> {
    let mut plan_components: Vec<PlanComponent> = Vec::new();
    for c in components {
        let paths = resolve_safetensors_in_dir(&c.dir).map_err(into_anyhow)?;
        plan_components.push(PlanComponent {
            name: c.name.clone(),
            bytes: paths
                .iter()
                .filter_map(|p| std::fs::metadata(p).ok())
                .map(|m| m.len())
                .sum(),
            paths,
            prefix: c.prefix.clone().unwrap_or_default(),
            enabled: true,
            note: String::new(),
        });
    }
    // Вспомогательные файлы — из `--files`, иначе из каталога первого
    // компонента: у VoxCPM2 tokenizer.json лежит рядом с первой подсетью.
    let aux_root = files_dir
        .map(|p| p.to_path_buf())
        .or_else(|| components.first().map(|c| c.dir.clone()));
    let aux: Vec<PlanFile> = match aux_root.as_deref() {
        Some(dir) => pack_plan::aux_files(dir)?,
        None => Vec::new(),
    };
    let root = aux_root.unwrap_or_else(|| PathBuf::from("."));
    let id = pack_plan::normalize_id(&components[0].name);
    Ok(PackPlan {
        root,
        kind: SourceKind::MultiComponentDir,
        components: plan_components,
        aux,
        meta: pack_plan::GuessedMeta {
            id,
            version: "1.0.0".into(),
            arch: String::new(),
            purpose: String::new(),
            arch_from: Guess::Unknown,
            purpose_from: Guess::Unknown,
            version_from: Guess::Unknown,
        },
        warnings: Vec::new(),
    })
}

fn print_plan(plan: &PackPlan, out: &Path) {
    eprintln!("{} → {}", plan.root.display(), out.display());
    eprintln!(
        "  id={} version={} arch={} purpose={}",
        plan.meta.id,
        plan.meta.version,
        dash(&plan.meta.arch),
        dash(&plan.meta.purpose)
    );
    for c in &plan.components {
        if !c.enabled {
            eprintln!("  - {:<20} пропуск: {}", c.name, c.note);
            continue;
        }
        eprintln!(
            "  + {:<20} {:>10}  шардов: {}{}",
            c.name,
            human(c.bytes),
            c.paths.len(),
            if c.prefix.is_empty() { String::new() } else { format!("  префикс: {}", c.prefix) }
        );
    }
    let aux_bytes: u64 = plan.aux.iter().filter(|f| f.enabled).map(|f| f.bytes).sum();
    eprintln!(
        "  файлов: {} ({}), всего payload: {}",
        plan.aux.iter().filter(|f| f.enabled).count(),
        human(aux_bytes),
        human(plan.payload_bytes())
    );
    for w in &plan.warnings {
        eprintln!("  ! {w}");
    }
}

/// Проверка места до начала записи: во время упаковки рядом с `out.syn.tmp`
/// живёт промежуточный stage самого крупного компонента. Узнать об этом
/// через десять минут на последнем байте — худший из возможных вариантов.
fn check_space(plan: &PackPlan, out: &Path) -> anyhow::Result<()> {
    let dir = out.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
    let need = plan.required_space();
    let avail = synaptix_bundle::available_space(dir).unwrap_or(u64::MAX);
    if avail < need {
        anyhow::bail!(
            "на {} свободно {}, а нужно ≈{} (payload {} + stage {} + запас)",
            dir.display(),
            human(avail),
            human(need),
            human(plan.payload_bytes()),
            human(plan.max_component_bytes())
        );
    }
    Ok(())
}

fn dash(s: &str) -> String {
    if s.is_empty() { "—".to_string() } else { s.to_string() }
}

fn human(n: u64) -> String {
    const KB: f64 = 1024.0;
    let n = n as f64;
    if n >= KB * KB * KB {
        format!("{:.2} GB", n / (KB * KB * KB))
    } else if n >= KB * KB {
        format!("{:.1} MB", n / (KB * KB))
    } else if n >= KB {
        format!("{:.0} KB", n / KB)
    } else {
        format!("{n} B")
    }
}

fn usage() {
    eprintln!(
        "syn-pack — упаковать модель в один файл .syn\n\n\
         ИСПОЛЬЗОВАНИЕ:\n\
         \x20 syn-pack <каталог|файл.safetensors> -o <out.syn> [флаги]\n\
         \x20 syn-pack -o <out.syn> --component <name>:<dir>[:<prefix>] … [--files <aux_dir>]\n\n\
         ФЛАГИ:\n\
         \x20 -o, --output <путь>     куда писать .syn\n\
         \x20 --id <name>             идентификатор бандла (по умолчанию — из имени)\n\
         \x20 --version <v>           версия (по умолчанию — из имени или 1.0.0)\n\
         \x20 --arch <name>           архитектура (по умолчанию — model_type из config.json)\n\
         \x20 --purpose <p>           назначение: embed / asr / tts / video / lora …\n\
         \x20 --prefix <p>            префикс имён тензоров главного компонента\n\
         \x20 --component <name>:<dir>[:<prefix>]   задать состав вручную (повторяемый)\n\
         \x20 --files <aux_dir>       откуда брать config.json/tokenizer.json при --component\n\
         \x20 --sha256 | --blake3     контрольные суммы чанков и манифеста\n\
         \x20 --dry-run               показать план и выйти, ничего не записывая\n\n\
         Раскладка определяется сама: model.safetensors → шарды по index.json →\n\
         подкаталоги пайплайна. Огрызки докачки и дубли другой точности\n\
         отбрасываются; посмотреть план заранее — `syn-scan <путь> --plan`."
    );
}

fn into_anyhow(e: impl std::fmt::Display) -> anyhow::Error {
    anyhow::anyhow!("{e}")
}

#[allow(dead_code)]
fn _doc_check(_: Result<()>) {}
