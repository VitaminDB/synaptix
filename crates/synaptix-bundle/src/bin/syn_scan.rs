//! syn-scan — что лежит в каталоге и как оно упакуется.
//!
//! Без аргументов-режимов печатает содержимое папки: готовые `.syn` и
//! модели-кандидаты. С `--plan` разбирает конкретную модель (каталог или
//! одиночный `.safetensors`) и показывает план: компоненты, файлы, догадки
//! о метаданных и требуемое место. Это тот же [`PackPlan`], который
//! использует `syn-pack` и страница пакетов в synthos, — что напечатано,
//! то и будет упаковано.

use std::path::PathBuf;

use synaptix_bundle::inspect;
use synaptix_bundle::pack_plan::{self, FoundItem, Guess, PackPlan};

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let path: PathBuf = args
        .next()
        .ok_or_else(|| anyhow::anyhow!("usage: syn-scan <dir> [--plan]"))?
        .into();
    let plan_mode = args.next().as_deref() == Some("--plan");

    if plan_mode {
        print_plan(&PackPlan::scan(&path)?);
    } else {
        print_collection(&path)?;
    }
    Ok(())
}

fn print_collection(dir: &PathBuf) -> anyhow::Result<()> {
    let items = pack_plan::scan_collection(dir)?;
    if items.is_empty() {
        println!("{}: ни пакетов, ни моделей", dir.display());
        return Ok(());
    }
    let bundles: Vec<&FoundItem> = items
        .iter()
        .filter(|i| matches!(i, FoundItem::Bundle { .. }))
        .collect();
    let sources: Vec<&FoundItem> = items
        .iter()
        .filter(|i| matches!(i, FoundItem::Source(_)))
        .collect();

    if !bundles.is_empty() {
        println!("Пакеты ({}):", bundles.len());
        for b in bundles {
            println!("  {:<48} {}", b.name(), human(b.bytes()));
        }
    }
    if !sources.is_empty() {
        println!("Можно упаковать ({}):", sources.len());
        for s in sources {
            let FoundItem::Source(c) = s else { continue };
            let arch = if c.arch.is_empty() { "—".to_string() } else { c.arch.clone() };
            println!(
                "  {:<48} {:>10}  {:<16} шардов: {:<3} компонентов: {}",
                c.name,
                human(c.bytes),
                arch,
                c.shard_count,
                c.component_count
            );
        }
    }
    Ok(())
}

fn print_plan(plan: &PackPlan) {
    println!("{}  ({:?})", plan.root.display(), plan.kind);
    println!(
        "  id={} version={} arch={} purpose={}",
        plan.meta.id,
        plan.meta.version,
        show(&plan.meta.arch),
        show(&plan.meta.purpose)
    );
    println!(
        "  источники догадок: arch={:?} purpose={:?} version={:?}",
        plan.meta.arch_from, plan.meta.purpose_from, plan.meta.version_from
    );
    if plan.meta.arch_from == Guess::Unknown {
        println!("  ! архитектуру определить не удалось — заполните вручную");
    }
    println!();

    println!("компоненты:");
    for c in &plan.components {
        let mark = if c.enabled { "+" } else { "-" };
        println!(
            "  {mark} {:<20} {:>10}  шардов: {:<3} {}",
            c.name,
            human(c.bytes),
            c.paths.len(),
            c.note
        );
    }
    let shown = 12.min(plan.aux.len());
    println!("\nфайлы ({}):", plan.aux.len());
    for f in plan.aux.iter().take(shown) {
        println!("  + {:<52} {:>10}  [{}]", f.rel, human(f.bytes), f.tag.as_str());
    }
    if plan.aux.len() > shown {
        println!("  … ещё {}", plan.aux.len() - shown);
    }
    print_roles(plan);
    println!();
    println!("  payload:  {}", human(plan.payload_bytes()));
    println!("  нужно на диске: {}", human(plan.required_space()));
    println!("  этапов: {}", plan.item_count());
    for w in &plan.warnings {
        println!("  ! {w}");
    }
}

/// Состав главного компонента по ролям слоёв. Заголовки читаются у всех
/// шардов: по одному первому картина врёт — в нём лежит vision-башня и
/// доля выходит 86 % вместо полутора процентов.
fn print_roles(plan: &PackPlan) {
    let Some(comp) = plan.components.iter().find(|c| c.enabled) else {
        return;
    };
    let mut tensors: Vec<inspect::TensorInfo> = Vec::new();
    for shard in &comp.paths {
        if let Ok(mut t) = inspect::read_header_file(shard) {
            tensors.append(&mut t);
        }
    }
    if tensors.is_empty() {
        return;
    }
    let hint = format!("{} {} {}", comp.name, plan.meta.purpose, plan.meta.id);
    let total: u64 = tensors.iter().map(|t| t.bytes).sum();
    let mut by_role: Vec<_> = inspect::bytes_by_role(&tensors, Some(&hint)).into_iter().collect();
    by_role.sort_by(|a, b| b.1.dense.cmp(&a.1.dense));
    println!("\nслои «{}» — тензоров: {}", comp.name, tensors.len());
    for (role, est) in by_role {
        let pct = if total == 0 { 0.0 } else { est.dense as f64 / total as f64 * 100.0 };
        println!(
            "  {:<14} {:>10}  {:>5.1} %   nvfp4 {:>10}  mxfp8 {:>10}",
            role.key(),
            human(est.dense),
            pct,
            human(est.nvfp4),
            human(est.mxfp8)
        );
    }
}

fn show(s: &str) -> String {
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
