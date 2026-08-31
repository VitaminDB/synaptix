//! Базовая линия, ради которой заведена арена экспертов.
//!
//! Кэш MoE вытесняет резидентов вразнобой («часы» с битом обращения), а
//! `cuMemPoolTrimTo` отдаёт драйверу только полностью свободные сегменты
//! пула — значит поштучное вытеснение не возвращает НИЧЕГО. На живой сессии
//! это выглядело как фантомный бюджет: кэш ужимался 13.07 → 9.4 ГБ по своему
//! учёту, `cuMemGetInfo` показывал 40 МБ свободных, и следующий префилл падал
//! с `alloc_uninit(215654400) OOM`, имея рядом десять отдаваемых гигабайт.
//!
//! Тест держит это свойство под наблюдением при ВЫКЛЮЧЕННОЙ арене: если
//! однажды драйвер научится отдавать такую россыпь сам, тест упадёт — и
//! `expert_arena` можно будет упростить. Как арена решает ту же задачу,
//! проверяет `expert_arena.rs`.

use synaptix_core::device::cuda;

/// Размер эксперта qwen3.8-flash-next: 13.07 ГБ на 4567 резидентов.
const EXPERT_BYTES: usize = 3_000_000;
const BLOCKS: usize = 512;

fn mb(x: usize) -> usize {
    x / (1024 * 1024)
}

#[test]
fn scattered_eviction_returns_nothing_without_the_arena() {
    // Ставим ДО первого обращения к арене: её выключатель читается один раз.
    std::env::set_var("SYN_EXPERT_ARENA", "0");

    let Ok(stream) = cuda::default_stream(0) else {
        eprintln!("CUDA недоступна — тест пропущен");
        return;
    };
    let Ok((free_start, _)) = cuda::mem_info(0) else {
        eprintln!("CUDA недоступна — тест пропущен");
        return;
    };

    let mut blocks = Vec::with_capacity(BLOCKS);
    {
        let _experts = cuda::ExpertsAllocGuard::for_device(synaptix_core::device::Device::Cuda(0));
        for _ in 0..BLOCKS {
            match unsafe { cuda::alloc_bytes_uninit(&stream, EXPERT_BYTES) } {
                Ok(b) => blocks.push(b),
                Err(e) => {
                    eprintln!("не хватило VRAM на подготовку ({e:?}) — тест пропущен");
                    return;
                }
            }
        }
    }
    let _ = cuda::synchronize_all(0);
    let (free_full, _) = cuda::mem_info(0).expect("mem_info");

    // Вытеснение «часами»: половина резидентов уходит, половина остаётся жить
    // вперемешку с освобождёнными — ровно то, что делал ExpertCache.
    let mut evicted = 0usize;
    let mut kept = Vec::new();
    for (i, b) in blocks.drain(..).enumerate() {
        if i % 2 == 0 {
            drop(b);
            evicted += EXPERT_BYTES;
        } else {
            kept.push(b);
        }
    }
    let _ = cuda::synchronize_all(0);
    let _ = cuda::trim_experts_pool(0);
    let (free_after, _) = cuda::mem_info(0).expect("mem_info");

    let returned = free_after.saturating_sub(free_full);
    eprintln!(
        "свободно: старт {} MB → после набивки {} MB → после вытеснения {} MB; \
         освобождено по учёту {} MB, вернулось драйверу {} MB",
        mb(free_start),
        mb(free_full),
        mb(free_after),
        mb(evicted),
        mb(returned),
    );

    assert!(
        returned * 2 < evicted,
        "пул вернул драйверу {} MB из {} MB освобождённых россыпью — это \
         больше половины: похоже, поштучного вытеснения теперь достаточно и \
         арену экспертов можно упростить",
        mb(returned),
        mb(evicted),
    );

    drop(kept);
    let _ = cuda::synchronize_all(0);
    let _ = cuda::trim_experts_pool(0);
}
