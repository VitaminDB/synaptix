//! Арена экспертов: вытеснение slab'ом возвращает VRAM драйверу.
//!
//! Пара к `experts_pool_fragmentation.rs`, который показывает исходную
//! болезнь: освобождение резидентов вразнобой не возвращает драйверу ничего,
//! потому что `cuMemPoolTrimTo` отдаёт только полностью свободные сегменты.
//! Арена нарезает эксперты внутри крупных slab'ов, и когда из slab'а ушёл
//! последний резидент, драйвер получает его целиком.

use cudarc::driver::DevicePtr;
use synaptix_core::device::cuda;
use synaptix_core::device::Device;
use synaptix_core::memory::expert_arena;

/// Размер эксперта qwen3.8-flash-next: 13.07 ГБ на 4567 резидентов.
const EXPERT_BYTES: usize = 3_000_000;
const BLOCKS: usize = 512;

fn mb(x: usize) -> usize {
    x / (1024 * 1024)
}

#[test]
fn evicting_a_whole_slab_returns_it_to_the_driver() {
    let Ok(stream) = cuda::default_stream(0) else {
        eprintln!("CUDA недоступна — тест пропущен");
        return;
    };
    if !expert_arena::enabled() {
        eprintln!("арена выключена через SYN_EXPERT_ARENA — тест пропущен");
        return;
    }

    // Набиваем арену «экспертами». Группируем по slab'ам, как это делает кэш.
    let mut blocks: Vec<(u64, cudarc::driver::CudaSlice<u8>)> = Vec::with_capacity(BLOCKS);
    {
        let _experts = cuda::ExpertsAllocGuard::for_device(Device::Cuda(0));
        for _ in 0..BLOCKS {
            let buf = match unsafe { cuda::alloc_bytes_uninit(&stream, EXPERT_BYTES) } {
                Ok(b) => b,
                Err(e) => {
                    eprintln!("не хватило VRAM на подготовку ({e:?}) — тест пропущен");
                    return;
                }
            };
            let slab = {
                let (ptr, _sync) = buf.device_ptr(&stream);
                expert_arena::slab_of(ptr).expect("блок обязан лежать в арене")
            };
            blocks.push((slab, buf));
        }
    }
    let _ = cuda::synchronize_all(0);

    let st = expert_arena::stats();
    assert!(
        st.slabs >= 2,
        "на {} блоках по {} МБ ожидали несколько slab'ов, получили {}",
        BLOCKS,
        mb(EXPERT_BYTES),
        st.slabs
    );
    let (free_full, _) = cuda::mem_info(0).expect("mem_info");

    // Жертва — самый старый slab: ровно тот выбор, что делает кэш экспертов.
    let victim = expert_arena::slabs_by_age()[0];
    let victim_blocks = blocks.iter().filter(|(s, _)| *s == victim).count();
    blocks.retain(|(s, _)| *s != victim);
    let _ = cuda::synchronize_all(0);
    let returned = expert_arena::release_empty(0);
    let (free_after, _) = cuda::mem_info(0).expect("mem_info");
    let by_driver = free_after.saturating_sub(free_full);

    eprintln!(
        "slab'ов {}, в жертве {} блоков; арена отпустила {} MB, драйвер увидел {} MB",
        st.slabs,
        victim_blocks,
        mb(returned),
        mb(by_driver),
    );

    // Slab чуть короче номинала: длина режется по размеру слота, хвост в
    // неполный слот не выделяется (89 слотов по ~2.9 МБ из 256 МБ).
    assert!(
        returned * 10 >= expert_arena::slab_bytes() as usize * 9,
        "арена не отдала slab целиком: {} MB из номинальных {} MB",
        mb(returned),
        mb(expert_arena::slab_bytes())
    );
    assert!(
        by_driver * 10 >= returned * 8,
        "драйверу вернулось {} MB из отпущенных ареной {} MB",
        mb(by_driver),
        mb(returned)
    );

    // Уцелевшие резиденты должны пережить освобождение чужого slab'а.
    assert!(!blocks.is_empty());
    drop(blocks);
    let _ = cuda::synchronize_all(0);
    expert_arena::release_empty(0);
}
