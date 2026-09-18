use std::path::PathBuf;

use synaptix_video_minimax_h3::refs::{
    block_timestamps, encoder_sample_indices, ref_image_size, resolve_canvas_size,
    round_half_even, snap_ref_frames, validate, RefImageSize, RefKind, RefSource,
};
use synaptix_video_minimax_h3::text_encoder::{
    presentation_ref2va, ImageGrid, RefItem, VideoBlock,
};

fn image(n: usize) -> RefSource {
    RefSource::Image(PathBuf::from(format!("{n}.png")))
}

#[test]
fn kind_by_extension() {
    assert_eq!(RefKind::of_path("a/Hero.PNG".as_ref()), Some(RefKind::Image));
    assert_eq!(RefKind::of_path("clip.mov".as_ref()), Some(RefKind::Video));
    assert_eq!(RefKind::of_path("voice.flac".as_ref()), Some(RefKind::Audio));
    assert_eq!(RefKind::of_path("notes.txt".as_ref()), None);
    assert!(RefSource::from_path("notes.txt", true).is_err());
    assert_eq!(
        RefSource::from_path("clip.mp4", true).unwrap(),
        RefSource::Video { path: "clip.mp4".into(), use_audio: true }
    );
}

#[test]
fn limits_of_released_checkpoint() {
    assert!(validate(&[]).is_ok());
    assert!(validate(&(0..9).map(image).collect::<Vec<_>>()).is_ok());
    assert!(validate(&(0..10).map(image).collect::<Vec<_>>()).is_err());

    let video = |n: usize| RefSource::Video { path: format!("{n}.mp4").into(), use_audio: true };
    assert!(validate(&(0..3).map(video).collect::<Vec<_>>()).is_ok());
    assert!(validate(&(0..4).map(video).collect::<Vec<_>>()).is_err());

    let audio = |n: usize| RefSource::Audio(format!("{n}.wav").into());
    assert!(validate(&[audio(0)]).is_err(), "аудио не бывает единственным референсом");
    assert!(validate(&[image(0), audio(0)]).is_ok());

    let mut full: Vec<RefSource> = (0..9).map(image).collect();
    full.extend((0..3).map(video));
    assert!(validate(&full).is_ok());
    full.push(audio(0));
    assert!(validate(&full).is_err(), "13 референсов при потолке 12");
}

#[test]
fn python_rounding() {
    assert_eq!(round_half_even(22.5), 22.0);
    assert_eq!(round_half_even(23.5), 24.0);
    assert_eq!(round_half_even(22.4), 22.0);
    assert_eq!(round_half_even(22.6), 23.0);
}

#[test]
fn canvas_follows_reference_rule() {
    assert_eq!(resolve_canvas_size(1920, 1080).unwrap(), (1344, 768));
    assert_eq!(resolve_canvas_size(16, 9).unwrap(), (1344, 768));
    assert_eq!(resolve_canvas_size(1080, 1920).unwrap(), (768, 1344));
    assert_eq!(resolve_canvas_size(1000, 1000).unwrap(), (768, 768));
    assert_eq!(resolve_canvas_size(4, 3).unwrap(), (1024, 768));
    assert!(resolve_canvas_size(5000, 1000).is_err());
}

#[test]
fn reference_image_sizes() {
    // max: короткая сторона 2048, в обе стороны, кратно 32.
    assert_eq!(ref_image_size(1920, 1080, RefImageSize::Max, 1344, 768).unwrap(), (3648, 2048));
    assert_eq!(ref_image_size(512, 512, RefImageSize::Max, 1344, 768).unwrap(), (2048, 2048));
    // match: вниз до площади кадра, маленькое не растягивается.
    assert_eq!(ref_image_size(3840, 2160, RefImageSize::Match, 1344, 768).unwrap(), (1344, 768));
    assert_eq!(ref_image_size(640, 480, RefImageSize::Match, 1344, 768).unwrap(), (640, 480));
    assert!(ref_image_size(4100, 1000, RefImageSize::Match, 1344, 768).is_err());
}

#[test]
fn video_frames_snap_down_to_vae_grid() {
    assert_eq!(snap_ref_frames(124), 124);
    assert_eq!(snap_ref_frames(100), 90);
    assert_eq!(snap_ref_frames(30), 22);
    assert_eq!(snap_ref_frames(22), 22);
}

#[test]
fn encoder_reads_video_at_two_fps() {
    let idx = encoder_sample_indices(124);
    assert_eq!(idx, (0..11).map(|i| i * 12).collect::<Vec<_>>());
    let stamps = block_timestamps(idx.len(), 2);
    assert_eq!(stamps, vec![0.25, 1.25, 2.25, 3.25, 4.25, 5.0]);
    // Метка первой пары — «0.2», а не «0.3»: как `f"{:.1f}"` у эталона.
    assert_eq!(format!("{:.1}", stamps[0]), "0.2");
}

#[test]
fn presentation_labels_soundtrack_before_its_video() {
    let grid = ImageGrid { t: 1, h: 4, w: 4 };
    let refs = [
        RefItem::Video { blocks: vec![VideoBlock { grid, seconds: 0.25 }], with_audio: true },
        RefItem::Image { grid },
        RefItem::Audio,
        RefItem::Video { blocks: vec![VideoBlock { grid, seconds: 0.25 }], with_audio: false },
    ];
    let p = presentation_ref2va("prompt", &refs, 2);
    assert_eq!(
        p.text_chunks(),
        vec![
            "<Audio 1>: ",
            "<Video 1>: ",
            "<0.2 seconds>",
            "<Picture 1>: ",
            "<Audio 2>: ",
            "<Video 2>: ",
            "<0.2 seconds>",
            "prompt",
        ]
    );
}

fn ffmpeg(args: &[&str]) -> bool {
    std::process::Command::new("ffmpeg")
        .args(["-v", "error", "-y"])
        .args(args)
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Живой ffmpeg: 30 fps 640x360 со звуком 44.1 кГц моно → 24 fps на холсте
/// своего аспекта и 32 кГц стерео, обрезанные по длине генерации.
#[test]
fn decode_normalizes_media_with_ffmpeg() {
    use synaptix_video_minimax_h3::refs::{decode, RefMedia, RefOptions};

    let dir = std::env::temp_dir().join(format!("h3_refs_test_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let video = dir.join("clip.mp4");
    let picture = dir.join("hero.png");
    let voice = dir.join("voice.wav");
    let made = ffmpeg(&[
        "-f", "lavfi", "-i", "testsrc=size=640x360:rate=30:duration=3",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=3",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
        video.to_str().unwrap(),
    ]) && ffmpeg(&[
        "-f", "lavfi", "-i", "testsrc=size=3000x2000:rate=1:duration=1", "-frames:v", "1",
        picture.to_str().unwrap(),
    ]) && ffmpeg(&[
        "-f", "lavfi", "-i", "sine=frequency=220:sample_rate=48000:duration=9",
        voice.to_str().unwrap(),
    ]);
    if !made {
        eprintln!("ffmpeg недоступен — тест пропущен");
        return;
    }

    let sources = [
        RefSource::from_path(&video, true).unwrap(),
        RefSource::from_path(&picture, true).unwrap(),
        RefSource::from_path(&voice, true).unwrap(),
    ];
    let opts = RefOptions {
        image_size: RefImageSize::Match,
        target_width: 1344,
        target_height: 768,
        frame_count: 124,
    };
    let media = decode(&sources, &opts).unwrap();
    assert_eq!(media.len(), 3);

    let RefMedia::Video(v) = &media[0] else { panic!("первым шло видео") };
    assert_eq!((v.width, v.height), (1344, 768));
    // 3 с при 30 fps → 72 кадра на сетке 24 fps.
    assert!((70..=74).contains(&v.count), "кадров {}", v.count);
    assert_eq!(v.frames.len(), v.count * v.width * v.height * 3);
    let wave = v.audio.as_ref().expect("дорожка видео");
    assert_eq!(wave.samples.len(), wave.len * 2);
    assert!((wave.len as i64 - 3 * 32_000).abs() < 3_200, "сэмплов {}", wave.len);

    let RefMedia::Image(img) = &media[1] else { panic!("вторым шла картинка") };
    // 3000x2000 вниз до площади 1344x768, кратно 32.
    assert_eq!((img.width, img.height), (1248, 832));
    assert_eq!(img.rgb.len(), img.width * img.height * 3);

    let RefMedia::Audio(a) = &media[2] else { panic!("третьим шло аудио") };
    // 9 с обрезаны по длине генерации: 124 кадра = 5,1(6) с.
    let want = (124.0 / 24.0 * 32_000.0) as i64;
    assert!((a.len as i64 - want).abs() < 1_600, "сэмплов {}", a.len);

    // Без дорожки, если её просили не брать.
    let muted = decode(&[RefSource::from_path(&video, false).unwrap()], &opts).unwrap();
    let RefMedia::Video(v) = &muted[0] else { panic!() };
    assert!(v.audio.is_none());

    let _ = std::fs::remove_dir_all(&dir);
}
