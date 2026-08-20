//! LTX-2.3 Vocoder (BigVGAN-v2 + BWE): log-mel `[B,2,T,64]` → 48кГц стерео
//! waveform. fp32. Веса FUSED (без weight_norm); kaiser-фильтры Activation1d
//! ХРАНЯТСЯ в чекпойнте. Base-генератор 16кГц + BWE 16→48кГц + STFT-as-conv +
//! sinc-resample skip.

use synaptix_core::{device::Device, dtype::DType, error::SynaptixError, tensor::Tensor};
use synaptix_io::weights::safetensors::SafetensorsLoader;
use synaptix_io::weights::WeightLoader;
use synaptix_ops::conv::conv1d::conv1d;
use synaptix_ops::conv::conv_transpose1d::conv_transpose1d;
use synaptix_ops::conv::depthwise::depthwise_conv;

use crate::LtxError;

type R<T> = Result<T, SynaptixError>;
const SNAKE_EPS: f32 = 1e-9;

/// replicate-pad по времени (последняя ось): кромочные значения.
fn replicate_pad1d(x: &Tensor, left: usize, right: usize) -> R<Tensor> {
    let (b, c, l) = (x.dims()[0], x.dims()[1], x.dims()[2]);
    let mut parts: Vec<Tensor> = Vec::new();
    if left > 0 {
        parts.push(x.narrow(2, 0, 1)?.broadcast_as(vec![b, c, left])?.contiguous()?);
    }
    parts.push(x.contiguous()?);
    if right > 0 {
        parts.push(x.narrow(2, l - 1, 1)?.broadcast_as(vec![b, c, right])?.contiguous()?);
    }
    let refs: Vec<&Tensor> = parts.iter().collect();
    Tensor::cat(&refs, 2)
}

/// SnakeBeta: `x + (1/(exp(β)+ε))·sin(x·exp(α))²`. α,β `[C]`.
fn snakebeta(x: &Tensor, alpha: &Tensor, beta: &Tensor) -> R<Tensor> {
    let c = alpha.dims()[0];
    let a = alpha.reshape(vec![1, c, 1])?.exp()?;
    let b = beta.reshape(vec![1, c, 1])?.exp()?.add_scalar(SNAKE_EPS)?.powf(-1.0)?; // 1/(exp(β)+ε)
    let s = x.broadcast_mul(&a)?.sin()?.sqr()?;
    x.add(&s.broadcast_mul(&b)?)
}

/// Хранёный depthwise-фильтр `[1,1,K]` → `[C,1,K]`.
fn expand_filt(f: &Tensor, c: usize) -> R<Tensor> {
    let k = f.dims()[2];
    f.broadcast_as(vec![c, 1, k])?.contiguous()
}

struct Act1d {
    alpha: Tensor,
    beta: Tensor,
    up_filt: Tensor,   // [1,1,12]
    down_filt: Tensor, // [1,1,12]
}
impl Act1d {
    /// anti-aliased: upsample ×2 (replicate-pad 5/5 + 2·convT depthwise + crop
    /// 15/15) → snakebeta → downsample ÷2 (replicate-pad 5/6 + dwconv stride2).
    fn forward(&self, x: &Tensor) -> R<Tensor> {
        let c = x.dims()[1];
        // upsample
        let xp = replicate_pad1d(x, 5, 5)?;
        let filt = expand_filt(&self.up_filt, c)?;
        let y = conv_transpose1d(&xp, &filt, None, 2, 0, 0, c, 1)?.mul_scalar(2.0)?;
        let yl = y.dims()[2];
        let up = y.narrow(2, 15, yl - 30)?.contiguous()?; // [pad_left:-pad_right]=[15:-15]
        // snakebeta
        let a = snakebeta(&up, &self.alpha, &self.beta)?;
        // downsample
        let dp = replicate_pad1d(&a, 5, 6)?;
        let df = expand_filt(&self.down_filt, c)?;
        depthwise_conv(&dp, &df, None, 2, 0, c)
    }
}

struct Conv1 {
    w: Tensor,
    b: Option<Tensor>,
    pad: usize,
}
impl Conv1 {
    fn fwd(&self, x: &Tensor) -> R<Tensor> {
        conv1d(x, &self.w, self.b.as_ref(), 1, self.pad)
    }
}

/// compute-dtype вокодера: F32 (bf16-эксперимент дал ×2 МЕДЛЕННЕЕ — наш conv1d
/// bf16-путь слабее f32).
fn voc_dtype() -> DType {
    DType::F32
}

/// Финальная активация генератора (PixArt-vocoder: tanh/clamp/нет).
#[derive(Clone, Copy)]
enum FinalAct {
    None,
    Clamp,
}

/// AMPBlock1: 3×(snakebeta-act → dilated conv1 → snakebeta-act → conv2) + residual.
struct AmpBlock {
    convs1: Vec<(Tensor, Tensor, usize, usize)>, // (w,b,pad,dilation)
    convs2: Vec<(Tensor, Tensor, usize)>,        // (w,b,pad) dilation=1
    acts1: Vec<Act1d>,
    acts2: Vec<Act1d>,
}
impl AmpBlock {
    fn forward(&self, x: &Tensor) -> R<Tensor> {
        let mut x = x.clone();
        for i in 0..3 {
            let xt = self.acts1[i].forward(&x)?;
            let (w1, b1, p1, d1) = &self.convs1[i];
            let xt = conv1d_dil(&xt, w1, b1, *p1, *d1)?;
            let xt = self.acts2[i].forward(&xt)?;
            let (w2, b2, p2) = &self.convs2[i];
            let xt = conv1d(&xt, w2, Some(b2), 1, *p2)?;
            x = x.add(&xt)?;
        }
        Ok(x)
    }
}

/// conv1d с dilation: разворачиваем через зону приёма (dilation реализуем
/// прорежённым ядром невозможно прямо → используем зануление между tap'ами).
/// Проще: dilation эквивалентен conv с расширенным ядром (вставка нулей).
fn conv1d_dil(x: &Tensor, w: &Tensor, b: &Tensor, pad: usize, dilation: usize) -> R<Tensor> {
    if dilation == 1 {
        return conv1d(x, w, Some(b), 1, pad);
    }
    // расширяем ядро: [Cout,Cin,K] → [Cout,Cin,(K-1)*dil+1] вставкой нулей.
    let (co, ci, k) = (w.dims()[0], w.dims()[1], w.dims()[2]);
    let kd = (k - 1) * dilation + 1;
    let z = Tensor::zeros(vec![co, ci, dilation - 1], w.dtype(), w.device())?;
    // интерлив w по последней оси: [.., k, 1] cat zeros [..,k,dil-1] → [..,k,dil] → [..,k*dil], crop kd
    let w4 = w.reshape(vec![co, ci, k, 1])?;
    let z4 = z.reshape(vec![co, ci, 1, dilation - 1])?.broadcast_as(vec![co, ci, k, dilation - 1])?.contiguous()?;
    let we = Tensor::cat(&[&w4, &z4], 3)?.contiguous()?.reshape(vec![co, ci, k * dilation])?
        .narrow(2, 0, kd)?.contiguous()?;
    conv1d(x, &we, Some(b), 1, pad)
}

/// Base BigVGAN-v2 генератор: conv_pre → 6×(up + 3 amp-resblock mean) →
/// act_post → conv_post. Вход `[B, in_ch, L]`, выход `[B, 2, L_out]`.
pub struct Generator {
    conv_pre: Conv1,
    ups: Vec<(Tensor, Tensor, usize, usize)>, // (w,b,stride,pad)
    resblocks: Vec<AmpBlock>,                 // num_upsamples*3
    act_post: Act1d,
    conv_post: Conv1,
    num_upsamples: usize,
    num_kernels: usize,
    final_act: FinalAct,
}

impl Generator {
    fn forward(&self, x: &Tensor) -> R<Tensor> {
        // пер-уровневая разбивка (sync) — охота на доминанту.
        let prof = crate::runtime::ltx_voc_prof();
        let sync = || { let _ = synaptix_core::device::cuda::synchronize(0); };
        let mut tl = std::time::Instant::now();
        let mut x = self.conv_pre.fwd(x)?;
        if prof { sync(); eprintln!("[VOC] conv_pre {:?}: {:.0}ms", x.dims(), tl.elapsed().as_secs_f32()*1e3); tl = std::time::Instant::now(); }
        for i in 0..self.num_upsamples {
            let (w, b, stride, pad) = &self.ups[i];
            x = conv_transpose1d(&x, w, Some(b), *stride, *pad, 0, 1, 1)?;
            if prof { sync(); eprintln!("[VOC] up{i} convT {:?}: {:.0}ms", x.dims(), tl.elapsed().as_secs_f32()*1e3); tl = std::time::Instant::now(); }
            // 3 resblock'а с одним входом → mean
            let start = i * self.num_kernels;
            let mut acc: Option<Tensor> = None;
            for j in 0..self.num_kernels {
                let o = self.resblocks[start + j].forward(&x)?;
                acc = Some(match acc {
                    Some(a) => a.add(&o)?,
                    None => o,
                });
            }
            x = acc.unwrap().mul_scalar(1.0 / self.num_kernels as f32)?;
            if prof { sync(); eprintln!("[VOC] up{i} resblocks: {:.0}ms", tl.elapsed().as_secs_f32()*1e3); tl = std::time::Instant::now(); }
        }
        x = self.act_post.forward(&x)?;
        let x = self.conv_post.fwd(&x)?;
        if prof { sync(); eprintln!("[VOC] post: {:.0}ms", tl.elapsed().as_secs_f32()*1e3); }
        match self.final_act {
            FinalAct::None => Ok(x),
            FinalAct::Clamp => x.clamp(-1.0, 1.0),
        }
    }
}

/// Загрузчик одного генератора (base `vocoder.vocoder.*` или `vocoder.bwe_generator.*`).
fn load_generator(
    ld: &SafetensorsLoader,
    prefix: &str,
    up_rates: &[usize],
    up_kernels: &[usize],
    rb_kernels: &[usize],
    rb_dils: &[[usize; 3]],
    final_act: FinalAct,
    device: Device,
) -> Result<Generator, LtxError> {
    // F32-компьют: наш conv1d bf16-путь ×2 медленнее f32 на вокодере.
    let vdt = voc_dtype();
    let g = |n: &str| -> Result<Tensor, LtxError> {
        ld.load_to(&format!("{prefix}.{n}"), device, vdt).map_err(|e| LtxError::Load(format!("{n}: {e}")))
    };
    let conv = |n: &str, pad: usize, bias: bool| -> Result<Conv1, LtxError> {
        let b = if bias { Some(g(&format!("{n}.bias"))?) } else { None };
        Ok(Conv1 { w: g(&format!("{n}.weight"))?, b, pad })
    };
    let act = |n: &str| -> Result<Act1d, LtxError> {
        Ok(Act1d {
            alpha: g(&format!("{n}.act.alpha"))?,
            beta: g(&format!("{n}.act.beta"))?,
            up_filt: g(&format!("{n}.upsample.filter"))?,
            down_filt: g(&format!("{n}.downsample.lowpass.filter"))?,
        })
    };
    let num_up = up_rates.len();
    let num_k = rb_kernels.len();
    let mut ups = Vec::new();
    for i in 0..num_up {
        let pad = (up_kernels[i] - up_rates[i]) / 2;
        ups.push((g(&format!("ups.{i}.weight"))?, g(&format!("ups.{i}.bias"))?, up_rates[i], pad));
    }
    let mut resblocks = Vec::new();
    for i in 0..num_up {
        for (ki, &ks) in rb_kernels.iter().enumerate() {
            let idx = i * num_k + ki;
            let dils = rb_dils[ki];
            let mut convs1 = Vec::new();
            let mut convs2 = Vec::new();
            let mut acts1 = Vec::new();
            let mut acts2 = Vec::new();
            for j in 0..3 {
                let p1 = (ks * dils[j] - dils[j]) / 2; // get_padding(ks, dil)
                convs1.push((g(&format!("resblocks.{idx}.convs1.{j}.weight"))?, g(&format!("resblocks.{idx}.convs1.{j}.bias"))?, p1, dils[j]));
                let p2 = (ks - 1) / 2; // get_padding(ks, 1)
                convs2.push((g(&format!("resblocks.{idx}.convs2.{j}.weight"))?, g(&format!("resblocks.{idx}.convs2.{j}.bias"))?, p2));
                acts1.push(act(&format!("resblocks.{idx}.acts1.{j}"))?);
                acts2.push(act(&format!("resblocks.{idx}.acts2.{j}"))?);
            }
            resblocks.push(AmpBlock { convs1, convs2, acts1, acts2 });
        }
    }
    Ok(Generator {
        conv_pre: conv("conv_pre", 3, true)?,
        ups,
        resblocks,
        act_post: act("act_post")?,
        conv_post: conv("conv_post", 3, false)?,
        num_upsamples: num_up,
        num_kernels: num_k,
        final_act,
    })
}

/// Base-генератор LTX-2.3 (16кГц), `vocoder.vocoder.*`.
pub struct BaseVocoder {
    generator: Generator,
}
impl BaseVocoder {
    pub fn load(path: impl AsRef<std::path::Path>, device: Device) -> Result<Self, LtxError> {
        // `.syn`-бандл или сырой safetensors — как у чекпойнта (вокодер лежит
        // в том же файле под префиксом `vocoder.`).
        let ld = crate::loader::open_weights(path.as_ref())?.with_device(device);
        let generator = load_generator(
            &ld, "vocoder.vocoder",
            &[5, 2, 2, 2, 2, 2], &[11, 4, 4, 4, 4, 4],
            &[3, 7, 11], &[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            FinalAct::Clamp, device,
        )?;
        Ok(Self { generator })
    }

    /// Mel `[B,2,T,64]` → 16кГц wave `[B,2,L]`. (transpose→stereo-merge→generator)
    pub fn forward(&self, mel: &Tensor) -> Result<Tensor, LtxError> {
        Ok(self.generator.forward(&mel_to_channels(mel)?)?)
    }
}

/// Mel `[B,2,T,64]` → `[B,128,T]`: `transpose(2,3)` затем `(s c)`-склейка
/// (как `einops "b s c t -> b (s c) t"` поверх `[B,2,64,T]`). Вход кастуется в
/// compute-dtype вокодера (F32).
fn mel_to_channels(mel: &Tensor) -> R<Tensor> {
    let vdt = voc_dtype();
    let (b, s, t, m) = (mel.dims()[0], mel.dims()[1], mel.dims()[2], mel.dims()[3]);
    mel.to_dtype(vdt)?.transpose(2, 3)?.contiguous()?.reshape(vec![b, s * m, t])
}

/// STFT-как-свёртка: каузальный zero-pad слева `win−hop`, conv1d по хранёному
/// `forward_basis [2·F,1,win]` (stride=hop), |·| из real/imag, проекция на
/// `mel_basis [M,F]`, `log(clamp(·,1e-5))`. Все буферы из чекпойнта.
struct MelStft {
    forward_basis: Tensor, // [514,1,512]
    mel_basis: Tensor,     // [64,257]
    hop: usize,
    win: usize,
}
impl MelStft {
    /// `y` `[N,T]` → log-mel `[N,M,frames]`.
    fn log_mel(&self, y: &Tensor) -> R<Tensor> {
        let (n, t) = (y.dims()[0], y.dims()[1]);
        let y = y.reshape(vec![n, 1, t])?; // [N,1,T]
        let left = self.win - self.hop; // каузально: только слева
        let yp = if left > 0 {
            let z = Tensor::zeros(vec![n, 1, left], y.dtype(), y.device())?;
            Tensor::cat(&[&z, &y], 2)?
        } else {
            y
        };
        let spec = conv1d(&yp, &self.forward_basis, None, self.hop, 0)?; // [N,514,frames]
        let nfreq = spec.dims()[1] / 2;
        let real = spec.narrow(1, 0, nfreq)?;
        let imag = spec.narrow(1, nfreq, nfreq)?;
        let mag = real.sqr()?.add(&imag.sqr()?)?.sqrt()?; // [N,F,frames]
        let mel = self.mel_basis.matmul(&mag)?; // [M,F]·[N,F,frames] → [N,M,frames]
        mel.clamp(1e-5, f32::INFINITY)?.log()
    }
}

/// Hann-окно sinc-ресемплер 16→48кГц (`UpSample1d window_type="hann"`, ratio=3).
/// Фильтр вычисляется (в чекпойнте НЕ хранится, `persistent=False`).
struct HannResampler {
    filt: Tensor, // [1,1,K]
    ratio: usize,
    pad: usize,
    pad_left: usize,
    pad_right: usize,
}
impl HannResampler {
    /// `ratio = out_sr/in_sr` (=3). См. `vocoder.py::UpSample1d` hann-ветку.
    fn new(ratio: usize, device: Device) -> R<Self> {
        let rolloff = 0.99f64;
        let lpw = 6.0f64; // lowpass_filter_width
        let width = (lpw / rolloff).ceil() as usize; // 7
        let k = 2 * width * ratio + 1; // 43
        let pad = width; // 7
        let pad_left = 2 * width * ratio; // 42
        let pad_right = k - ratio; // 40
        let mut taps = vec![0f32; k];
        let pi = std::f64::consts::PI;
        for (i, tap) in taps.iter_mut().enumerate() {
            let ta = (i as f64 / ratio as f64 - width as f64) * rolloff; // time_axis
            let tc = ta.clamp(-lpw, lpw); // time_clamped
            let win = (tc * pi / lpw / 2.0).cos().powi(2);
            // torch.sinc(x) = sin(pi x)/(pi x), sinc(0)=1
            let sinc = if ta == 0.0 { 1.0 } else { (pi * ta).sin() / (pi * ta) };
            *tap = (sinc * win * rolloff / ratio as f64) as f32;
        }
        let filt = Tensor::from_vec(taps, (1, 1, k), device)?.to_dtype(voc_dtype())?;
        Ok(Self { filt, ratio, pad, pad_left, pad_right })
    }

    /// `x` `[B,C,L]` → `[B,C,L·ratio]`.
    fn forward(&self, x: &Tensor) -> R<Tensor> {
        let c = x.dims()[1];
        let xp = replicate_pad1d(x, self.pad, self.pad)?;
        let filt = expand_filt(&self.filt, c)?; // [C,1,K]
        let y = conv_transpose1d(&xp, &filt, None, self.ratio, 0, 0, c, 1)?
            .mul_scalar(self.ratio as f32)?;
        let yl = y.dims()[2];
        y.narrow(2, self.pad_left, yl - self.pad_left - self.pad_right)?.contiguous()
    }
}

/// Полный LTX-2.3 вокодер: base 16кГц → BWE-расширение до 48кГц. fp32.
/// `forward(mel [B,2,T,64]) → wave [B,2,L48]`. Логика `VocoderWithBWE`.
pub struct VocoderWithBwe {
    base: Generator,
    bwe: Generator,
    stft: MelStft,
    resampler: HannResampler,
    hop: usize,
    in_sr: usize,
    out_sr: usize,
}
impl VocoderWithBwe {
    pub fn load(path: impl AsRef<std::path::Path>, device: Device) -> Result<Self, LtxError> {
        // `.syn`-бандл или сырой safetensors — как у чекпойнта (вокодер лежит
        // в том же файле под префиксом `vocoder.`).
        let ld = crate::loader::open_weights(path.as_ref())?.with_device(device);
        let base = load_generator(
            &ld, "vocoder.vocoder",
            &[5, 2, 2, 2, 2, 2], &[11, 4, 4, 4, 4, 4],
            &[3, 7, 11], &[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            FinalAct::Clamp, device,
        )?;
        let bwe = load_generator(
            &ld, "vocoder.bwe_generator",
            &[6, 5, 2, 2, 2], &[12, 11, 4, 4, 4],
            &[3, 7, 11], &[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            FinalAct::None, device,
        )?;
        let g = |n: &str| ld.load_to(&format!("vocoder.mel_stft.{n}"), device, voc_dtype())
            .map_err(|e| LtxError::Load(format!("{n}: {e}")));
        let stft = MelStft {
            forward_basis: g("stft_fn.forward_basis")?,
            mel_basis: g("mel_basis")?,
            hop: 80,
            win: 512,
        };
        let (in_sr, out_sr) = (16000usize, 48000usize);
        let resampler = HannResampler::new(out_sr / in_sr, device)?;
        Ok(Self { base, bwe, stft, resampler, hop: 80, in_sr, out_sr })
    }

    /// log-mel из base-волны `[B,C,L]` → `[B,C,M,frames]`.
    fn compute_mel(&self, audio: &Tensor) -> R<Tensor> {
        let (b, c, l) = (audio.dims()[0], audio.dims()[1], audio.dims()[2]);
        let flat = audio.contiguous()?.reshape(vec![b * c, l])?;
        let mel = self.stft.log_mel(&flat)?; // [B*C,M,frames]
        let (m, fr) = (mel.dims()[1], mel.dims()[2]);
        mel.reshape(vec![b, c, m, fr])
    }

    /// `mel [B,2,T,64]` → 48кГц wave `[B,2,L48]`. (fp32; clamp(residual+skip,-1,1))
    pub fn forward(&self, mel: &Tensor) -> Result<Tensor, LtxError> {
        // base 16кГц
        let x = self.base.forward(&mel_to_channels(mel)?)?; // [B,2,L16]
        let l16 = x.dims()[2];
        let out_len = l16 * self.out_sr / self.in_sr;
        // pad до кратности hop (для точного числа mel-кадров)
        let rem = l16 % self.hop;
        let x = if rem != 0 {
            let (b, c) = (x.dims()[0], x.dims()[1]);
            let z = Tensor::zeros(vec![b, c, self.hop - rem], x.dtype(), x.device())?;
            Tensor::cat(&[&x, &z], 2)?
        } else {
            x
        };
        // mel из base-волны → bwe_generator → residual
        let mel_b = self.compute_mel(&x)?; // [B,C,M,frames]
        let mel_for_bwe = mel_b.transpose(2, 3)?.contiguous()?; // [B,C,frames,M]
        let residual = self.bwe.forward(&mel_to_channels(&mel_for_bwe)?)?; // [B,2,L48]
        // skip = sinc-ресемпл base-волны
        let skip = self.resampler.forward(&x)?; // [B,2,L48]
        let sum = residual.add(&skip)?.clamp(-1.0, 1.0)?;
        Ok(sum.narrow(2, 0, out_len)?.contiguous()?)
    }
}

