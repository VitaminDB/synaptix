//! Три маленькие сети G2P по графам ONNX (`context.onnx`, `oov-encoder.onnx`, `oov-decoder.onnx`),
//! посчитанные простыми циклами на хосте (F32 между операциями, скалярные произведения — в f64):
//! сети крошечные, а важны точность и детерминизм argmax.
//!
//! * context: эмбеддинги 14 признаков слова → Linear 152→128 → ×маска → 4 остаточных блока
//!   (ReLU → Conv k3 с дилатацией 1/2/4/1 → ReLU → Conv 1×1 → ×маска) → ReLU → form [52], join [2];
//! * энкодер: эмбеддинг символа + позиция → 3 pre-LN слоя (MHA 4×40, FF 640, ReLU) → LayerNorm;
//! * декодер: эмбеддинг символа фонемы + позиция → 3 pre-LN слоя (каузальное MHA, cross-MHA, FF) →
//!   LayerNorm → голова 51; жадный выбор (`argmax`, первый максимум как в numpy).

use synaptix_core::tensor::Tensor;

use crate::{LkError, Result};

pub type Getter<'a> = &'a dyn Fn(&str) -> Result<Tensor>;

fn load(get: Getter, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    let t = get(name)?;
    if t.dims() != shape {
        return Err(LkError::Load(format!("{name}: форма {:?}, ожидалась {:?}", t.dims(), shape)));
    }
    Ok(t.flatten_all()?.to_vec1::<f32>()?)
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0f64;
    for (x, y) in a.iter().zip(b) {
        s += *x as f64 * *y as f64;
    }
    s as f32
}

/// `x @ W + b`, W хранится транспонированной [out][in] для последовательного доступа.
struct Lin {
    wt: Vec<f32>,
    b: Vec<f32>,
    din: usize,
    dout: usize,
}

impl Lin {
    fn new(get: Getter, w: &str, b: &str, din: usize, dout: usize) -> Result<Self> {
        let w = load(get, w, &[din, dout])?;
        let mut wt = vec![0f32; din * dout];
        for i in 0..din {
            for o in 0..dout {
                wt[o * din + i] = w[i * dout + o];
            }
        }
        Ok(Self { wt, b: load(get, b, &[dout])?, din, dout })
    }

    fn apply(&self, x: &[f32]) -> Vec<f32> {
        (0..self.dout).map(|o| self.b[o] + dot(&self.wt[o * self.din..(o + 1) * self.din], x)).collect()
    }
}

struct Norm {
    g: Vec<f32>,
    b: Vec<f32>,
}

impl Norm {
    fn new(get: Getter, p: &str, d: usize) -> Result<Self> {
        Ok(Self { g: load(get, &format!("{p}.weight"), &[d])?, b: load(get, &format!("{p}.bias"), &[d])? })
    }

    fn apply(&self, x: &[f32]) -> Vec<f32> {
        let n = x.len() as f64;
        let mean = x.iter().map(|&v| v as f64).sum::<f64>() / n;
        let var = x.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
        let inv = 1.0 / (var + 1e-5).sqrt();
        x.iter().enumerate().map(|(i, &v)| (((v as f64 - mean) * inv) as f32) * self.g[i] + self.b[i]).collect()
    }
}

const D: usize = 160;
const HEADS: usize = 4;
const DH: usize = 40;
const FF: usize = 640;
const SCALE: f32 = 0.158_113_88;

struct Attn {
    q: Lin,
    k: Lin,
    v: Lin,
    o: Lin,
}

impl Attn {
    fn new(get: Getter, w: &str, b: &str) -> Result<Self> {
        let l = |n: &str| Lin::new(get, &format!("{w}.{n}.MatMul.w1"), &format!("{b}.{n}.bias"), D, D);
        Ok(Self { q: l("q")?, k: l("k")?, v: l("v")?, o: l("o")? })
    }

    /// Внимание одного запроса `q` (уже спроецированного) к ключам/значениям `ks`, `vs`.
    fn attend(&self, q: &[f32], ks: &[Vec<f32>], vs: &[Vec<f32>]) -> Vec<f32> {
        let mut out = vec![0f32; D];
        for h in 0..HEADS {
            let r = h * DH..(h + 1) * DH;
            let sc: Vec<f32> = ks.iter().map(|k| dot(&q[r.clone()], &k[r.clone()]) * SCALE).collect();
            let m = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let e: Vec<f64> = sc.iter().map(|&s| ((s - m) as f64).exp()).collect();
            let z: f64 = e.iter().sum();
            for d in r.clone() {
                let mut acc = 0f64;
                for (j, v) in vs.iter().enumerate() {
                    acc += (e[j] / z) as f32 as f64 * v[d] as f64;
                }
                out[d] = acc as f32;
            }
        }
        self.o.apply(&out)
    }
}

struct Ff {
    a: Lin,
    b: Lin,
}

impl Ff {
    fn new(get: Getter, w: &str, b: &str) -> Result<Self> {
        Ok(Self {
            a: Lin::new(get, &format!("{w}.a.MatMul.w1"), &format!("{b}.a.bias"), D, FF)?,
            b: Lin::new(get, &format!("{w}.b.MatMul.w1"), &format!("{b}.b.bias"), FF, D)?,
        })
    }

    fn apply(&self, x: &[f32]) -> Vec<f32> {
        let h: Vec<f32> = self.a.apply(x).into_iter().map(|v| v.max(0.0)).collect();
        self.b.apply(&h)
    }
}

fn add(x: &mut [f32], y: &[f32]) {
    for (a, b) in x.iter_mut().zip(y) {
        *a += *b;
    }
}

struct EncLayer {
    n1: Norm,
    att: Attn,
    n2: Norm,
    ff: Ff,
}

struct DecLayer {
    n1: Norm,
    att: Attn,
    n2: Norm,
    cross: Attn,
    n3: Norm,
    ff: Ff,
}

/// Буквы → фонемы для незнакомых слов (энкодер + жадный декодер).
pub struct Oov {
    src: Vec<f32>,
    n_src: usize,
    pos_s: Vec<f32>,
    n_pos_s: usize,
    enc: Vec<EncLayer>,
    enc_norm: Norm,
    tgt: Vec<f32>,
    n_tgt: usize,
    pos_t: Vec<f32>,
    n_pos_t: usize,
    dec: Vec<DecLayer>,
    norm: Norm,
    head: Lin,
}

impl Oov {
    pub fn new(get: Getter) -> Result<Self> {
        let rows = |name: &str| -> Result<usize> { Ok(get(name)?.dims()[0]) };
        let (n_src, n_pos_s) = (rows("enc.m.src.weight")?, rows("enc.m.pos_s.weight")?);
        let (n_tgt, n_pos_t) = (rows("dec.m.tgt.weight")?, rows("dec.m.pos_t.weight")?);
        let mut enc = Vec::new();
        for i in 0..3 {
            let (w, b) = (format!("enc.enc_layers.{i}"), format!("enc.m.enc_layers.{i}"));
            enc.push(EncLayer {
                n1: Norm::new(get, &format!("{b}.n1"), D)?,
                att: Attn::new(get, &format!("{w}.att"), &format!("{b}.att"))?,
                n2: Norm::new(get, &format!("{b}.n2"), D)?,
                ff: Ff::new(get, &format!("{w}.ff"), &format!("{b}.ff"))?,
            });
        }
        let mut dec = Vec::new();
        for i in 0..3 {
            let (w, b) = (format!("dec.dec_layers.{i}"), format!("dec.m.dec_layers.{i}"));
            dec.push(DecLayer {
                n1: Norm::new(get, &format!("{b}.n1"), D)?,
                att: Attn::new(get, &format!("{w}.att"), &format!("{b}.att"))?,
                n2: Norm::new(get, &format!("{b}.n2"), D)?,
                cross: Attn::new(get, &format!("{w}.cross"), &format!("{b}.cross"))?,
                n3: Norm::new(get, &format!("{b}.n3"), D)?,
                ff: Ff::new(get, &format!("{w}.ff"), &format!("{b}.ff"))?,
            });
        }
        Ok(Self {
            src: load(get, "enc.m.src.weight", &[n_src, D])?,
            n_src,
            pos_s: load(get, "enc.m.pos_s.weight", &[n_pos_s, D])?,
            n_pos_s,
            enc,
            enc_norm: Norm::new(get, "enc.m.enc_norm", D)?,
            tgt: load(get, "dec.m.tgt.weight", &[n_tgt, D])?,
            n_tgt,
            pos_t: load(get, "dec.m.pos_t.weight", &[n_pos_t, D])?,
            n_pos_t,
            dec,
            norm: Norm::new(get, "dec.m.norm", D)?,
            head: Lin::new(get, "dec.head.MatMul.w1", "dec.m.head.bias", D, n_tgt)?,
        })
    }

    fn encode(&self, ids: &[usize]) -> Vec<Vec<f32>> {
        let mut x: Vec<Vec<f32>> = ids
            .iter()
            .enumerate()
            .map(|(p, &c)| {
                let c = c.min(self.n_src - 1);
                let p = p.min(self.n_pos_s - 1);
                (0..D).map(|d| self.src[c * D + d] + self.pos_s[p * D + d]).collect()
            })
            .collect();
        for l in &self.enc {
            let h: Vec<Vec<f32>> = x.iter().map(|r| l.n1.apply(r)).collect();
            let qs: Vec<Vec<f32>> = h.iter().map(|r| l.att.q.apply(r)).collect();
            let ks: Vec<Vec<f32>> = h.iter().map(|r| l.att.k.apply(r)).collect();
            let vs: Vec<Vec<f32>> = h.iter().map(|r| l.att.v.apply(r)).collect();
            for (i, q) in qs.iter().enumerate() {
                let a = l.att.attend(q, &ks, &vs);
                add(&mut x[i], &a);
            }
            for r in x.iter_mut() {
                let f = l.ff.apply(&l.n2.apply(r));
                add(r, &f);
            }
        }
        x.iter().map(|r| self.enc_norm.apply(r)).collect()
    }

    /// Жадное декодирование: id символов слова (с BOS=2 и EOS=1) → id выходных символов до EOS/PAD.
    pub fn decode(&self, ids: &[usize], max_out: usize) -> Vec<usize> {
        let mem = self.encode(ids);
        let cross_kv: Vec<(Vec<Vec<f32>>, Vec<Vec<f32>>)> = self
            .dec
            .iter()
            .map(|l| (mem.iter().map(|m| l.cross.k.apply(m)).collect(), mem.iter().map(|m| l.cross.v.apply(m)).collect()))
            .collect();
        let mut kc: Vec<Vec<Vec<f32>>> = vec![Vec::new(); self.dec.len()];
        let mut vc: Vec<Vec<Vec<f32>>> = vec![Vec::new(); self.dec.len()];
        let mut out = Vec::new();
        let mut tok = 2usize; // BOS
        for pos in 0..max_out {
            let t = tok.min(self.n_tgt - 1);
            let p = pos.min(self.n_pos_t - 1);
            let mut x: Vec<f32> = (0..D).map(|d| self.tgt[t * D + d] + self.pos_t[p * D + d]).collect();
            for (li, l) in self.dec.iter().enumerate() {
                let h = l.n1.apply(&x);
                let q = l.att.q.apply(&h);
                kc[li].push(l.att.k.apply(&h));
                vc[li].push(l.att.v.apply(&h));
                let a = l.att.attend(&q, &kc[li], &vc[li]);
                add(&mut x, &a);
                let h = l.n2.apply(&x);
                let q = l.cross.q.apply(&h);
                let a = l.cross.attend(&q, &cross_kv[li].0, &cross_kv[li].1);
                add(&mut x, &a);
                let f = l.ff.apply(&l.n3.apply(&x));
                add(&mut x, &f);
            }
            let logits = self.head.apply(&self.norm.apply(&x));
            let mut best = 0;
            for (i, &v) in logits.iter().enumerate() {
                if v > logits[best] {
                    best = i;
                }
            }
            out.push(best);
            if best <= 1 {
                break;
            }
            tok = best;
        }
        out
    }
}

/// Контекстная модель: по признакам слов клаузы — логиты (слот, край) и «писать слитно со следующим».
pub struct Context {
    embs: Vec<(Vec<f32>, usize, usize)>,
    inp: Lin,
    convs: Vec<(Vec<f32>, Vec<f32>, usize)>,
    form: Lin,
    join: Lin,
}

pub const N_FEATURES: usize = 14;
const C: usize = 128;
const DIL: [usize; 4] = [1, 2, 4, 1];

impl Context {
    pub fn new(get: Getter) -> Result<Self> {
        let mut embs = Vec::new();
        let mut din = 0;
        for i in 0..N_FEATURES {
            let name = format!("context.embs.{i}.weight");
            let t = get(&name)?;
            let (rows, dim) = (t.dims()[0], t.dims()[1]);
            embs.push((t.flatten_all()?.to_vec1::<f32>()?, rows, dim));
            din += dim;
        }
        let mut convs = Vec::new();
        for i in 0..8 {
            let k = if i % 2 == 0 { 3 } else { 1 };
            convs.push((
                load(get, &format!("context.convs.{i}.weight"), &[C, C, k])?,
                load(get, &format!("context.convs.{i}.bias"), &[C])?,
                k,
            ));
        }
        let form_out = get("context.form.bias")?.dims()[0];
        Ok(Self {
            embs,
            inp: Lin::new(get, "context.inp.MatMul.w1", "context.inp.bias", din, C)?,
            convs,
            form: Lin::new(get, "context.form.MatMul.w1", "context.form.bias", C, form_out)?,
            join: Lin::new(get, "context.join.MatMul.w1", "context.join.bias", C, 2)?,
        })
    }

    /// Свёртка по времени: x [n][C] → [n][C], ядро k, дилатация dil, нулевые поля.
    fn conv(&self, x: &[Vec<f32>], idx: usize, dil: usize) -> Vec<Vec<f32>> {
        let (w, b, k) = &self.convs[idx];
        let n = x.len();
        let pad = (k - 1) / 2 * dil;
        (0..n)
            .map(|t| {
                (0..C)
                    .map(|o| {
                        let mut s = 0f64;
                        for i in 0..C {
                            for j in 0..*k {
                                let src = t as isize + (j * dil) as isize - pad as isize;
                                if src < 0 || src >= n as isize {
                                    continue;
                                }
                                s += w[(o * C + i) * k + j] as f64 * x[src as usize][i] as f64;
                            }
                        }
                        (s as f32) + b[o]
                    })
                    .collect()
            })
            .collect()
    }

    /// feats [n][14] → (form [n][52], join [n][2]).
    pub fn run(&self, feats: &[[i64; N_FEATURES]]) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let mask: Vec<f32> = feats.iter().map(|f| if f[0] > 0 { 1.0 } else { 0.0 }).collect();
        let mut x: Vec<Vec<f32>> = feats
            .iter()
            .enumerate()
            .map(|(t, f)| {
                let mut e = Vec::new();
                for (k, (tab, rows, dim)) in self.embs.iter().enumerate() {
                    let r = (f[k].max(0) as usize).min(rows - 1);
                    e.extend_from_slice(&tab[r * dim..(r + 1) * dim]);
                }
                self.inp.apply(&e).into_iter().map(|v| v * mask[t]).collect()
            })
            .collect();
        for (blk, &dil) in DIL.iter().enumerate() {
            let h: Vec<Vec<f32>> = x.iter().map(|r| r.iter().map(|v| v.max(0.0)).collect()).collect();
            let h: Vec<Vec<f32>> =
                self.conv(&h, 2 * blk, dil).into_iter().map(|r| r.into_iter().map(|v| v.max(0.0)).collect()).collect();
            let h = self.conv(&h, 2 * blk + 1, 1);
            for (t, r) in x.iter_mut().enumerate() {
                for (a, b) in r.iter_mut().zip(&h[t]) {
                    *a += *b * mask[t];
                }
            }
        }
        let out: Vec<Vec<f32>> = x.iter().map(|r| r.iter().map(|v| v.max(0.0)).collect()).collect();
        (out.iter().map(|r| self.form.apply(r)).collect(), out.iter().map(|r| self.join.apply(r)).collect())
    }
}
