//! Компенсация ошибки квантования (GPTQ, блочный вариант): после округления
//! блока столбцов его ошибка гасится поправкой ещё не округлённых столбцов
//! так, чтобы ошибка ВЫХОДА слоя `‖(W − Q)·x‖²` была минимальной.
//!
//! Вход — матрица вторых моментов входа слоя `H = E[x·xᵀ]` `[k, k]`. Блок —
//! единица кодировщика (у IQ3_S-128 — 128 столбцов): внутри блока значения
//! кодируются совместно, поправка идёт только на следующие блоки.
//!
//! Математика: `H⁻¹ = UᵀU` (U — верхняя треугольная). После блока `b` с
//! ошибкой `E = W_b − Q_b` остаток строки правится как
//! `W_rest −= E · U_bb⁻¹ · U_b,rest`. Здесь считаются матрицы переноса
//! `T_b = U_bb⁻¹ · U_b,rest` — одни на все матрицы с этим входом.
//!
//! Если у матрицы свой масштаб входов (`H_e = D·H·D`, `D = diag(d)`), перенос
//! тот же с поправкой: `W_rest −= ((E ⊙ d_b) · T_b) ⊘ d_rest`.

use rayon::prelude::*;

/// Матрицы переноса по блокам: `out[b]` — `[block, k − (b+1)·block]`
/// построчно, для последнего блока пусто. `damp` — доля среднего диагонали,
/// добавляемая к диагонали (0,01 — обычное значение).
pub fn block_transfers(h: &[f32], k: usize, block: usize, damp: f64) -> Result<Vec<Vec<f32>>, String> {
    if h.len() != k * k || block == 0 || k % block != 0 {
        return Err(format!("gptq: H {} значений, k={k}, блок {block}", h.len()));
    }
    // PHP: обратный порядок столбцов; симметризация и демпфирование.
    let rev = |i: usize| k - 1 - i;
    let mut a = vec![0f64; k * k];
    let mut mean = 0f64;
    for i in 0..k {
        mean += h[i * k + i] as f64;
    }
    mean /= k as f64;
    if !(mean > 0.0) {
        return Err("gptq: нулевая диагональ H".into());
    }
    a.par_chunks_mut(k).enumerate().for_each(|(i, row)| {
        for j in 0..k {
            row[j] = 0.5 * (h[rev(i) * k + rev(j)] as f64 + h[rev(j) * k + rev(i)] as f64);
        }
        // Мёртвый вход (нулевая диагональ) — единичная строка: его веса
        // округляются как есть и никуда не переносятся.
        if row[i] <= 0.0 {
            row.fill(0.0);
            row[i] = mean;
        }
        row[i] += damp * mean;
    });
    // Холецкий `A = L·Lᵀ`, L — нижняя, на месте (построчно, скалярные произведения строк).
    for i in 0..k {
        let (head, tail) = a.split_at_mut(i * k);
        let row = &mut tail[..k];
        for j in 0..=i {
            let lj = if j < i { &head[j * k..j * k + j] } else { &[][..] };
            let s: f64 = if j < i { row[..j].iter().zip(lj).map(|(x, y)| x * y).sum() } else { row[..j].iter().map(|x| x * x).sum() };
            if j < i {
                row[j] = (row[j] - s) / head[j * k + j];
            } else {
                let d = row[i] - s;
                if !(d > 0.0) {
                    return Err(format!("gptq: H не положительно определена (столбец {i})"));
                }
                row[i] = d.sqrt();
            }
        }
        row[i + 1..].fill(0.0);
    }
    // L⁻¹ (нижняя): столбец j — решение L·y = e_j.
    let mut linv = vec![0f64; k * k];
    let cols: Vec<Vec<f64>> = (0..k)
        .into_par_iter()
        .map(|j| {
            let mut y = vec![0f64; k];
            y[j] = 1.0 / a[j * k + j];
            for i in j + 1..k {
                let s: f64 = a[i * k + j..i * k + i].iter().zip(&y[j..i]).map(|(l, v)| l * v).sum();
                y[i] = -s / a[i * k + i];
            }
            y
        })
        .collect();
    for (j, y) in cols.iter().enumerate() {
        for i in j..k {
            linv[i * k + j] = y[i];
        }
    }
    drop(a);
    // U = P·L⁻¹·P — верхняя, `H⁻¹ = UᵀU`.
    let u = |i: usize, j: usize| linv[rev(i) * k + rev(j)];
    let nb = k / block;
    let out: Vec<Vec<f32>> = (0..nb)
        .into_par_iter()
        .map(|b| {
            let c0 = (b + 1) * block;
            let rest = k - c0;
            if rest == 0 {
                return Vec::new();
            }
            // Решаем U_bb · X = U_b,rest обратной подстановкой по строкам снизу вверх.
            let r0 = b * block;
            let mut x = vec![0f64; block * rest];
            for i in (0..block).rev() {
                let (above, below) = x.split_at_mut((i + 1) * rest);
                let row = &mut above[i * rest..];
                for (c, v) in row.iter_mut().enumerate() {
                    *v = u(r0 + i, c0 + c);
                }
                for j in i + 1..block {
                    let f = u(r0 + i, r0 + j);
                    if f != 0.0 {
                        let src = &below[(j - i - 1) * rest..(j - i) * rest];
                        for (v, s) in row.iter_mut().zip(src) {
                            *v -= f * s;
                        }
                    }
                }
                let d = u(r0 + i, r0 + i);
                for v in row.iter_mut() {
                    *v /= d;
                }
            }
            x.iter().map(|v| *v as f32).collect()
        })
        .collect();
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Перенос обязан давать минимум ошибки выхода: для ошибки `E` первого
    /// блока поправка остатка `−E·T_0` совпадает с решением наименьших
    /// квадратов `−E·H_b,r·H_rr⁻¹`, а ошибка выхода с поправкой меньше, чем без.
    #[test]
    fn transfer_minimizes_output_error() {
        let (k, block, n) = (12usize, 4usize, 64usize);
        let mut s = 5u64;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f64 / (1u64 << 24) as f64 - 0.5
        };
        // Коррелированные входы: x = z·M.
        let m: Vec<f64> = (0..k * k).map(|_| rnd()).collect();
        let xs: Vec<Vec<f64>> = (0..n)
            .map(|_| {
                let z: Vec<f64> = (0..k).map(|_| rnd()).collect();
                (0..k).map(|j| (0..k).map(|i| z[i] * m[i * k + j]).sum()).collect()
            })
            .collect();
        let mut h = vec![0f32; k * k];
        for x in &xs {
            for i in 0..k {
                for j in 0..k {
                    h[i * k + j] += (x[i] * x[j] / n as f64) as f32;
                }
            }
        }
        let t = block_transfers(&h, k, block, 1e-6).unwrap();
        assert_eq!(t[0].len(), block * (k - block));
        assert!(t[2].is_empty());
        let e: Vec<f64> = (0..block).map(|_| rnd()).collect();
        // Ошибка веса: первый блок = −e (Q − W), остаток = поправка.
        let out_err = |delta: &[f64]| -> f64 { xs.iter().map(|x| x.iter().zip(delta).map(|(a, b)| a * b).sum::<f64>().powi(2)).sum() };
        let mut plain = vec![0f64; k];
        let mut comp = vec![0f64; k];
        for i in 0..block {
            plain[i] = -e[i];
            comp[i] = -e[i];
        }
        for c in 0..k - block {
            // W_rest −= E·T ⇒ отклонение остатка от исходного = −E·T.
            comp[block + c] = -(0..block).map(|i| e[i] * t[0][i * (k - block) + c] as f64).sum::<f64>();
        }
        let (ep, ec) = (out_err(&plain), out_err(&comp));
        assert!(ec < ep * 0.9, "с поправкой {ec}, без {ep}");
        // Малое шевеление поправки ошибку не уменьшает (это минимум).
        for c in 0..k - block {
            for d in [1e-3, -1e-3] {
                let mut p = comp.clone();
                p[block + c] += d;
                assert!(out_err(&p) >= ec * (1.0 - 1e-4), "не минимум по столбцу {c}");
            }
        }
    }
}
