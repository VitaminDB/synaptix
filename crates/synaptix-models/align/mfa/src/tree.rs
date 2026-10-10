//! Дерево контекстов Kaldi (`ContextDependency`, EventMap).

use serde::{Deserialize, Serialize};

use crate::kaldi_text::Tokens;
use crate::{MfaError, Result};

/// Ключ события: −1 — класс состояния HMM, 0..N−1 — позиции контекста.
pub const PDF_CLASS: i32 = -1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventMap {
    Constant(i32),
    Split { key: i32, set: Vec<i32>, yes: Box<EventMap>, no: Box<EventMap> },
    Table { key: i32, table: Vec<Option<EventMap>> },
}

impl EventMap {
    fn parse(t: &mut Tokens) -> Result<Option<EventMap>> {
        match t.next()? {
            "NULL" => Ok(None),
            "CE" => Ok(Some(EventMap::Constant(t.int()? as i32))),
            "SE" => {
                let key = t.int()? as i32;
                t.expect("[")?;
                let mut set = Vec::new();
                loop {
                    let x = t.next()?;
                    if x == "]" {
                        break;
                    }
                    set.push(x.parse::<i32>().map_err(|_| MfaError::Format(format!("SE: `{x}`")))?);
                }
                t.expect("{")?;
                let yes = Self::parse(t)?.ok_or_else(|| MfaError::Format("SE yes = NULL".into()))?;
                let no = Self::parse(t)?.ok_or_else(|| MfaError::Format("SE no = NULL".into()))?;
                t.expect("}")?;
                Ok(Some(EventMap::Split { key, set, yes: Box::new(yes), no: Box::new(no) }))
            }
            "TE" => {
                let key = t.int()? as i32;
                let size = t.int()? as usize;
                t.expect("(")?;
                let mut table = Vec::with_capacity(size);
                for _ in 0..size {
                    table.push(Self::parse(t)?);
                }
                t.expect(")")?;
                Ok(Some(EventMap::Table { key, table }))
            }
            x => Err(MfaError::Format(format!("EventMap: неизвестный тип `{x}`"))),
        }
    }

    /// Ответ для события `(ключ, значение)`.
    pub fn compute(&self, event: &[(i32, i32)]) -> Option<i32> {
        let get = |k: i32| event.iter().find(|(kk, _)| *kk == k).map(|(_, v)| *v);
        match self {
            EventMap::Constant(a) => Some(*a),
            EventMap::Split { key, set, yes, no } => {
                let v = get(*key)?;
                if set.binary_search(&v).is_ok() { yes.compute(event) } else { no.compute(event) }
            }
            EventMap::Table { key, table } => {
                let v = get(*key)?;
                table.get(usize::try_from(v).ok()?)?.as_ref()?.compute(event)
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextDependency {
    pub width: usize,
    pub central: usize,
    pub map: EventMap,
}

impl ContextDependency {
    /// `copy-tree --binary=false`.
    pub fn from_text(src: &str) -> Result<Self> {
        let mut t = Tokens::new(src);
        t.expect("ContextDependency")?;
        let width = t.int()? as usize;
        let central = t.int()? as usize;
        t.expect("ToPdf")?;
        let mut map = EventMap::parse(&mut t)?.ok_or_else(|| MfaError::Format("пустое дерево".into()))?;
        sort_sets(&mut map);
        Ok(Self { width, central, map })
    }

    /// pdf для фонемы в контексте (`ctx` длины `width`, 0 — граница) и класса.
    pub fn pdf(&self, ctx: &[i32], pdf_class: i32) -> Option<usize> {
        let mut ev: Vec<(i32, i32)> = ctx.iter().enumerate().map(|(i, &p)| (i as i32, p)).collect();
        ev.push((PDF_CLASS, pdf_class));
        self.map.compute(&ev).map(|p| p as usize)
    }
}

fn sort_sets(m: &mut EventMap) {
    match m {
        EventMap::Split { set, yes, no, .. } => {
            set.sort_unstable();
            sort_sets(yes);
            sort_sets(no);
        }
        EventMap::Table { table, .. } => {
            for e in table.iter_mut().flatten() {
                sort_sets(e);
            }
        }
        EventMap::Constant(_) => {}
    }
}
