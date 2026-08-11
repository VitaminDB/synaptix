use crate::preprocess::ImageGrid;

pub const VISION_START: u32 = 151652;
pub const VISION_END: u32 = 151653;
pub const IMAGE_PAD: u32 = 151655;
pub const VIDEO_PAD: u32 = 151656;
pub const PAD: u32 = 151643;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenTag {
    Video = 0,
    Text = 1,
}

#[derive(Debug, Clone)]
pub enum PresentationItem {
    Text(String),
    Vision { grid: ImageGrid, merge: usize, video: bool },
}

impl PresentationItem {
    pub fn vision_tokens(&self) -> usize {
        match self {
            PresentationItem::Vision { grid, merge, .. } => grid.tokens(*merge),
            PresentationItem::Text(_) => 0,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct H3Presentation {
    pub items: Vec<PresentationItem>,
}

impl H3Presentation {
    pub fn t2va(prompt: &str) -> Self {
        Self { items: vec![PresentationItem::Text(prompt.to_string())] }
    }

    pub fn fl2va(prompt: &str, keyframes: &[ImageGrid], merge: usize) -> Self {
        let mut items = Vec::with_capacity(keyframes.len() * 2 + 1);
        for (i, g) in keyframes.iter().enumerate() {
            items.push(PresentationItem::Text(format!("<Picture {}>: ", i + 1)));
            items.push(PresentationItem::Vision { grid: *g, merge, video: false });
        }
        items.push(PresentationItem::Text(prompt.to_string()));
        Self { items }
    }

    pub fn ref2va(prompt: &str, refs: &[RefItem], merge: usize) -> Self {
        let mut items = Vec::new();
        let (mut n_img, mut n_vid, mut n_aud) = (0usize, 0usize, 0usize);
        for r in refs {
            match r {
                RefItem::Image { grid } => {
                    n_img += 1;
                    items.push(PresentationItem::Text(format!("<Picture {n_img}>: ")));
                    items.push(PresentationItem::Vision { grid: *grid, merge, video: false });
                }
                RefItem::Audio => {
                    n_aud += 1;
                    items.push(PresentationItem::Text(format!("<Audio {n_aud}>: ")));
                }
                RefItem::Video { blocks } => {
                    n_vid += 1;
                    items.push(PresentationItem::Text(format!("<Video {n_vid}>: ")));
                    for b in blocks {
                        items.push(PresentationItem::Text(format!("<{:.1} seconds>", b.seconds)));
                        items.push(PresentationItem::Vision {
                            grid: b.grid,
                            merge,
                            video: true,
                        });
                    }
                }
            }
        }
        items.push(PresentationItem::Text(prompt.to_string()));
        Self { items }
    }

    pub fn text_chunks(&self) -> Vec<&str> {
        self.items
            .iter()
            .filter_map(|i| match i {
                PresentationItem::Text(s) => Some(s.as_str()),
                _ => None,
            })
            .collect()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct VideoBlock {
    pub grid: ImageGrid,
    pub seconds: f32,
}

#[derive(Debug, Clone)]
pub enum RefItem {
    Image { grid: ImageGrid },
    Audio,
    Video { blocks: Vec<VideoBlock> },
}

#[derive(Debug, Clone)]
pub struct EncodedPresentation {
    pub ids: Vec<u32>,
    pub tags: Vec<u8>,
    pub vision_rows: Vec<Vec<usize>>,
    pub vision_grids: Vec<ImageGrid>,
}

impl EncodedPresentation {
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    pub fn all_vision_rows(&self) -> Vec<usize> {
        self.vision_rows.iter().flatten().copied().collect()
    }
}

pub fn assemble(
    presentation: &H3Presentation,
    mut tokenize: impl FnMut(&str) -> Vec<u32>,
) -> EncodedPresentation {
    let mut ids = Vec::new();
    let mut tags = Vec::new();
    let mut vision_rows = Vec::new();
    let mut vision_grids = Vec::new();

    for item in &presentation.items {
        match item {
            PresentationItem::Text(s) => {
                let t = tokenize(s);
                tags.extend(std::iter::repeat_n(TokenTag::Text as u8, t.len()));
                ids.extend(t);
            }
            PresentationItem::Vision { grid, merge, video } => {
                let n = grid.tokens(*merge);
                let pad = if *video { VIDEO_PAD } else { IMAGE_PAD };
                let start = ids.len();
                ids.push(VISION_START);
                let rows: Vec<usize> = (ids.len()..ids.len() + n).collect();
                ids.extend(std::iter::repeat_n(pad, n));
                ids.push(VISION_END);
                tags.extend(std::iter::repeat_n(TokenTag::Video as u8, ids.len() - start));
                vision_rows.push(rows);
                vision_grids.push(*grid);
            }
        }
    }

    EncodedPresentation { ids, tags, vision_rows, vision_grids }
}
