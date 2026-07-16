use std::time::Instant;

use synaptix_llm_common::{GenerationConfig, GenerationStats};

use super::template::Prompt;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

pub struct Msg {
    pub role: Role,
    pub text: String,
}

pub enum Submit {
    Generate { prompt: String, cfg: GenerationConfig },
    Reset,
    None,
}

pub struct App {
    pub messages: Vec<Msg>,
    pub input: String,
    pub cursor: usize,
    pub scroll: u16,
    pub follow: bool,
    pub generating: bool,
    pub should_quit: bool,
    pub status: String,
    pub spinner: usize,
    pub arch_label: String,
    pub model_label: String,
    pub cfg: GenerationConfig,
    prompt: Prompt,
    system: Option<String>,
    turn_start: Option<Instant>,
}

impl App {
    pub fn new(
        prompt: Prompt,
        system: Option<String>,
        arch_label: String,
        model_label: String,
        cfg: GenerationConfig,
    ) -> Self {
        let mut app = Self {
            messages: Vec::new(),
            input: String::new(),
            cursor: 0,
            scroll: 0,
            follow: true,
            generating: false,
            should_quit: false,
            status: "готово".into(),
            spinner: 0,
            arch_label,
            model_label,
            cfg,
            prompt,
            system,
            turn_start: None,
        };
        app.seed_system();
        app
    }

    fn seed_system(&mut self) {
        if let Some(sys) = self.system.clone() {
            self.messages.push(Msg { role: Role::System, text: sys });
        }
    }

    pub fn reset(&mut self) {
        self.messages.clear();
        self.seed_system();
        self.status = "история очищена".into();
        self.scroll = 0;
        self.follow = true;
    }

    pub fn submit(&mut self) -> Submit {
        let text = self.input.trim().to_string();
        self.input.clear();
        self.cursor = 0;
        if text.is_empty() {
            return Submit::None;
        }
        match text.as_str() {
            "/quit" | "/exit" => {
                self.should_quit = true;
                return Submit::None;
            }
            "/reset" => {
                self.reset();
                return Submit::Reset;
            }
            _ => {}
        }
        // Полный рендер истории каждый ход (jinja корректно стрипает прошлый
        // reasoning по шаблону Qwen). Токен-дельта/prefix-кэш для hybrid+thinking
        // невозможны корректно: KV содержит think-токены, которые шаблон убирает из
        // истории → пере-префиллим всю историю (OOM ограничен чанкованием prefill).
        self.messages.push(Msg { role: Role::User, text });
        let prompt = match self.prompt.render(&self.messages, true) {
            Ok(p) => p,
            Err(e) => {
                self.status = format!("ошибка chat-template: {e}");
                self.messages.pop();
                return Submit::None;
            }
        };
        self.messages.push(Msg { role: Role::Assistant, text: String::new() });
        self.generating = true;
        self.turn_start = Some(Instant::now());
        self.status = "генерация…".into();
        self.follow = true;
        let mut cfg = self.cfg.clone();
        if cfg.seed == 0 {
            cfg.seed = time_seed();
        }
        Submit::Generate { prompt, cfg }
    }

    pub fn push_delta(&mut self, delta: &str) {
        if let Some(m) = self.messages.last_mut() {
            if m.role == Role::Assistant {
                m.text.push_str(delta);
            }
        }
        self.follow = true;
    }

    pub fn finish(&mut self, stats: &GenerationStats, cached: usize, ctx_used: usize, ctx_max: usize) {
        if let Some(m) = self.messages.last_mut() {
            if m.role == Role::Assistant {
                let trimmed = m.text.trim_end().to_string();
                m.text = trimmed;
            }
        }
        self.generating = false;
        let prefilled = stats.prompt_tokens.saturating_sub(cached);
        let prefill_tps = if stats.prefill_ms > 0 {
            prefilled as f32 / (stats.prefill_ms as f32 / 1000.0)
        } else {
            0.0
        };
        let decode = stats.new_tokens.saturating_sub(1);
        let decode_tps = if stats.decode_ms > 0 && decode > 0 {
            decode as f32 / (stats.decode_ms as f32 / 1000.0)
        } else {
            0.0
        };
        self.status = format!(
            "ctx {ctx_used}/{ctx_max} • cached {cached} • prefill {prefill_tps:.0} tok/s • decode {decode_tps:.1} tok/s"
        );
    }

    pub fn fail(&mut self, err: &str) {
        if let Some(m) = self.messages.last_mut() {
            if m.role == Role::Assistant && m.text.is_empty() {
                m.text.push_str(&format!("[ошибка: {err}]"));
            }
        }
        self.generating = false;
        self.status = format!("ошибка: {err}");
    }

    pub fn tick_spinner(&mut self) {
        self.spinner = self.spinner.wrapping_add(1);
    }
}

pub fn time_seed() -> u64 {
    use std::time::SystemTime;
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(1)
        | 1
}
