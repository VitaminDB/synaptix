use std::path::PathBuf;
use std::time::Instant;

use synaptix::facade::llm::{GenerationOptions, MediaKind, Message};

use super::engine::{EngineCmd, Turn, TurnStats};
use crate::commands::llm_facade::{entropy_seed, Part};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

pub struct Msg {
    pub role: Role,
    pub text: String,
    pub reasoning: String,
    pub tool: String,
    pub attachments: Vec<String>,
    prompt_prefix: String,
}

impl Msg {
    fn new(role: Role, text: String) -> Self {
        Self { role, text, reasoning: String::new(), tool: String::new(), attachments: Vec::new(), prompt_prefix: String::new() }
    }
}

pub struct Settings {
    pub opts: GenerationOptions,
    pub thinking: bool,
    pub effort: Option<String>,
    pub preset: String,
}

pub struct App {
    pub messages: Vec<Msg>,
    pub input: String,
    pub cursor: usize,
    pub scroll: u16,
    pub follow: bool,
    pub generating: bool,
    pub busy_attach: bool,
    pub should_quit: bool,
    pub status: String,
    pub spinner: usize,
    pub arch_label: String,
    pub model_label: String,
    pub settings: Settings,
    pending_prefix: String,
    pending_labels: Vec<String>,
    system: Option<String>,
    turn_start: Option<Instant>,
}

impl App {
    pub fn new(system: Option<String>, arch_label: String, model_label: String, settings: Settings) -> Self {
        let mut app = Self {
            messages: Vec::new(),
            input: String::new(),
            cursor: 0,
            scroll: 0,
            follow: true,
            generating: false,
            busy_attach: false,
            should_quit: false,
            status: "готово".into(),
            spinner: 0,
            arch_label,
            model_label,
            settings,
            pending_prefix: String::new(),
            pending_labels: Vec::new(),
            system,
            turn_start: None,
        };
        app.seed_system();
        app
    }

    fn seed_system(&mut self) {
        if let Some(sys) = self.system.clone() {
            self.messages.push(Msg::new(Role::System, sys));
        }
    }

    pub fn reset(&mut self) {
        self.messages.clear();
        self.pending_prefix.clear();
        self.pending_labels.clear();
        self.seed_system();
        self.status = "история очищена".into();
        self.scroll = 0;
        self.follow = true;
    }

    pub fn attach_cmd(path: &str, kind: MediaKind) -> Option<EngineCmd> {
        let path = path.trim();
        if path.is_empty() {
            return None;
        }
        Some(EngineCmd::Attach { path: PathBuf::from(path), kind })
    }

    pub fn attached(&mut self, block: &str, tokens: usize, label: &str) {
        self.busy_attach = false;
        self.pending_prefix.push_str(block);
        self.pending_labels.push(format!("{label} ({tokens} ток.)"));
        self.status = format!("вложение: {label} — {tokens} токенов; напишите сообщение");
    }

    fn history(&self) -> Vec<Message> {
        self.messages
            .iter()
            .filter(|m| !(m.role == Role::Assistant && m.text.trim().is_empty() && m.tool.is_empty()))
            .map(|m| match m.role {
                Role::System => Message::system(m.text.clone()),
                Role::User => Message::user(format!("{}{}", m.prompt_prefix, m.text)),
                Role::Assistant => Message::assistant(m.text.trim().to_string()),
            })
            .collect()
    }

    pub fn submit(&mut self) -> Option<EngineCmd> {
        let text = self.input.trim().to_string();
        self.input.clear();
        self.cursor = 0;
        if text.is_empty() {
            return None;
        }
        if let Some(cmd) = self.command(&text) {
            return cmd;
        }
        let mut msg = Msg::new(Role::User, text);
        msg.prompt_prefix = std::mem::take(&mut self.pending_prefix);
        msg.attachments = std::mem::take(&mut self.pending_labels);
        self.messages.push(msg);
        let messages = self.history();
        self.messages.push(Msg::new(Role::Assistant, String::new()));
        self.generating = true;
        self.turn_start = Some(Instant::now());
        self.status = "генерация…".into();
        self.follow = true;
        let mut opts = self.settings.opts.clone();
        if opts.seed == 0 {
            opts.seed = entropy_seed();
        }
        Some(EngineCmd::Generate(Box::new(Turn {
            messages,
            opts,
            thinking: self.settings.thinking,
            effort: self.settings.effort.clone(),
        })))
    }

    fn command(&mut self, text: &str) -> Option<Option<EngineCmd>> {
        let (head, arg) = text.split_once(' ').unwrap_or((text, ""));
        let r = match head {
            "/quit" | "/exit" => {
                self.should_quit = true;
                None
            }
            "/reset" => {
                self.reset();
                Some(EngineCmd::Reset)
            }
            "/image" | "/video" => {
                let kind = if head == "/image" { MediaKind::Image } else { MediaKind::Video };
                let cmd = Self::attach_cmd(arg, kind);
                if cmd.is_some() {
                    self.busy_attach = true;
                    self.status = format!("кодирую {arg}…");
                } else {
                    self.status = format!("{head} <путь>");
                }
                cmd
            }
            "/think" => {
                self.settings.thinking = !matches!(arg.trim(), "off" | "0" | "no");
                self.status = format!("размышления: {}", if self.settings.thinking { "вкл" } else { "выкл" });
                None
            }
            "/effort" => {
                self.settings.effort = Some(arg.trim().to_string()).filter(|s| !s.is_empty());
                self.status = format!("глубина размышлений: {}", self.settings.effort.as_deref().unwrap_or("по шаблону"));
                None
            }
            "/temp" => {
                if let Ok(v) = arg.trim().parse() {
                    self.settings.opts.temperature = v;
                }
                self.status = format!("temperature {}", self.settings.opts.temperature);
                None
            }
            _ if head.starts_with('/') => {
                self.status = "команды: /image, /video, /think on|off, /effort <уровень>, /temp <t>, /reset, /quit".into();
                None
            }
            _ => return None,
        };
        Some(r)
    }

    pub fn push_delta(&mut self, part: Part, delta: &str) {
        if let Some(m) = self.messages.last_mut() {
            if m.role == Role::Assistant {
                match part {
                    Part::Answer => m.text.push_str(delta),
                    Part::Reasoning => m.reasoning.push_str(delta),
                    Part::Tool => m.tool.push_str(delta),
                }
            }
        }
        self.follow = true;
    }

    pub fn finish(&mut self, s: &TurnStats) {
        if let Some(m) = self.messages.last_mut() {
            if m.role == Role::Assistant {
                m.text = m.text.trim().to_string();
                m.reasoning = m.reasoning.trim().to_string();
            }
        }
        self.generating = false;
        let prefilled = s.prompt_tokens.saturating_sub(s.cached);
        let prefill_tps = if s.prefill_ms > 0 { prefilled as f32 / (s.prefill_ms as f32 / 1000.0) } else { 0.0 };
        let decode = s.new_tokens.saturating_sub(1);
        let decode_tps = if s.decode_ms > 0 && decode > 0 { decode as f32 / (s.decode_ms as f32 / 1000.0) } else { 0.0 };
        self.status = format!(
            "ctx {}/{} • cached {} • prefill {prefill_tps:.0} tok/s • decode {decode_tps:.1} tok/s",
            s.prompt_tokens + s.new_tokens,
            s.ctx_max,
            s.cached
        );
    }

    pub fn fail(&mut self, err: &str) {
        self.busy_attach = false;
        if self.generating {
            if let Some(m) = self.messages.last_mut() {
                if m.role == Role::Assistant && m.text.is_empty() {
                    m.text.push_str(&format!("[ошибка: {err}]"));
                }
            }
        }
        self.generating = false;
        self.status = format!("ошибка: {err}");
    }

    pub fn tick_spinner(&mut self) {
        self.spinner = self.spinner.wrapping_add(1);
    }
}
