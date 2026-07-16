use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use super::app::{App, Submit};
use super::engine::EngineHandle;

pub fn handle_key(app: &mut App, key: KeyEvent, engine: &EngineHandle) {
    if key.modifiers.contains(KeyModifiers::CONTROL) && matches!(key.code, KeyCode::Char('c')) {
        app.should_quit = true;
        return;
    }

    match key.code {
        KeyCode::Enter => {
            if app.generating {
                return;
            }
            match app.submit() {
                Submit::Generate { prompt, cfg } => engine.generate(prompt, cfg),
                Submit::Reset => engine.reset(),
                Submit::None => {}
            }
        }
        KeyCode::Char(c) => {
            if app.generating {
                return;
            }
            app.input.insert(app.cursor, c);
            app.cursor += c.len_utf8();
        }
        KeyCode::Backspace => {
            if app.generating || app.cursor == 0 {
                return;
            }
            let prev = prev_boundary(&app.input, app.cursor);
            app.input.replace_range(prev..app.cursor, "");
            app.cursor = prev;
        }
        KeyCode::Delete => {
            if app.generating || app.cursor >= app.input.len() {
                return;
            }
            let next = next_boundary(&app.input, app.cursor);
            app.input.replace_range(app.cursor..next, "");
        }
        KeyCode::Left => {
            if app.cursor > 0 {
                app.cursor = prev_boundary(&app.input, app.cursor);
            }
        }
        KeyCode::Right => {
            if app.cursor < app.input.len() {
                app.cursor = next_boundary(&app.input, app.cursor);
            }
        }
        KeyCode::Home => app.cursor = 0,
        KeyCode::End => app.cursor = app.input.len(),
        KeyCode::Up => {
            app.follow = false;
            app.scroll = app.scroll.saturating_sub(1);
        }
        KeyCode::Down => {
            app.scroll = app.scroll.saturating_add(1);
        }
        KeyCode::PageUp => {
            app.follow = false;
            app.scroll = app.scroll.saturating_sub(10);
        }
        KeyCode::PageDown => {
            app.scroll = app.scroll.saturating_add(10);
        }
        KeyCode::Esc => {
            if app.generating {
                engine.cancel();
            }
        }
        _ => {}
    }
}

fn prev_boundary(s: &str, idx: usize) -> usize {
    let mut i = idx - 1;
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

fn next_boundary(s: &str, idx: usize) -> usize {
    let mut i = idx + 1;
    while i < s.len() && !s.is_char_boundary(i) {
        i += 1;
    }
    i
}
