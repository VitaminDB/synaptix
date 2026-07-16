use ratatui::layout::{Constraint, Layout, Position};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Paragraph, Wrap};
use ratatui::Frame;

use super::app::{App, Role};

const SPINNER: [&str; 8] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧"];

pub fn draw(frame: &mut Frame, app: &mut App) {
    let chunks = Layout::vertical([
        Constraint::Min(1),
        Constraint::Length(3),
        Constraint::Length(1),
    ])
    .split(frame.area());

    draw_transcript(frame, app, chunks[0]);
    draw_input(frame, app, chunks[1]);
    draw_status(frame, app, chunks[2]);
}

fn draw_transcript(frame: &mut Frame, app: &mut App, area: ratatui::layout::Rect) {
    let block = Block::bordered().title(" чат ");
    let inner_h = block.inner(area).height;
    let lines = build_lines(app);
    let para = Paragraph::new(lines).block(block).wrap(Wrap { trim: false });

    let total = para.line_count(area.width) as u16;
    let max_scroll = total.saturating_sub(inner_h);
    if app.follow {
        app.scroll = max_scroll;
    } else if app.scroll > max_scroll {
        app.scroll = max_scroll;
    }

    frame.render_widget(para.scroll((app.scroll, 0)), area);
}

fn build_lines(app: &App) -> Vec<Line<'static>> {
    let mut lines: Vec<Line<'static>> = Vec::new();
    for m in &app.messages {
        let (label, color) = match m.role {
            Role::System => ("система", Color::DarkGray),
            Role::User => ("вы", Color::Cyan),
            Role::Assistant => ("модель", Color::Green),
        };
        lines.push(Line::from(Span::styled(
            format!("{label}:"),
            Style::default().fg(color).add_modifier(Modifier::BOLD),
        )));
        if m.text.is_empty() {
            lines.push(Line::from(""));
        } else {
            for seg in m.text.split('\n') {
                lines.push(Line::from(seg.to_string()));
            }
        }
        lines.push(Line::from(""));
    }
    lines
}

fn draw_input(frame: &mut Frame, app: &App, area: ratatui::layout::Rect) {
    let title = if app.generating {
        " генерация… (Esc — отмена) "
    } else {
        " сообщение (Enter — отправить, /reset, /quit) "
    };
    let block = Block::bordered().title(title);
    let inner = block.inner(area);
    let before = &app.input[..app.cursor.min(app.input.len())];
    let col = before.chars().count() as u16;
    let scroll_x = col.saturating_sub(inner.width.saturating_sub(1));
    let para = Paragraph::new(app.input.as_str()).block(block).scroll((0, scroll_x));
    frame.render_widget(para, area);
    if !app.generating {
        let cx = inner.x + col.saturating_sub(scroll_x);
        frame.set_cursor_position(Position::new(cx, inner.y));
    }
}

fn draw_status(frame: &mut Frame, app: &App, area: ratatui::layout::Rect) {
    let spin = if app.generating {
        SPINNER[app.spinner % SPINNER.len()]
    } else {
        "•"
    };
    let text = format!(
        " {spin} {} · {} · t={:.2} top_k={} top_p={:.2} · {}",
        app.model_label, app.arch_label, app.cfg.temperature, app.cfg.top_k, app.cfg.top_p, app.status
    );
    let para = Paragraph::new(text).style(Style::default().fg(Color::DarkGray));
    frame.render_widget(para, area);
}
