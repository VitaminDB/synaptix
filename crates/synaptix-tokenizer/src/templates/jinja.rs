use std::sync::Arc;

use minijinja::value::Value;
use minijinja::{Environment, Template};

use crate::error::{Result, TokenizerError};
use crate::templates::pycompat;

#[derive(Clone)]
pub struct JinjaEnv {
    inner: Arc<EnvCell>,
}

struct EnvCell {
    env: Environment<'static>,
}

impl JinjaEnv {
    pub fn new() -> Self {
        let mut env = Environment::new();
        // Как jinja2 в transformers: один завершающий `\n` исходника шаблона
        // срезается. Иначе у Llama-3.x (шаблон из GGUF кончается `{%- endif %}\n`)
        // промпт хода получал лишний `\n` после заголовка ассистента, модель
        // видела не тот префикс, а перерендеренная история с ним расходилась —
        // префикс-KV не переиспользовался ни на одном ходу.
        env.set_keep_trailing_newline(false);
        env.set_trim_blocks(false);
        env.set_lstrip_blocks(false);
        pycompat::register_all(&mut env);
        Self { inner: Arc::new(EnvCell { env }) }
    }

    pub fn render(&self, template_src: &str, ctx: Value) -> Result<String> {
        let mut env = self.inner.env.clone();
        let prepared = pycompat::preprocess(template_src);
        env.add_template_owned("__synaptix_tokenizer_template", prepared)
            .map_err(|e| TokenizerError::template_syntax(format!("{e:#}")))?;
        let tmpl: Template<'_, '_> = env
            .get_template("__synaptix_tokenizer_template")
            .map_err(|e| TokenizerError::template(format!("{e:#}")))?;
        tmpl.render(ctx).map_err(|e| TokenizerError::template(format!("{e:#}")))
    }

    pub fn render_named(&self, name: &str, ctx: Value) -> Result<String> {
        let env = &self.inner.env;
        let tmpl = env
            .get_template(name)
            .map_err(|e| TokenizerError::template(format!("{e:#}")))?;
        tmpl.render(ctx).map_err(|e| TokenizerError::template(format!("{e:#}")))
    }

    pub fn register_template(&mut self, name: &'static str, source: String) -> Result<()> {
        let prepared = pycompat::preprocess(&source);
        let cell = Arc::make_mut(&mut self.inner);
        cell.env
            .add_template_owned(name, prepared)
            .map_err(|e| TokenizerError::template_syntax(format!("{e:#}")))?;
        Ok(())
    }
}

impl Default for JinjaEnv {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for EnvCell {
    fn clone(&self) -> Self {
        Self { env: self.env.clone() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use minijinja::context;

    #[test]
    fn basic_render() {
        let env = JinjaEnv::new();
        let out = env.render("Hello {{ name }}!", context! { name => "world" }).unwrap();
        assert_eq!(out, "Hello world!");
    }

    #[test]
    fn pycompat_startswith() {
        let env = JinjaEnv::new();
        let src = "{% if name.startswith('A') %}A!{% else %}other{% endif %}";
        let out = env.render(src, context! { name => "Alice" }).unwrap();
        assert_eq!(out, "A!");
    }

    #[test]
    fn pycompat_raise_exception() {
        let env = JinjaEnv::new();
        let src = "{{ raise_exception('oops') }}";
        let err = env.render(src, context! {}).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("oops"), "expected `oops` in error, got {msg}");
    }

    #[test]
    fn pycompat_python_string_method_args() {
        // Qwen3: `.split('</think>')[-1].lstrip('\n')` на ходах ассистента в
        // истории — раньше strip-семейство не принимало аргумент.
        let env = JinjaEnv::new();
        let src = "{{ c.split('</think>')[-1].lstrip('\\n') }}|{{ c.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') }}|{{ '--x--'.strip('-') }}|{{ '  y '.strip() }}|{{ 'a,b,c'.split(',', 1) | join(';') }}|{{ ' p  q r '.split(none, 1) | join(';') }}|{{ 'aaa'.replace('a', 'b', 2) }}|{{ 'aaa'.replace('a', 'b') }}";
        let out = env.render(src, context! { c => "<think>\nhm\n</think>\n\nhello" }).unwrap();
        assert_eq!(out, "hello|hm|x|y|a;b,c|p;q r |bba|bbb");
    }

    #[test]
    fn pycompat_tojson_indent_kwarg() {
        // Llama 3.x: `{{ t | tojson(indent=4) }}` при переданных tools.
        let env = JinjaEnv::new();
        let src = "{% for t in tools %}{{ t | tojson(indent=4) }}|{{ t | tojson(2) }}|{{ t | tojson }}{% endfor %}";
        let out = env.render(src, context! { tools => vec![context! { name => "f" }] }).unwrap();
        assert_eq!(out, "{\n    \"name\": \"f\"\n}|{\n  \"name\": \"f\"\n}|{\"name\": \"f\"}");
    }

    #[test]
    fn trailing_newline_of_template_is_dropped() {
        let env = JinjaEnv::new();
        let src = "{%- if add_generation_prompt %}{{ 'assistant\n\n' }}{%- endif %}\n";
        assert_eq!(env.render(src, context! { add_generation_prompt => true }).unwrap(), "assistant\n\n");
    }
}
