//! Генерация одним вызовом поверх крейтов моделей synaptix — для службы
//! `synaptixd` и приложений: выбор пайплайна по модели, умолчания как у
//! `synaptix-cli`, прогресс и отмена.

pub mod color_match;
pub mod h3_stages;
pub mod image;
pub mod image_stages;
pub mod ltx_stages;
pub mod ltx_text;
pub mod music;
pub mod music_stages;
pub mod tail;
pub mod video;
