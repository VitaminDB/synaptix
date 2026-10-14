//! Генерация одним вызовом поверх крейтов моделей synaptix — для службы
//! `synaptixd` и приложений: выбор пайплайна по модели, умолчания как у
//! `synaptix-cli`, прогресс и отмена.

pub mod h3_stages;
pub mod image;
pub mod image_stages;
pub mod ltx_stages;
pub mod music;
pub mod video;
