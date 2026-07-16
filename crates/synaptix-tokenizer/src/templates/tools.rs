use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ToolDef {
    #[serde(rename = "function")]
    Function {
        function: ToolFunction,
    },
}

impl ToolDef {
    pub fn function(name: impl Into<String>, description: impl Into<String>, parameters: ToolParameterSchema) -> Self {
        Self::Function {
            function: ToolFunction {
                name: name.into(),
                description: Some(description.into()),
                parameters,
            },
        }
    }

    pub fn name(&self) -> &str {
        match self {
            Self::Function { function } => &function.name,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolFunction {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: ToolParameterSchema,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolParameterSchema {
    #[serde(rename = "type")]
    pub schema_type: String,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub properties: BTreeMap<String, ToolParamProperty>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub required: Vec<String>,
}

impl ToolParameterSchema {
    pub fn object() -> Self {
        Self { schema_type: "object".into(), properties: BTreeMap::new(), required: Vec::new() }
    }

    pub fn property(mut self, name: impl Into<String>, prop: ToolParamProperty) -> Self {
        self.properties.insert(name.into(), prop);
        self
    }

    pub fn require(mut self, name: impl Into<String>) -> Self {
        self.required.push(name.into());
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolParamProperty {
    #[serde(rename = "type")]
    pub schema_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, rename = "enum", skip_serializing_if = "Vec::is_empty")]
    pub enum_values: Vec<String>,
}

impl ToolParamProperty {
    pub fn new(schema_type: impl Into<String>) -> Self {
        Self { schema_type: schema_type.into(), description: None, enum_values: Vec::new() }
    }

    pub fn description(mut self, d: impl Into<String>) -> Self {
        self.description = Some(d.into());
        self
    }

    pub fn enumerated(mut self, vals: impl IntoIterator<Item = String>) -> Self {
        self.enum_values = vals.into_iter().collect();
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolCall {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type", default = "default_type")]
    pub call_type: String,
    pub function: ToolCallFunction,
}

fn default_type() -> String {
    "function".into()
}

impl ToolCall {
    pub fn function(name: impl Into<String>, arguments: impl Into<String>) -> Self {
        Self {
            id: None,
            call_type: "function".into(),
            function: ToolCallFunction { name: name.into(), arguments: arguments.into() },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolCallFunction {
    pub name: String,
    pub arguments: String,
}
