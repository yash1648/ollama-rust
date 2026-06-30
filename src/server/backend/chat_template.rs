//! Chat template formatting for multi-turn conversations.
//!
//! Different model architectures use different prompt formats
//! for chat. This module detects the correct template from
//! the model architecture and formats a sequence of messages
//! into a single prompt string.
//!
//! ## Supported Templates
//!
//! | Template | Models | Format |
//! |----------|--------|--------|
//! | [`ChatML`](ChatTemplate::ChatML) | Qwen2, llama2, Yi, DeepSeek, CodeLlama | `<\|im_start\|>{role}\n{content}<\|im_end\|>\n` |
//! | [`Llama3`](ChatTemplate::Llama3) | Llama 3, 3.1, 3.2 | `<\|start_header_id\|>{role}<\|end_header_id\|>\n\n{content}<\|eot_id\|>` |
//! | [`Mistral`](ChatTemplate::Mistral) | Mistral, Mixtral | `<s>[INST] {user} [/INST] {assistant}</s>` |
//! | [`Phi3`](ChatTemplate::Phi3) | Phi-3, Phi-3-mini | `<\|user\|>\n{content}<\|end\|>\n<\|assistant\|>\n` |
//! | [`Gemma`](ChatTemplate::Gemma) | Gemma 2/3 | `<start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n` |

use crate::model::types::Message;

/// Supported chat template formats, mapped from model architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatTemplate {
    /// Llama 3+ format with `<|start_header_id|>` / `<|eot_id|>` tokens.
    Llama3,
    /// Mistral format with `[INST]` / `[/INST]` tokens.
    Mistral,
    /// Phi-3 format with `<|user|>` / `<|end|>` / `<|assistant|>` tokens.
    Phi3,
    /// Gemma format with `<start_of_turn>` / `<end_of_turn>` tokens.
    Gemma,
    /// ChatML format with `<|im_start|>` / `<|im_end|>` tokens.
    ChatML,
}

impl ChatTemplate {
    /// Detect the chat template from a GGUF `general.architecture` string.
    ///
    /// Falls back to [`ChatML`](ChatTemplate::ChatML) (the most common generic format)
    /// for unrecognised architectures.
    pub fn from_architecture(arch: &str) -> Self {
        match arch {
            "llama3" | "llama3.1" | "llama3.2" | "llama-3" | "llama-3.1" | "llama-3.2" => {
                Self::Llama3
            }
            "mistral" | "mixtral" => Self::Mistral,
            "phi3" | "phi-3" | "phi-3-mini" => Self::Phi3,
            "gemma" | "gemma2" | "gemma3" => Self::Gemma,
            // Default: qwen2, llama2, yi, deepseek2, codellama, and unknown
            _ => Self::ChatML,
        }
    }

    /// Format a sequence of chat messages into a single prompt string,
    /// appending the final assistant-turn header for generation.
    ///
    /// ## Example
    ///
    /// ```
    /// use ollama_rs::server::backend::chat_template::ChatTemplate;
    /// use ollama_rs::model::types::Message;
    ///
    /// let msgs = vec![
    ///     Message { role: "user".into(), content: "Hello!".into(), images: None },
    /// ];
    /// let prompt = ChatTemplate::ChatML.format(&msgs);
    /// assert_eq!(prompt, "<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n");
    /// ```
    pub fn format(&self, messages: &[Message]) -> String {
        if messages.is_empty() {
            return String::new();
        }
        match self {
            Self::Llama3 => format_llama3(messages),
            Self::Mistral => format_mistral(messages),
            Self::Phi3 => format_phi3(messages),
            Self::Gemma => format_gemma(messages),
            Self::ChatML => format_chatml(messages),
        }
    }
}

// ── Template implementations ───────────────────────────────────────────────

/// Llama 3+ format:
/// `<|begin_of_text|><|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>`
/// Then `<|start_header_id|>assistant<|end_header_id|>\n\n` for generation.
fn format_llama3(messages: &[Message]) -> String {
    let mut out = String::from("<|begin_of_text|>");
    for msg in messages {
        out.push_str(&format!(
            "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>",
            msg.role, msg.content
        ));
    }
    out.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    out
}

/// Mistral format:
/// `<s>[INST] {user} [/INST] {assistant}</s><s>[INST] {user} [/INST]`
/// With optional leading system message folded into first `[INST]`.
fn format_mistral(messages: &[Message]) -> String {
    let mut out = String::from("<s>");
    let mut i = 0;

    // Optional system message folded into first [INST]
    if let Some(first) = messages.first() {
        if first.role == "system" {
            out.push_str(&format!("[INST] {}\n\n", first.content));
            i = 1;
        } else {
            out.push_str("[INST] ");
        }
    }

    while i < messages.len() {
        match messages[i].role.as_str() {
            "user" => {
                out.push_str(&messages[i].content);
                i += 1;
                // Check if next message is an assistant response
                if i < messages.len() && messages[i].role == "assistant" {
                    out.push_str(&format!(" [/INST] {} </s>", messages[i].content));
                    i += 1;
                    if i < messages.len() {
                        out.push_str("<s>[INST] ");
                    }
                } else {
                    out.push_str(" [/INST]");
                }
            }
            "assistant" => {
                // Orphan assistant message — prefix as response
                out.push_str(&format!(" [/INST] {} </s>", messages[i].content));
                i += 1;
                if i < messages.len() {
                    out.push_str("<s>[INST] ");
                }
            }
            _ => {
                i += 1;
            }
        }
    }

    // Ensure we end with [/INST] for generation
    if !out.ends_with("[/INST] ") && !out.ends_with("[/INST]") {
        out.push_str(" [/INST]");
    }

    out
}

/// Phi-3 format:
/// `<|system|>\n{content}<|end|>\n<|user|>\n{content}<|end|>\n<|assistant|>\n{content}<|end|>\n`
/// Then `<|assistant|>\n` for generation.
fn format_phi3(messages: &[Message]) -> String {
    let mut out = String::new();
    for msg in messages {
        out.push_str(&format!("<|{}|>\n{}<|end|>\n", msg.role, msg.content));
    }
    out.push_str("<|assistant|>\n");
    out
}

/// Gemma format:
/// `<bos><start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n{content}<end_of_turn>\n`
/// Then `<start_of_turn>model\n` for generation.
fn format_gemma(messages: &[Message]) -> String {
    let mut out = String::from("<bos>");
    for msg in messages {
        let role_label = match msg.role.as_str() {
            "assistant" => "model",
            other => other,
        };
        out.push_str(&format!(
            "<start_of_turn>{}\n{}<end_of_turn>\n",
            role_label, msg.content
        ));
    }
    out.push_str("<start_of_turn>model\n");
    out
}

/// ChatML format:
/// `<|im_start|>{role}\n{content}<|im_end|>\n`
/// Then `<|im_start|>assistant\n` for generation.
fn format_chatml(messages: &[Message]) -> String {
    let mut out = String::new();
    for msg in messages {
        out.push_str(&format!(
            "<|im_start|>{}\n{}<|im_end|>\n",
            msg.role, msg.content
        ));
    }
    out.push_str("<|im_start|>assistant\n");
    out
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::types::Message;

    fn msg(role: &str, content: &str) -> Message {
        Message {
            role: role.into(),
            content: content.into(),
            images: None,
        }
    }

    #[test]
    fn test_chatml_single_user() {
        let msgs = vec![msg("user", "Hello")];
        let result = ChatTemplate::ChatML.format(&msgs);
        assert_eq!(result, "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n");
    }

    #[test]
    fn test_chatml_system_user_assistant() {
        let msgs = vec![
            msg("system", "You are helpful."),
            msg("user", "Hi"),
            msg("assistant", "Hello!"),
        ];
        let result = ChatTemplate::ChatML.format(&msgs);
        assert_eq!(
            result,
            "<|im_start|>system\nYou are helpful.<|im_end|>\n\
             <|im_start|>user\nHi<|im_end|>\n\
             <|im_start|>assistant\nHello!<|im_end|>\n\
             <|im_start|>assistant\n"
        );
    }

    #[test]
    fn test_llama3_single_user() {
        let msgs = vec![msg("user", "Hello")];
        let result = ChatTemplate::Llama3.format(&msgs);
        assert_eq!(
            result,
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|>\
             <|start_header_id|>assistant<|end_header_id|>\n\n"
        );
    }

    #[test]
    fn test_llama3_system_user() {
        let msgs = vec![msg("system", "Be nice."), msg("user", "Hi")];
        let result = ChatTemplate::Llama3.format(&msgs);
        assert_eq!(
            result,
            "<|begin_of_text|>\
             <|start_header_id|>system<|end_header_id|>\n\nBe nice.<|eot_id|>\
             <|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>\
             <|start_header_id|>assistant<|end_header_id|>\n\n"
        );
    }

    #[test]
    fn test_mistral_single_user() {
        let msgs = vec![msg("user", "Hello")];
        let result = ChatTemplate::Mistral.format(&msgs);
        assert_eq!(result, "<s>[INST] Hello [/INST]");
    }

    #[test]
    fn test_mistral_system_user() {
        let msgs = vec![msg("system", "You are a bot."), msg("user", "Hi")];
        let result = ChatTemplate::Mistral.format(&msgs);
        assert_eq!(result, "<s>[INST] You are a bot.\n\nHi [/INST]");
    }

    #[test]
    fn test_mistral_user_assistant_roundtrip() {
        let msgs = vec![
            msg("user", "Hello"),
            msg("assistant", "Hi there!"),
            msg("user", "How are you?"),
        ];
        let result = ChatTemplate::Mistral.format(&msgs);
        assert_eq!(
            result,
            "<s>[INST] Hello [/INST] Hi there! </s><s>[INST] How are you? [/INST]"
        );
    }

    #[test]
    fn test_mistral_system_user_assistant() {
        let msgs = vec![
            msg("system", "Be concise."),
            msg("user", "What's Rust?"),
            msg("assistant", "A systems language."),
            msg("user", "Thanks"),
        ];
        let result = ChatTemplate::Mistral.format(&msgs);
        assert_eq!(
            result,
            "<s>[INST] Be concise.\n\nWhat's Rust? [/INST] A systems language. </s><s>[INST] Thanks [/INST]"
        );
    }

    #[test]
    fn test_phi3_single_user() {
        let msgs = vec![msg("user", "Hello")];
        let result = ChatTemplate::Phi3.format(&msgs);
        assert_eq!(result, "<|user|>\nHello<|end|>\n<|assistant|>\n");
    }

    #[test]
    fn test_phi3_system_user() {
        let msgs = vec![msg("system", "You are an AI."), msg("user", "Hi")];
        let result = ChatTemplate::Phi3.format(&msgs);
        assert_eq!(
            result,
            "<|system|>\nYou are an AI.<|end|>\n<|user|>\nHi<|end|>\n<|assistant|>\n"
        );
    }

    #[test]
    fn test_gemma_single_user() {
        let msgs = vec![msg("user", "Hello")];
        let result = ChatTemplate::Gemma.format(&msgs);
        assert_eq!(
            result,
            "<bos><start_of_turn>user\nHello<end_of_turn>\n<start_of_turn>model\n"
        );
    }

    #[test]
    fn test_gemma_user_assistant_roundtrip() {
        let msgs = vec![
            msg("user", "Hi"),
            msg("assistant", "Hello!"),
            msg("user", "How are you?"),
        ];
        let result = ChatTemplate::Gemma.format(&msgs);
        assert_eq!(
            result,
            "<bos>\
             <start_of_turn>user\nHi<end_of_turn>\n\
             <start_of_turn>model\nHello!<end_of_turn>\n\
             <start_of_turn>user\nHow are you?<end_of_turn>\n\
             <start_of_turn>model\n"
        );
    }

    #[test]
    fn test_from_architecture() {
        assert_eq!(ChatTemplate::from_architecture("llama3"), ChatTemplate::Llama3);
        assert_eq!(ChatTemplate::from_architecture("llama3.1"), ChatTemplate::Llama3);
        assert_eq!(ChatTemplate::from_architecture("llama3.2"), ChatTemplate::Llama3);
        assert_eq!(ChatTemplate::from_architecture("mistral"), ChatTemplate::Mistral);
        assert_eq!(ChatTemplate::from_architecture("mixtral"), ChatTemplate::Mistral);
        assert_eq!(ChatTemplate::from_architecture("phi3"), ChatTemplate::Phi3);
        assert_eq!(ChatTemplate::from_architecture("phi-3-mini"), ChatTemplate::Phi3);
        assert_eq!(ChatTemplate::from_architecture("gemma3"), ChatTemplate::Gemma);
        assert_eq!(ChatTemplate::from_architecture("gemma2"), ChatTemplate::Gemma);
        assert_eq!(ChatTemplate::from_architecture("gemma"), ChatTemplate::Gemma);
        assert_eq!(ChatTemplate::from_architecture("qwen2"), ChatTemplate::ChatML);
        assert_eq!(ChatTemplate::from_architecture("llama2"), ChatTemplate::ChatML);
        assert_eq!(ChatTemplate::from_architecture("unknown"), ChatTemplate::ChatML);
    }

    #[test]
    fn test_empty_messages() {
        let result = ChatTemplate::ChatML.format(&[]);
        assert_eq!(result, "");
    }
}
