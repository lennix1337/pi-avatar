//! Companion protocol (`companion/PROTOCOL.md`): one JSON object per line.
//!
//! Parsing is lenient on purpose: unknown fields are ignored and an invalid line yields `None`,
//! so one bad message never drops the connection or the frames around it.

use serde_json::Value;

/// Longest line the pet will look at; anything larger is dropped unparsed.
pub const MAX_LINE_BYTES: usize = 64 * 1024;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Visemes {
    pub aa: f32,
    pub ih: f32,
    pub ou: f32,
    pub ee: f32,
    pub oh: f32,
}

impl Visemes {
    pub fn to_array(self) -> [f32; 5] {
        [self.aa, self.ih, self.ou, self.ee, self.oh]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmotionKind {
    Neutral,
    Happy,
    Angry,
    Sad,
    Relaxed,
    Surprised,
    Smug,
    Teasing,
    Embarrassed,
    Bored,
    Excited,
    Confused,
}

impl EmotionKind {
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name.trim().to_ascii_lowercase().as_str() {
            "neutral" => Self::Neutral,
            "happy" => Self::Happy,
            "angry" => Self::Angry,
            "sad" => Self::Sad,
            "relaxed" => Self::Relaxed,
            "surprised" => Self::Surprised,
            "smug" => Self::Smug,
            "teasing" => Self::Teasing,
            "embarrassed" => Self::Embarrassed,
            "bored" => Self::Bored,
            "excited" => Self::Excited,
            "confused" => Self::Confused,
            _ => return None,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GestureKind {
    Wave,
    Cheer,
    Nod,
    Shake,
    Tilt,
    Laugh,
    Shrug,
    LeanIn,
    LookAway,
}

impl GestureKind {
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name.trim().to_ascii_lowercase().as_str() {
            "wave" => Self::Wave,
            "cheer" => Self::Cheer,
            "nod" => Self::Nod,
            "shake" => Self::Shake,
            "tilt" => Self::Tilt,
            "laugh" => Self::Laugh,
            "shrug" => Self::Shrug,
            "lean_in" => Self::LeanIn,
            "look_away" => Self::LookAway,
            _ => return None,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Incoming {
    State { listening: bool, thinking: bool, speaking: bool },
    Say { id: String, text: String, emotion: Option<EmotionKind> },
    Mouth(Visemes),
    Emotion { kind: EmotionKind, intensity: f32 },
    Gesture(GestureKind),
    End { id: String },
}

fn flag(v: &Value, key: &str) -> Option<bool> {
    match v.get(key) {
        None | Some(Value::Null) => Some(false),
        Some(Value::Bool(b)) => Some(*b),
        _ => None,
    }
}

fn unit(v: &Value, key: &str, default: f32) -> Option<f32> {
    match v.get(key) {
        None | Some(Value::Null) => Some(default),
        Some(n) => n.as_f64().map(|x| (x as f32).clamp(0.0, 1.0)),
    }
}

fn string<'a>(v: &'a Value, key: &str) -> Option<&'a str> {
    v.get(key).and_then(Value::as_str)
}

/// Parses one protocol line. `None` means "drop it": malformed JSON, an unknown `t`, or a
/// field of the wrong type.
pub fn parse_line(line: &str) -> Option<Incoming> {
    if line.len() > MAX_LINE_BYTES {
        return None;
    }
    let v: Value = serde_json::from_str(line.trim()).ok()?;
    match string(&v, "t")? {
        "state" => Some(Incoming::State {
            listening: flag(&v, "listening")?,
            thinking: flag(&v, "thinking")?,
            speaking: flag(&v, "speaking")?,
        }),
        "say" => Some(Incoming::Say {
            id: string(&v, "id").unwrap_or_default().to_string(),
            text: string(&v, "text")?.to_string(),
            emotion: string(&v, "emotion").and_then(EmotionKind::from_name),
        }),
        "mouth" => Some(Incoming::Mouth(Visemes {
            aa: unit(&v, "aa", 0.0)?,
            ih: unit(&v, "ih", 0.0)?,
            ou: unit(&v, "ou", 0.0)?,
            ee: unit(&v, "ee", 0.0)?,
            oh: unit(&v, "oh", 0.0)?,
        })),
        "emotion" => Some(Incoming::Emotion {
            kind: EmotionKind::from_name(string(&v, "name")?)?,
            intensity: unit(&v, "intensity", 1.0)?,
        }),
        "gesture" => Some(Incoming::Gesture(GestureKind::from_name(string(&v, "name")?)?)),
        "end" => Some(Incoming::End { id: string(&v, "id").unwrap_or_default().to_string() }),
        _ => None,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PokeKind {
    Pat,
    Double,
    Drag,
    Drop,
}

impl PokeKind {
    fn name(self) -> &'static str {
        match self {
            Self::Pat => "pat",
            Self::Double => "double",
            Self::Drag => "drag",
            Self::Drop => "drop",
        }
    }
}

pub fn hello_line() -> String {
    format!("{}\n", serde_json::json!({ "t": "hello", "version": env!("CARGO_PKG_VERSION") }))
}

pub fn poke_line(kind: PokeKind) -> String {
    format!("{}\n", serde_json::json!({ "t": "poke", "kind": kind.name() }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_every_incoming_message_of_the_contract() {
        assert_eq!(
            parse_line(r#"{"t":"state","listening":false,"thinking":true,"speaking":false}"#),
            Some(Incoming::State { listening: false, thinking: true, speaking: false })
        );
        assert_eq!(
            parse_line(r#"{"t":"say","id":"u7","text":"Você de novo?","emotion":"smug"}"#),
            Some(Incoming::Say {
                id: "u7".into(),
                text: "Você de novo?".into(),
                emotion: Some(EmotionKind::Smug)
            })
        );
        assert_eq!(
            parse_line(r#"{"t":"mouth","aa":0.62,"ih":0.0,"ou":0.08,"ee":0.0,"oh":0.12}"#),
            Some(Incoming::Mouth(Visemes { aa: 0.62, ih: 0.0, ou: 0.08, ee: 0.0, oh: 0.12 }))
        );
        assert_eq!(
            parse_line(r#"{"t":"emotion","name":"embarrassed","intensity":0.4}"#),
            Some(Incoming::Emotion { kind: EmotionKind::Embarrassed, intensity: 0.4 })
        );
        assert_eq!(
            parse_line(r#"{"t":"gesture","name":"lean_in"}"#),
            Some(Incoming::Gesture(GestureKind::LeanIn))
        );
        assert_eq!(parse_line(r#"{"t":"end","id":"u7"}"#), Some(Incoming::End { id: "u7".into() }));
    }

    #[test]
    fn unknown_fields_are_ignored() {
        let m = parse_line(r#"{"t":"mouth","aa":0.5,"future":[1,2],"at":3.2}"#);
        assert_eq!(m, Some(Incoming::Mouth(Visemes { aa: 0.5, ..Visemes::default() })));
        let e = parse_line(r#"{"t":"emotion","name":"happy","extra":{"x":1}}"#);
        assert_eq!(e, Some(Incoming::Emotion { kind: EmotionKind::Happy, intensity: 1.0 }));
    }

    #[test]
    fn malformed_or_invalid_lines_are_dropped() {
        for bad in [
            "",
            "not json",
            "{",
            "[1,2]",
            r#"{"t":42}"#,
            r#"{"x":1}"#,
            r#"{"t":"teleport"}"#,
            r#"{"t":"say","id":"1"}"#,
            r#"{"t":"say","id":"1","text":7}"#,
            r#"{"t":"mouth","aa":"wide"}"#,
            r#"{"t":"emotion","name":"grumpy"}"#,
            r#"{"t":"gesture","name":"moonwalk"}"#,
            r#"{"t":"state","speaking":"yes"}"#,
        ] {
            assert_eq!(parse_line(bad), None, "should drop {bad:?}");
        }
    }

    #[test]
    fn oversized_lines_are_dropped() {
        let big = format!(r#"{{"t":"say","id":"1","text":"{}"}}"#, "a".repeat(MAX_LINE_BYTES));
        assert_eq!(parse_line(&big), None);
    }

    #[test]
    fn viseme_and_intensity_values_are_clamped() {
        assert_eq!(
            parse_line(r#"{"t":"mouth","aa":3.0,"oh":-1.0}"#),
            Some(Incoming::Mouth(Visemes { aa: 1.0, oh: 0.0, ..Visemes::default() }))
        );
        assert_eq!(
            parse_line(r#"{"t":"emotion","name":"sad","intensity":9}"#),
            Some(Incoming::Emotion { kind: EmotionKind::Sad, intensity: 1.0 })
        );
    }

    #[test]
    fn missing_state_flags_default_to_false_and_say_tolerates_unknown_emotion() {
        assert_eq!(
            parse_line(r#"{"t":"state","speaking":true}"#),
            Some(Incoming::State { listening: false, thinking: false, speaking: true })
        );
        assert_eq!(
            parse_line(r#"{"t":"say","id":"a","text":"oi","emotion":"gleeful"}"#),
            Some(Incoming::Say { id: "a".into(), text: "oi".into(), emotion: None })
        );
    }

    #[test]
    fn outgoing_lines_are_single_json_objects() {
        let hello: Value = serde_json::from_str(hello_line().trim_end()).unwrap();
        assert_eq!(hello["t"], "hello");
        assert!(hello["version"].is_string());
        assert!(hello_line().ends_with('\n') && !hello_line().trim_end().contains('\n'));
        for (kind, name) in [
            (PokeKind::Pat, "pat"),
            (PokeKind::Double, "double"),
            (PokeKind::Drag, "drag"),
            (PokeKind::Drop, "drop"),
        ] {
            let v: Value = serde_json::from_str(poke_line(kind).trim_end()).unwrap();
            assert_eq!((v["t"].as_str(), v["kind"].as_str()), (Some("poke"), Some(name)));
        }
    }
}
