//! Local JSON harness state watcher (zero network calls) mapping agent/CI status to pet moods
//! and short speech bubbles.

use std::path::{Path, PathBuf};
use std::time::SystemTime;

use serde::Deserialize;

#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", default)]
pub struct HarnessState {
    pub ci_red: bool,
    pub merged: bool,
    pub workers: u32,
    pub waiting_operator: bool,
    pub idle_minutes: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HarnessMood {
    Worried,
    Celebrate,
    Working,
    Wave,
    Sleep,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MoodPresentation {
    pub mood: HarnessMood,
    pub pose: &'static str,
    pub expression: &'static str,
    pub eyes_closed: bool,
    pub bubble_text: String,
    pub accent_rgb: [u8; 3],
}

pub fn mood_preset(mood: HarnessMood, workers: u32) -> MoodPresentation {
    match mood {
        HarnessMood::Worried => MoodPresentation {
            mood,
            pose: "think",
            expression: "sad",
            eyes_closed: false,
            bubble_text: "CI vermelho!".to_string(),
            accent_rgb: [239, 68, 68],
        },
        HarnessMood::Celebrate => MoodPresentation {
            mood,
            pose: "cheer",
            expression: "happy",
            eyes_closed: false,
            bubble_text: "PR mergeado!".to_string(),
            accent_rgb: [16, 185, 129],
        },
        HarnessMood::Wave => MoodPresentation {
            mood,
            pose: "wave",
            expression: "happy",
            eyes_closed: false,
            bubble_text: "Sua vez!".to_string(),
            accent_rgb: [245, 158, 11],
        },
        HarnessMood::Working => {
            let n = workers.max(1);
            let label = if n == 1 {
                "1 worker ativo".to_string()
            } else {
                format!("{n} workers ativos")
            };
            MoodPresentation {
                mood,
                pose: "point",
                expression: "neutral",
                eyes_closed: false,
                bubble_text: label,
                accent_rgb: [59, 130, 246],
            }
        }
        HarnessMood::Sleep => MoodPresentation {
            mood,
            pose: "sit",
            expression: "relaxed",
            eyes_closed: true,
            bubble_text: "Zzz...".to_string(),
            accent_rgb: [139, 92, 246],
        },
    }
}

/// Deterministic priority mapping from local harness state to pet mood & speech bubble.
pub fn evaluate_mood(state: &HarnessState) -> Option<MoodPresentation> {
    if state.ci_red {
        Some(mood_preset(HarnessMood::Worried, state.workers))
    } else if state.merged {
        Some(mood_preset(HarnessMood::Celebrate, state.workers))
    } else if state.waiting_operator {
        Some(mood_preset(HarnessMood::Wave, state.workers))
    } else if state.workers > 0 {
        Some(mood_preset(HarnessMood::Working, state.workers))
    } else if state.idle_minutes >= 15.0 {
        Some(mood_preset(HarnessMood::Sleep, 0))
    } else {
        None
    }
}

pub struct HarnessWatcher {
    path: Option<PathBuf>,
    last_sig: Option<(u64, Option<SystemTime>)>,
    current: Option<MoodPresentation>,
    reads: u64,
}

impl HarnessWatcher {
    pub fn new(path: Option<PathBuf>) -> Self {
        Self {
            path,
            last_sig: None,
            current: None,
            reads: 0,
        }
    }

    /// Checks file metadata (`len` + `modified`) first and skips disk read & JSON parse on unchanged ticks.
    pub fn poll(&mut self) -> Option<&MoodPresentation> {
        let path = self.path.as_ref()?;
        let Ok(meta) = std::fs::metadata(path) else {
            self.last_sig = None;
            self.current = None;
            return None;
        };
        if !meta.is_file() || meta.len() > 256 * 1024 {
            return self.current.as_ref();
        }
        let sig = (meta.len(), meta.modified().ok());
        if self.last_sig.as_ref() == Some(&sig) {
            return self.current.as_ref();
        }
        self.last_sig = Some(sig);
        self.reads += 1;
        if let Ok(raw) = std::fs::read_to_string(path)
            && let Ok(parsed) = serde_json::from_str::<HarnessState>(&raw)
        {
            self.current = evaluate_mood(&parsed);
        }
        self.current.as_ref()
    }

    pub fn current(&self) -> Option<&MoodPresentation> {
        self.current.as_ref()
    }

    #[cfg(test)]
    pub fn read_count(&self) -> u64 {
        self.reads
    }

    #[allow(dead_code)]
    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_all_five_moods_with_clear_priority_and_bubbles() {
        let worried = evaluate_mood(&HarnessState {
            ci_red: true,
            merged: true,
            workers: 3,
            waiting_operator: true,
            idle_minutes: 30.0,
        })
        .expect("ciRed -> worried");
        assert_eq!(worried.mood, HarnessMood::Worried);
        assert!(!worried.bubble_text.is_empty());

        let celebrate = evaluate_mood(&HarnessState {
            merged: true,
            workers: 2,
            ..Default::default()
        })
        .expect("merged -> celebrate");
        assert_eq!(celebrate.mood, HarnessMood::Celebrate);
        assert_eq!(celebrate.pose, "cheer");
        assert_eq!(celebrate.expression, "happy");

        let wave = evaluate_mood(&HarnessState {
            waiting_operator: true,
            workers: 1,
            ..Default::default()
        })
        .expect("waitingOperator -> wave");
        assert_eq!(wave.mood, HarnessMood::Wave);
        assert_eq!(wave.pose, "wave");

        let working = evaluate_mood(&HarnessState {
            workers: 2,
            ..Default::default()
        })
        .expect("workers -> working");
        assert_eq!(working.mood, HarnessMood::Working);
        assert!(working.bubble_text.contains('2'));

        let sleep = evaluate_mood(&HarnessState {
            idle_minutes: 20.0,
            ..Default::default()
        })
        .expect("idleMinutes -> sleep");
        assert_eq!(sleep.mood, HarnessMood::Sleep);
        assert!(sleep.eyes_closed);

        assert!(evaluate_mood(&HarnessState::default()).is_none());
    }

    #[test]
    fn watcher_skips_reread_when_file_unchanged_and_reloads_on_change() {
        let tmp = std::env::temp_dir().join(format!(
            "rune-pet-harness-test-{}.json",
            std::process::id()
        ));
        std::fs::write(&tmp, r#"{"workers": 3}"#).unwrap();

        let mut watcher = HarnessWatcher::new(Some(tmp.clone()));
        let m1 = watcher.poll().expect("initial mood").clone();
        assert_eq!(m1.mood, HarnessMood::Working);
        assert_eq!(watcher.read_count(), 1);

        let _ = watcher.poll();
        assert_eq!(watcher.read_count(), 1, "unchanged tick should skip file read");

        std::fs::write(&tmp, r#"{"ciRed": true, "workers": 0}"#).unwrap();
        let m2 = watcher.poll().expect("updated mood").clone();
        assert_eq!(m2.mood, HarnessMood::Worried);
        assert_eq!(watcher.read_count(), 2);

        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn missing_or_malformed_json_stays_safe() {
        let missing = std::env::temp_dir().join("rune-pet-nonexistent-state.json");
        let _ = std::fs::remove_file(&missing);
        let mut w = HarnessWatcher::new(Some(missing));
        assert!(w.poll().is_none());

        let bad = std::env::temp_dir().join(format!("rune-pet-bad-{}.json", std::process::id()));
        std::fs::write(&bad, "{not valid json").unwrap();
        let mut w2 = HarnessWatcher::new(Some(bad.clone()));
        assert!(w2.poll().is_none());
        let _ = std::fs::remove_file(&bad);
    }
}
