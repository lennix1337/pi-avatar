//! Non-blocking companion client: a background thread owns the TCP connection (reconnecting every
//! 2 s), parses lines into an inbox the UI thread drains, and sends `hello`/`poke`. The UI thread
//! only ever takes a short mutex or pushes to a channel, so a dead or slow companion never costs
//! a frame.

use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::path::Path;
use std::sync::mpsc::{Receiver, Sender, TryRecvError, channel};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use crate::protocol::{self, Incoming, MAX_LINE_BYTES, PokeKind};

pub const DEFAULT_ADDR: &str = "127.0.0.1:47800";
pub const RECONNECT_DELAY: Duration = Duration::from_secs(2);
const CONNECT_TIMEOUT: Duration = Duration::from_millis(800);
const READ_SLICE: Duration = Duration::from_millis(40);
/// A stalled UI thread must not let a flood of `mouth` frames grow memory without bound.
const INBOX_CAP: usize = 2048;

pub type Waker = Arc<dyn Fn() + Send + Sync>;

/// Decides how long to wait before the next connection attempt: attempts are at least
/// [`RECONNECT_DELAY`] apart, measured from the previous attempt, never from the disconnect.
#[derive(Debug, Default)]
pub struct ReconnectTimer {
    last_attempt: Option<Instant>,
}

impl ReconnectTimer {
    pub fn wait(&self, now: Instant) -> Duration {
        match self.last_attempt {
            None => Duration::ZERO,
            Some(t) => RECONNECT_DELAY.saturating_sub(now.saturating_duration_since(t)),
        }
    }

    pub fn mark_attempt(&mut self, now: Instant) {
        self.last_attempt = Some(now);
    }
}

/// Splits a byte stream into `\n` terminated lines. A line longer than [`MAX_LINE_BYTES`] is
/// discarded as a whole (its tail included) so a hostile or broken peer cannot grow memory.
#[derive(Debug, Default)]
pub struct LineBuffer {
    buf: Vec<u8>,
    discarding: bool,
}

impl LineBuffer {
    pub fn push(&mut self, bytes: &[u8]) -> Vec<String> {
        let mut lines = Vec::new();
        for &b in bytes {
            if b == b'\n' {
                if !self.discarding && !self.buf.is_empty() {
                    lines.push(String::from_utf8_lossy(&self.buf).into_owned());
                }
                self.buf.clear();
                self.discarding = false;
            } else if !self.discarding {
                if self.buf.len() >= MAX_LINE_BYTES {
                    self.buf.clear();
                    self.discarding = true;
                } else {
                    self.buf.push(b);
                }
            }
        }
        lines
    }
}

#[derive(Default)]
pub struct Inbox {
    queue: Mutex<Vec<Incoming>>,
}

impl Inbox {
    pub fn push(&self, msg: Incoming) {
        if let Ok(mut q) = self.queue.lock() {
            if q.len() >= INBOX_CAP {
                q.remove(0);
            }
            q.push(msg);
        }
    }

    pub fn drain(&self) -> Vec<Incoming> {
        self.queue.lock().map(|mut q| std::mem::take(&mut *q)).unwrap_or_default()
    }
}

pub struct Link {
    pub inbox: Inbox,
    outbox: Sender<PokeKind>,
}

static LINK: OnceLock<Arc<Link>> = OnceLock::new();

/// Messages that arrived since the last call (empty without a companion).
pub fn drain() -> Vec<Incoming> {
    LINK.get().map(|l| l.inbox.drain()).unwrap_or_default()
}

/// Tells the companion the user poked the avatar. A no-op without a connection.
pub fn send_poke(kind: PokeKind) {
    if let Some(link) = LINK.get() {
        let _ = link.outbox.send(kind);
    }
}

fn new_link() -> (Arc<Link>, Receiver<PokeKind>) {
    let (tx, rx) = channel();
    (Arc::new(Link { inbox: Inbox::default(), outbox: tx }), rx)
}

/// Starts the client thread for `addr`. Calling it twice keeps the first link.
pub fn start(addr: String, waker: Waker) {
    let (link, outbox_rx) = new_link();
    if LINK.set(link.clone()).is_err() {
        return;
    }
    let _ = std::thread::Builder::new()
        .name("companion-client".into())
        .spawn(move || client_loop(&addr, &link, &outbox_rx, &waker));
}

fn client_loop(addr: &str, link: &Link, outbox: &Receiver<PokeKind>, waker: &Waker) {
    let mut timer = ReconnectTimer::default();
    loop {
        // Pokes sent while nobody listens are dropped, not replayed on the next connection.
        let wait = timer.wait(Instant::now());
        let deadline = Instant::now() + wait;
        while Instant::now() < deadline {
            while outbox.try_recv().is_ok() {}
            std::thread::sleep(Duration::from_millis(50).min(deadline - Instant::now()));
        }
        timer.mark_attempt(Instant::now());
        let Some(stream) = addr
            .to_socket_addrs()
            .ok()
            .and_then(|mut a| a.find_map(|sa| TcpStream::connect_timeout(&sa, CONNECT_TIMEOUT).ok()))
        else {
            continue;
        };
        run_session(stream, link, outbox, waker);
        // Whatever the companion was doing stops when it goes away.
        link.inbox.push(Incoming::State { listening: false, thinking: false, speaking: false });
        link.inbox.push(Incoming::End { id: String::new() });
        waker();
    }
}

/// Serves one connection until it drops: `hello`, then read lines into the inbox and write the
/// pokes queued by the UI.
pub fn run_session(mut stream: TcpStream, link: &Link, outbox: &Receiver<PokeKind>, waker: &Waker) {
    let _ = stream.set_nodelay(true);
    if stream.set_read_timeout(Some(READ_SLICE)).is_err() {
        return;
    }
    if stream.write_all(protocol::hello_line().as_bytes()).is_err() {
        return;
    }
    let mut lines = LineBuffer::default();
    let mut chunk = [0u8; 4096];
    loop {
        match stream.read(&mut chunk) {
            Ok(0) => return,
            Ok(n) => {
                let mut any = false;
                for line in lines.push(&chunk[..n]) {
                    if let Some(msg) = protocol::parse_line(&line) {
                        link.inbox.push(msg);
                        any = true;
                    }
                }
                if any {
                    waker();
                }
            }
            Err(e) if matches!(e.kind(), std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut) => {}
            Err(_) => return,
        }
        loop {
            match outbox.try_recv() {
                Ok(kind) => {
                    if stream.write_all(protocol::poke_line(kind).as_bytes()).is_err() {
                        return;
                    }
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => return,
            }
        }
    }
}

/// Parses a replay file: protocol lines with an extra `"at"` (seconds since start). Lines with
/// no `at` follow the previous one. Invalid lines are skipped.
pub fn parse_replay(text: &str) -> Vec<(f64, Incoming)> {
    let mut out: Vec<(f64, Incoming)> = Vec::new();
    let mut last = 0.0;
    for line in text.lines() {
        let Some(msg) = protocol::parse_line(line) else { continue };
        let at = serde_json::from_str::<serde_json::Value>(line)
            .ok()
            .and_then(|v| v.get("at").and_then(serde_json::Value::as_f64))
            .filter(|t| t.is_finite() && *t >= 0.0)
            .unwrap_or(last);
        last = at;
        out.push((at, msg));
    }
    out.sort_by(|a, b| a.0.total_cmp(&b.0));
    out
}

/// Plays a recorded stream into the inbox in real time, as if a companion had sent it.
pub fn start_replay(path: &Path, waker: Waker) -> std::io::Result<()> {
    let events = parse_replay(&std::fs::read_to_string(path)?);
    let (link, _outbox) = new_link();
    if LINK.set(link.clone()).is_err() {
        return Ok(());
    }
    let _ = std::thread::Builder::new().name("companion-replay".into()).spawn(move || {
        let start = Instant::now();
        for (at, msg) in events {
            let due = start + Duration::from_secs_f64(at);
            if let Some(wait) = due.checked_duration_since(Instant::now()) {
                std::thread::sleep(wait);
            }
            link.inbox.push(msg);
            waker();
        }
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::Visemes;
    use std::io::{BufRead, BufReader};
    use std::net::TcpListener;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn reconnect_waits_two_seconds_from_the_last_attempt() {
        let mut t = ReconnectTimer::default();
        let t0 = Instant::now();
        assert_eq!(t.wait(t0), Duration::ZERO, "the first attempt is immediate");
        t.mark_attempt(t0);
        assert_eq!(t.wait(t0), RECONNECT_DELAY);
        assert_eq!(t.wait(t0 + Duration::from_millis(500)), Duration::from_millis(1500));
        assert_eq!(t.wait(t0 + Duration::from_secs(2)), Duration::ZERO);
        assert_eq!(t.wait(t0 + Duration::from_secs(60)), Duration::ZERO, "never a negative wait");
        assert_eq!(RECONNECT_DELAY, Duration::from_secs(2));
    }

    #[test]
    fn line_buffer_joins_split_reads_and_skips_blank_lines() {
        let mut b = LineBuffer::default();
        assert!(b.push(b"{\"t\":\"en").is_empty());
        assert_eq!(b.push(b"d\"}\n\n{\"t\":\"x\"}\n{\"t"), vec!["{\"t\":\"end\"}", "{\"t\":\"x\"}"]);
        assert_eq!(b.push(b"\":\"y\"}\n"), vec!["{\"t\":\"y\"}"]);
    }

    #[test]
    fn line_buffer_discards_an_oversized_line_and_recovers() {
        let mut b = LineBuffer::default();
        let mut junk = vec![b'x'; MAX_LINE_BYTES * 2];
        junk.push(b'\n');
        junk.extend_from_slice(b"ok\n");
        assert_eq!(b.push(&junk), vec!["ok"]);
    }

    #[test]
    fn inbox_is_bounded_and_keeps_the_newest() {
        let inbox = Inbox::default();
        for i in 0..(INBOX_CAP + 10) {
            inbox.push(Incoming::End { id: i.to_string() });
        }
        let all = inbox.drain();
        assert_eq!(all.len(), INBOX_CAP);
        assert_eq!(all.last(), Some(&Incoming::End { id: (INBOX_CAP + 9).to_string() }));
        assert!(inbox.drain().is_empty());
    }

    #[test]
    fn replay_orders_by_time_and_skips_garbage() {
        let text = concat!(
            "{\"t\":\"end\",\"id\":\"b\",\"at\":2.5}\n",
            "garbage\n",
            "{\"t\":\"state\",\"speaking\":true,\"at\":0.5}\n",
            "{\"t\":\"gesture\",\"name\":\"nod\"}\n",
        );
        let ev = parse_replay(text);
        assert_eq!(ev.len(), 3);
        assert_eq!(ev[0].0, 0.5);
        assert!(matches!(ev[0].1, Incoming::State { speaking: true, .. }));
        assert_eq!((ev[1].0, ev[2].0), (0.5, 2.5), "an `at`-less line follows the previous one");
        assert!(matches!(ev[1].1, Incoming::Gesture(_)));
    }

    #[test]
    fn session_sends_hello_receives_valid_lines_and_forwards_pokes() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (mut s, _) = listener.accept().unwrap();
            let mut r = BufReader::new(s.try_clone().unwrap());
            let mut hello = String::new();
            r.read_line(&mut hello).unwrap();
            s.write_all(b"{\"t\":\"mouth\",\"aa\":0.5}\nnot json\n{\"t\":\"teleport\"}\n").unwrap();
            let mut poke = String::new();
            r.read_line(&mut poke).unwrap();
            (hello, poke)
        });

        let (link, rx) = new_link();
        let wakes = Arc::new(AtomicUsize::new(0));
        let w = wakes.clone();
        let waker: Waker = Arc::new(move || {
            w.fetch_add(1, Ordering::SeqCst);
        });
        link.outbox.send(PokeKind::Pat).unwrap();
        let stream = TcpStream::connect(addr).unwrap();
        let session_link = link.clone();
        let session = std::thread::spawn(move || run_session(stream, &session_link, &rx, &waker));

        let (hello, poke) = server.join().unwrap();
        session.join().unwrap();
        assert!(hello.contains("\"hello\""), "{hello}");
        assert!(poke.contains("\"poke\"") && poke.contains("\"pat\""), "{poke}");
        assert_eq!(
            link.inbox.drain(),
            vec![Incoming::Mouth(Visemes { aa: 0.5, ..Visemes::default() })],
            "invalid lines are dropped without ending the session"
        );
        assert!(wakes.load(Ordering::SeqCst) >= 1, "the UI thread is woken for new messages");
    }

    #[test]
    fn connecting_to_a_closed_port_fails_fast_without_blocking() {
        let closed = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap();
        let t = Instant::now();
        assert!(TcpStream::connect_timeout(&closed, CONNECT_TIMEOUT).is_err());
        assert!(t.elapsed() < Duration::from_secs(2));
    }
}
