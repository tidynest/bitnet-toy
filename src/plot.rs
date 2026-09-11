//! `plot <log>`: turn a training log into an SVG loss curve.
//!
//! The trainer prints one `step N   train_loss = X ...` line per
//! `log_every` steps and adds `val_loss = Y` on eval steps. This module
//! parses those lines back out and draws them. No dependencies: the SVG is
//! a handful of `<polyline>` and `<text>` elements written by hand, which
//! is enough for a curve you open in a browser or drop into a doc.

use std::fmt::Write;

/// Losses by step, as printed. `val` is usually much sparser than `train`.
#[derive(Debug, Default, PartialEq)]
pub struct Series {
    pub train: Vec<(usize, f32)>,
    pub val: Vec<(usize, f32)>,
}

/// Value of `key = <number>` in `line`, if present. The log fields are
/// whitespace-separated `name = value` pairs, so this is a substring find
/// plus a parse; no regex needed.
fn field(line: &str, key: &str) -> Option<f32> {
    let start = line.find(key)? + key.len();
    let rest = line[start..].trim_start().strip_prefix('=')?.trim_start();
    let end = rest.find(char::is_whitespace).unwrap_or(rest.len());
    rest[..end].parse().ok()
}

/// Parse every `step N   train_loss = ...` line. Lines that start with
/// `step` but carry no `train_loss` (checkpoint notices) are skipped.
pub fn parse_log(text: &str) -> Series {
    let mut s = Series::default();
    for line in text.lines() {
        let Some(rest) = line.strip_prefix("step") else {
            continue;
        };
        let Some(step) = rest.split_whitespace().next().and_then(|n| n.parse().ok()) else {
            continue;
        };
        if let Some(l) = field(line, "train_loss") {
            s.train.push((step, l));
        }
        if let Some(l) = field(line, "val_loss") {
            s.val.push((step, l));
        }
    }
    s
}

const W: f32 = 800.0;
const H: f32 = 400.0;
const PAD_L: f32 = 56.0;
const PAD_R: f32 = 16.0;
const PAD_T: f32 = 28.0;
const PAD_B: f32 = 40.0;

/// Render `series` as a standalone SVG. Returns `None` when there is
/// nothing to draw. Axes span the data: steps on x, loss on y, five ticks
/// each. Train is a solid line, val is dots joined by a dashed line.
pub fn render_svg(series: &Series, title: &str) -> Option<String> {
    let all = series.train.iter().chain(&series.val);
    let (mut x_max, mut y_min, mut y_max) = (0usize, f32::INFINITY, f32::NEG_INFINITY);
    for &(s, l) in all {
        x_max = x_max.max(s);
        y_min = y_min.min(l);
        y_max = y_max.max(l);
    }
    if y_min == f32::INFINITY {
        return None;
    }
    let x_max = x_max.max(1) as f32;
    let y_span = (y_max - y_min).max(1e-3);
    let sx = |s: usize| PAD_L + (s as f32 / x_max) * (W - PAD_L - PAD_R);
    let sy = |l: f32| H - PAD_B - ((l - y_min) / y_span) * (H - PAD_T - PAD_B);
    let points = |pts: &[(usize, f32)]| {
        pts.iter()
            .map(|&(s, l)| format!("{:.1},{:.1}", sx(s), sy(l)))
            .collect::<Vec<_>>()
            .join(" ")
    };

    let mut o = String::new();
    let _ = writeln!(
        o,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{W}\" height=\"{H}\" viewBox=\"0 0 {W} {H}\" font-family=\"sans-serif\" font-size=\"12\">"
    );
    let _ = writeln!(o, "<rect width=\"{W}\" height=\"{H}\" fill=\"white\"/>");
    let _ = writeln!(
        o,
        "<text x=\"{}\" y=\"18\" text-anchor=\"middle\" font-size=\"14\">{}</text>",
        W / 2.0,
        escape(title)
    );
    for i in 0..=4 {
        let t = i as f32 / 4.0;
        let (x, y) = (
            PAD_L + t * (W - PAD_L - PAD_R),
            H - PAD_B - t * (H - PAD_T - PAD_B),
        );
        let _ = writeln!(
            o,
            "<line x1=\"{PAD_L}\" y1=\"{y:.1}\" x2=\"{:.1}\" y2=\"{y:.1}\" stroke=\"#ddd\"/>",
            W - PAD_R
        );
        let _ = writeln!(
            o,
            "<text x=\"{:.1}\" y=\"{y:.1}\" text-anchor=\"end\" dominant-baseline=\"middle\">{:.3}</text>",
            PAD_L - 6.0,
            y_min + t * y_span
        );
        let _ = writeln!(
            o,
            "<text x=\"{x:.1}\" y=\"{:.1}\" text-anchor=\"middle\">{}</text>",
            H - PAD_B + 16.0,
            (t * x_max).round() as usize
        );
    }
    let _ = writeln!(
        o,
        "<polyline points=\"{}\" fill=\"none\" stroke=\"#3b6ea5\" stroke-width=\"1.5\"/>",
        points(&series.train)
    );
    if !series.val.is_empty() {
        let _ = writeln!(
            o,
            "<polyline points=\"{}\" fill=\"none\" stroke=\"#ce412b\" stroke-width=\"1.5\" stroke-dasharray=\"4 3\"/>",
            points(&series.val)
        );
        for &(s, l) in &series.val {
            let _ = writeln!(
                o,
                "<circle cx=\"{:.1}\" cy=\"{:.1}\" r=\"2.5\" fill=\"#ce412b\"/>",
                sx(s),
                sy(l)
            );
        }
    }
    let _ = writeln!(
        o,
        "<text x=\"{}\" y=\"{}\" fill=\"#3b6ea5\">train_loss</text><text x=\"{}\" y=\"{}\" fill=\"#ce412b\">val_loss</text>",
        W - PAD_R - 150.0,
        PAD_T + 4.0,
        W - PAD_R - 70.0,
        PAD_T + 4.0
    );
    o.push_str("</svg>\n");
    Some(o)
}

fn escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

#[cfg(test)]
mod tests {
    use super::*;

    const LOG: &str = "\
step     0   train_loss = 4.5288   anchor_loss = 4.4599   min_seen = 4.4575   val_loss = 4.5308   val_ppl = 92.83   lr = 0.0000e0   |g| = 6.529
step     0   best val_loss 4.5308 -> models/x.best.f32.bin
step   100   train_loss = 2.7745   anchor_loss = 4.0659   min_seen = 2.7448   lr = 7.5000e-4   |g| = 1.134
";

    /// Pins the parse against real trainer output: two train points, one
    /// val point, and the checkpoint notice contributes nothing.
    #[test]
    fn parses_train_and_val_and_skips_notices() {
        let s = parse_log(LOG);
        assert_eq!(s.train, vec![(0, 4.5288), (100, 2.7745)]);
        assert_eq!(s.val, vec![(0, 4.5308)]);
    }

    #[test]
    fn empty_log_renders_nothing() {
        assert!(render_svg(&parse_log("nothing here\n"), "t").is_none());
    }

    #[test]
    fn svg_carries_both_series_and_the_title() {
        let svg = render_svg(&parse_log(LOG), "a & b").unwrap();
        assert!(svg.starts_with("<svg "));
        assert!(svg.contains("a &amp; b"));
        assert_eq!(svg.matches("<polyline").count(), 2);
        assert_eq!(svg.matches("<circle").count(), 1);
    }
}
