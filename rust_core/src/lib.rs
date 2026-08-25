//! THE OCTAGON — Rust inference core.
//!
//! STATUS: placeholder. The plan (see README roadmap) is a thin PyO3 module
//! that loads the ONNX-exported council and serves predictions faster than
//! the pure-Python path in inference_onnx/predict.py — but there is no
//! ONNX export yet to load, so this crate does nothing yet. Building it out
//! before that exists would just be dead code.

pub fn placeholder() -> &'static str {
    "octagon_core: not implemented yet — see README roadmap"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placeholder_returns_a_message() {
        assert!(!placeholder().is_empty());
    }
}
