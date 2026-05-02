use pyo3::prelude::*;
use pyo3::Bound;
use numpy::{PyArray1, IntoPyArray};

// ── Constants ──
const BLACK: u8 = 0;
const WHITE: u8 = 1;
const EMPTY: i8 = -1;
const BORING_NO_EAT_LIMIT: u32 = 50;
const UNUSED_BITS: u64 = 0b100000000100000000100000000100000000;
const BOARD_MASK: u64 = (1 << 36) - 1;
const REPETITION_LIMIT: usize = 12;

fn cell_bit(row: usize, col: usize) -> u64 {
    1 << (9 * row + col)
}

/// Return positions of set bits (LSB first), only looping over set bits.
/// Uses ARM `cls` instruction via u64::trailing_zeros.
fn set_bits(n: u64) -> Vec<u32> {
    let mut n = n;
    let mut out = Vec::new();
    while n != 0 {
        out.push(n.trailing_zeros());
        n &= n - 1;
    }
    out
}

#[derive(Clone)]
struct HistoryEntry {
    active: u8, passive: u8,
    forward: [u64; 2], backward: [u64; 2], pieces: [u64; 2],
    empty: u64, jump: bool,
    turn_count: u32, no_eat_count: u32,
}

#[pyclass]
#[derive(Clone)]
pub struct CheckerBoard {
    forward: [u64; 2], backward: [u64; 2], pieces: [u64; 2],
    active: u8, passive: u8,
    empty: u64, jump: bool,
    turn_count: u32, no_eat_count: u32,
    _history: Vec<HistoryEntry>,
}

#[pymethods]
impl CheckerBoard {
    #[new]
    pub fn new() -> Self {
        let mut cb = Self {
            forward: [0; 2], backward: [0; 2], pieces: [0; 2],
            active: 0, passive: 0, empty: 0, jump: false,
            turn_count: 0, no_eat_count: 0, _history: Vec::new(),
        };
        cb.new_game();
        cb
    }

    pub fn new_game(&mut self) {
        self.active = BLACK;
        self.passive = WHITE;
        self.forward[BLACK as usize] = 0x1eff;
        self.backward[BLACK as usize] = 0;
        self.pieces[BLACK as usize] = self.forward[BLACK as usize] | self.backward[BLACK as usize];
        self.forward[WHITE as usize] = 0;
        self.backward[WHITE as usize] = 0x7fbc00000;
        self.pieces[WHITE as usize] = self.forward[WHITE as usize] | self.backward[WHITE as usize];
        self.empty = UNUSED_BITS ^ BOARD_MASK ^ (self.pieces[BLACK as usize] | self.pieces[WHITE as usize]);
        self.jump = false;
        self.turn_count = 0;
        self.no_eat_count = 0;
        self._history.clear();
    }

    // ── Direction helpers ──

    fn right_forward(&self) -> u64 { (self.empty >> 4) & self.forward[self.active as usize] }
    fn left_forward(&self) -> u64 { (self.empty >> 5) & self.forward[self.active as usize] }
    fn right_backward(&self) -> u64 { (self.empty << 4) & self.backward[self.active as usize] }
    fn left_backward(&self) -> u64 { (self.empty << 5) & self.backward[self.active as usize] }
    fn right_forward_jumps(&self) -> u64 {
        (self.empty >> 8) & (self.pieces[self.passive as usize] >> 4) & self.forward[self.active as usize]
    }
    fn left_forward_jumps(&self) -> u64 {
        (self.empty >> 10) & (self.pieces[self.passive as usize] >> 5) & self.forward[self.active as usize]
    }
    fn right_backward_jumps(&self) -> u64 {
        (self.empty << 8) & (self.pieces[self.passive as usize] << 4) & self.backward[self.active as usize]
    }
    fn left_backward_jumps(&self) -> u64 {
        (self.empty << 10) & (self.pieces[self.passive as usize] << 5) & self.backward[self.active as usize]
    }

    // ── Move generation ──

    /// All jump moves available from current position (no mandatory-jump context).
    pub fn get_jumps(&self) -> Vec<i64> {
        let rfj = self.right_forward_jumps();
        let lfj = self.left_forward_jumps();
        let rbj = self.right_backward_jumps();
        let lbj = self.left_backward_jumps();
        if (rfj | lfj | rbj | lbj) == 0 { return Vec::new(); }
        let mut m = Vec::new();
        for i in set_bits(rfj) { m.push(-((0x101u64 << i) as i64)); }
        for i in set_bits(lfj) { m.push(-((0x401u64 << i) as i64)); }
        for i in set_bits(rbj) { m.push(-((0x101u64 << (i - 8)) as i64)); }
        for i in set_bits(lbj) { m.push(-((0x401u64 << (i - 10)) as i64)); }
        m
    }

    /// Non-jump regular moves.
    pub fn get_regular_moves(&self) -> Vec<i64> {
        let mut m = Vec::new();
        for i in set_bits(self.right_forward()) { m.push((0x11u64 << i) as i64); }
        for i in set_bits(self.left_forward()) { m.push((0x21u64 << i) as i64); }
        for i in set_bits(self.right_backward()) { m.push((0x11u64 << (i - 4)) as i64); }
        for i in set_bits(self.left_backward()) { m.push((0x21u64 << (i - 5)) as i64); }
        m
    }

    /// Returns true if any legal move exists (for is_over fast-path).
    /// Avoids allocating a Vec — just ORs bitboard results.
    pub fn has_any_moves(&self) -> bool {
        // Check jumps first
        let rfj = (self.empty >> 8) & (self.pieces[self.passive as usize] >> 4) & self.forward[self.active as usize];
        let lfj = (self.empty >> 10) & (self.pieces[self.passive as usize] >> 5) & self.forward[self.active as usize];
        let rbj = (self.empty << 8) & (self.pieces[self.passive as usize] << 4) & self.backward[self.active as usize];
        let lbj = (self.empty << 10) & (self.pieces[self.passive as usize] << 5) & self.backward[self.active as usize];
        if (rfj | lfj | rbj | lbj) != 0 { return true; }
        // Check regular moves
        self.right_forward() != 0 || self.left_forward() != 0
            || self.right_backward() != 0 || self.left_backward() != 0
    }

    /// All possible jumps from a specific piece bit.
    pub fn jumps_from(&self, piece: u64) -> Vec<i64> {
        let (rfj, lfj, rbj, lbj) = if self.active == BLACK {
            let r = (self.empty >> 8) & (self.pieces[self.passive as usize] >> 4) & piece;
            let l = (self.empty >> 10) & (self.pieces[self.passive as usize] >> 5) & piece;
            if (piece & self.backward[self.active as usize]) != 0 {
                (r, l,
                 (self.empty << 8) & (self.pieces[self.passive as usize] << 4) & piece,
                 (self.empty << 10) & (self.pieces[self.passive as usize] << 5) & piece)
            } else { (r, l, 0, 0) }
        } else {
            let rb = (self.empty << 8) & (self.pieces[self.passive as usize] << 4) & piece;
            let lb = (self.empty << 10) & (self.pieces[self.passive as usize] << 5) & piece;
            if (piece & self.forward[self.active as usize]) != 0 {
                ((self.empty >> 8) & (self.pieces[self.passive as usize] >> 4) & piece,
                 (self.empty >> 10) & (self.pieces[self.passive as usize] >> 5) & piece, rb, lb)
            } else { (0, 0, rb, lb) }
        };
        if (rfj | lfj | rbj | lbj) == 0 { return Vec::new(); }
        let mut m = Vec::new();
        for i in set_bits(rfj) { m.push(-((0x101u64 << i) as i64)); }
        for i in set_bits(lfj) { m.push(-((0x401u64 << i) as i64)); }
        for i in set_bits(rbj) { m.push(-((0x101u64 << (i - 8)) as i64)); }
        for i in set_bits(lbj) { m.push(-((0x401u64 << (i - 10)) as i64)); }
        m
    }

    // ── Make / push / pop ──

    /// Apply a move.  Returns (src_bit, dst_bit) for the caller to build
    /// a move string — avoids allocating a Python string in the hot path.
    pub fn make_move(&mut self, mv: i64) -> (u32, u32) {
        let move_abs = mv.unsigned_abs();
        let bits = set_bits(move_abs);
        let src_bit = bits[0];
        let dst_bit = bits[1];
        let active = self.active as usize;
        let passive = self.passive as usize;

        if mv < 0 {
            let mid = (src_bit + dst_bit) / 2;
            let taken = 1u64 << mid;
            self.pieces[passive] ^= taken;
            if taken & self.forward[passive] != 0 { self.forward[passive] ^= taken; }
            if taken & self.backward[passive] != 0 { self.backward[passive] ^= taken; }
            self.jump = true;
        }

        self.pieces[active] ^= move_abs;
        if move_abs & self.forward[active] != 0 { self.forward[active] ^= move_abs; }
        if move_abs & self.backward[active] != 0 { self.backward[active] ^= move_abs; }

        let dest = move_abs & self.pieces[active];
        self.empty = UNUSED_BITS ^ BOARD_MASK ^ (self.pieces[BLACK as usize] | self.pieces[WHITE as usize]);

        if self.jump { self.no_eat_count = 0; } else { self.no_eat_count += 1; }

        if active == BLACK as usize && (dest & 0x780000000) != 0 {
            self.backward[BLACK as usize] |= dest;
        } else if active == WHITE as usize && (dest & 0xf) != 0 {
            self.forward[WHITE as usize] |= dest;
        }

        self.jump = false;
        (src_bit, dst_bit)
    }

    /// Push with history.  Returns (src_bit, dst_bit) from make_move.
    pub fn push_move(&mut self, mv: i64) -> (u32, u32) {
        self._history.push(HistoryEntry {
            active: self.active, passive: self.passive,
            forward: self.forward, backward: self.backward, pieces: self.pieces,
            empty: self.empty, jump: self.jump,
            turn_count: self.turn_count, no_eat_count: self.no_eat_count,
        });
        self.make_move(mv)
    }

    pub fn pop_move(&mut self) {
        if let Some(e) = self._history.pop() {
            self.active = e.active; self.passive = e.passive;
            self.forward = e.forward; self.backward = e.backward; self.pieces = e.pieces;
            self.empty = e.empty; self.jump = e.jump;
            self.turn_count = e.turn_count; self.no_eat_count = e.no_eat_count;
        }
    }

    // ── State queries ──

    pub fn get_active(&self) -> u8 { self.active }
    pub fn get_passive(&self) -> u8 { self.passive }
    pub fn swap_active(&mut self) { self.active = self.passive; self.passive = if self.active == BLACK { WHITE } else { BLACK }; }
    pub fn get_turn_count(&self) -> u32 { self.turn_count }
    pub fn get_no_eat_count(&self) -> u32 { self.no_eat_count }
    pub fn get_jump_flag(&self) -> bool { self.jump }

    /// Returns numpy float32[32] weighted board position for NN evaluation.
    pub fn get_board_pos_weighted<'py>(&self, py: Python<'py>, colour: u8,
        w_empty: f32, w_black: f32, w_white: f32, w_bk: f32, w_wk: f32) -> Bound<'py, PyArray1<f32>>
    {
        let bk = self.backward[BLACK as usize];
        let bm = self.forward[BLACK as usize] ^ bk;
        let wk = self.forward[WHITE as usize];
        let wm = self.backward[WHITE as usize] ^ wk;
        let mut data = Vec::with_capacity(32);
        if colour == BLACK {
            for i in 0..4 { for j in 0..8 {
                let c = cell_bit(i, j);
                data.push(if c & bm != 0 { w_black } else if c & wm != 0 { w_white }
                    else if c & bk != 0 { w_bk } else if c & wk != 0 { w_wk } else { w_empty });
            }}
        } else {
            for i in (0..4).rev() { for j in (0..8).rev() {
                let c = cell_bit(i, j);
                data.push(if c & bm != 0 { w_white } else if c & wm != 0 { w_black }
                    else if c & bk != 0 { w_wk } else if c & wk != 0 { w_bk } else { w_empty });
            }}
        }
        data.into_pyarray(py)
    }

    /// True if any pieces of the given colour exist on the board.
    pub fn has_pieces(&self, colour: u8) -> bool {
        self.pieces[colour as usize] != 0
    }

    /// Returns numpy int8[32]: Black=0, White=1, empty=-1, blackKing=2, whiteKing=3
    pub fn get_rank<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i8>> {
        let bk = self.backward[BLACK as usize];
        let bm = self.forward[BLACK as usize] ^ bk;
        let wk = self.forward[WHITE as usize];
        let wm = self.backward[WHITE as usize] ^ wk;
        let mut data = Vec::with_capacity(32);
        for i in 0..4 { for j in 0..8 {
            let c = cell_bit(i, j);
            data.push(if c & bm != 0 { BLACK as i8 }
                else if c & wm != 0 { WHITE as i8 }
                else if c & bk != 0 { 2 }
                else if c & wk != 0 { 3 }
                else { EMPTY });
        }}
        data.into_pyarray(py)
    }

    pub fn copy(&self) -> Self { self.clone() }
}

#[pymodule]
fn checkers_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CheckerBoard>()?;
    Ok(())
}
