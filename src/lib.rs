use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::Bound;
use numpy::{PyArray1, IntoPyArray};
use std::collections::HashSet;

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

fn set_bits(n: u64) -> Vec<u32> {
    let mut n = n;
    let mut out = Vec::new();
    while n != 0 {
        out.push(n.trailing_zeros());
        n &= n - 1;
    }
    out
}

fn format_move_str(src_bit: u32, dst_bit: u32, is_jump: bool) -> String {
    let src = 1 + src_bit - src_bit / 9;
    let dst = 1 + dst_bit - dst_bit / 9;
    if is_jump {
        format!("{}x{}", src, dst)
    } else {
        format!("{}-{}", src, dst)
    }
}

#[derive(Clone)]
struct HistoryEntry {
    active: u8, passive: u8,
    forward: [u64; 2], backward: [u64; 2], pieces: [u64; 2],
    empty: u64, jump: bool,
    turn_count: u32, no_eat_count: u32,
    mandatory_jumps: Vec<i64>,
    multiple_jump_stack: Vec<String>,
    alt_move_stack_len: usize,
    moves_len: usize,
    winner: i8,
}

#[pyclass]
#[derive(Clone)]
pub struct CheckerBoard {
    forward: [u64; 2], backward: [u64; 2], pieces: [u64; 2],
    active: u8, passive: u8,
    empty: u64, jump: bool,
    turn_count: u32, no_eat_count: u32,
    mandatory_jumps: Vec<i64>,
    multiple_jump_stack: Vec<String>,
    alt_move_stack: Vec<(u8, String)>,
    moves: Vec<(u8, String)>,
    winner: i8,
    _history: Vec<HistoryEntry>,
}

#[pymethods]
impl CheckerBoard {
    #[new]
    pub fn new() -> Self {
        let mut cb = Self {
            forward: [0; 2], backward: [0; 2], pieces: [0; 2],
            active: 0, passive: 0, empty: 0, jump: false,
            turn_count: 0, no_eat_count: 0,
            mandatory_jumps: Vec::new(),
            multiple_jump_stack: Vec::new(),
            alt_move_stack: Vec::new(),
            moves: Vec::new(),
            winner: -1,
            _history: Vec::new(),
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
        self.mandatory_jumps.clear();
        self.multiple_jump_stack.clear();
        self.alt_move_stack.clear();
        self.moves.clear();
        self.winner = -1;
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

    /// All jump moves available from current position.
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

    /// All legal moves, respecting mandatory jump context.
    pub fn get_moves(&self) -> Vec<i64> {
        if self.jump && !self.mandatory_jumps.is_empty() {
            return self.mandatory_jumps.clone();
        }
        let jumps = self.get_jumps();
        if !jumps.is_empty() {
            return jumps;
        }
        self.get_regular_moves()
    }

    /// Formatted strings for all legal moves (for display / preload).
    pub fn get_move_strings(&self) -> Vec<String> {
        let moves = self.get_moves();
        let mut result = Vec::with_capacity(moves.len());
        for &mv in &moves {
            let move_abs = mv.unsigned_abs();
            let bits = set_bits(move_abs);
            if bits.len() >= 2 {
                result.push(format_move_str(bits[0], bits[1], mv < 0));
            }
        }
        result
    }

    /// Returns true if any legal move exists.
    pub fn has_any_moves(&self) -> bool {
        if self.jump && !self.mandatory_jumps.is_empty() {
            return true;
        }
        let rfj = (self.empty >> 8) & (self.pieces[self.passive as usize] >> 4) & self.forward[self.active as usize];
        let lfj = (self.empty >> 10) & (self.pieces[self.passive as usize] >> 5) & self.forward[self.active as usize];
        let rbj = (self.empty << 8) & (self.pieces[self.passive as usize] << 4) & self.backward[self.active as usize];
        let lbj = (self.empty << 10) & (self.pieces[self.passive as usize] << 5) & self.backward[self.active as usize];
        if (rfj | lfj | rbj | lbj) != 0 { return true; }
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

    /// Apply a move. Handles all game state: bitboard ops, move string generation,
    /// multi-jump tracking, king promotion, and player switching.
    pub fn make_move(&mut self, mv: i64) {
        let move_abs = mv.unsigned_abs();
        let bits = set_bits(move_abs);
        let src_bit = bits[0];
        let dst_bit = bits[1];
        let is_jump = mv < 0;
        let move_str = format_move_str(src_bit, dst_bit, is_jump);

        let active = self.active as usize;
        let passive = self.passive as usize;

        // Move the piece
        self.pieces[active] ^= move_abs;
        if move_abs & self.forward[active] != 0 { self.forward[active] ^= move_abs; }
        if move_abs & self.backward[active] != 0 { self.backward[active] ^= move_abs; }

        // Capture if jump
        if is_jump {
            let mid = (src_bit + dst_bit) / 2;
            let taken = 1u64 << mid;
            self.pieces[passive] ^= taken;
            if taken & self.forward[passive] != 0 { self.forward[passive] ^= taken; }
            if taken & self.backward[passive] != 0 { self.backward[passive] ^= taken; }
            self.jump = true;
            self.no_eat_count = 0;
        } else {
            self.no_eat_count += 1;
        }

        let dest = 1u64 << dst_bit;
        self.empty = UNUSED_BITS ^ BOARD_MASK ^ (self.pieces[BLACK as usize] | self.pieces[WHITE as usize]);

        // King promotion
        if active == BLACK as usize && (dest & 0x780000000) != 0 {
            self.backward[BLACK as usize] |= dest;
        } else if active == WHITE as usize && (dest & 0xf) != 0 {
            self.forward[WHITE as usize] |= dest;
        }

        // Store move in logs
        let colour = self.active;
        self.alt_move_stack.push((colour, move_str.clone()));
        self.moves.push((colour, move_str.clone()));

        // Handle multi-jump
        if is_jump {
            self.mandatory_jumps = self.jumps_from(dest);
            if !self.mandatory_jumps.is_empty() {
                let src_str = format!("{}", 1 + src_bit - src_bit / 9);
                let dst_str = format!("{}", 1 + dst_bit - dst_bit / 9);
                self.multiple_jump_stack.push(src_str);
                self.multiple_jump_stack.push(dst_str);
                return; // No player switch — mandatory jumps remain
            }
            self.multiple_jump_stack.clear();
            self.jump = false;
        }

        // Player switch
        self.active = self.passive;
        self.passive = if self.active == BLACK { WHITE } else { BLACK };
    }

    /// Push move with full state history.
    pub fn push_move(&mut self, mv: i64) {
        self._history.push(HistoryEntry {
            active: self.active, passive: self.passive,
            forward: self.forward, backward: self.backward, pieces: self.pieces,
            empty: self.empty, jump: self.jump,
            turn_count: self.turn_count, no_eat_count: self.no_eat_count,
            mandatory_jumps: self.mandatory_jumps.clone(),
            multiple_jump_stack: self.multiple_jump_stack.clone(),
            alt_move_stack_len: self.alt_move_stack.len(),
            moves_len: self.moves.len(),
            winner: self.winner,
        });
        self.make_move(mv);
    }

    /// Pop move, restoring full state from history.
    pub fn pop_move(&mut self) {
        if let Some(e) = self._history.pop() {
            self.active = e.active; self.passive = e.passive;
            self.forward = e.forward; self.backward = e.backward; self.pieces = e.pieces;
            self.empty = e.empty; self.jump = e.jump;
            self.turn_count = e.turn_count; self.no_eat_count = e.no_eat_count;
            self.mandatory_jumps = e.mandatory_jumps;
            self.multiple_jump_stack = e.multiple_jump_stack;
            self.alt_move_stack.truncate(e.alt_move_stack_len);
            self.moves.truncate(e.moves_len);
            self.winner = e.winner;
        }
    }

    // ── Game state ──

    /// Check if the game is over. Sets `winner` if terminal.
    pub fn is_over(&mut self, check_repetition: bool) -> bool {
        if self.no_eat_count >= BORING_NO_EAT_LIMIT {
            self.winner = -2; // draw by no-capture limit
            return true;
        }
        if !self.has_any_moves() {
            // Active player has no legal moves — they lose
            self.winner = self.passive as i8;
            return true;
        }
        if check_repetition && self.alt_move_stack.len() >= REPETITION_LIMIT {
            let start = self.alt_move_stack.len() - REPETITION_LIMIT;
            let mut p1: HashSet<&str> = HashSet::new();
            let mut p2: HashSet<&str> = HashSet::new();
            for (i, (_, m)) in self.alt_move_stack[start..].iter().enumerate() {
                if i % 2 == 0 { p1.insert(m); } else { p2.insert(m); }
            }
            if p1.len() < 4 && p2.len() < 4 {
                self.winner = -2; // draw by repetition
                return true;
            }
        }
        false
    }

    /// Return the winner: -1 = none, 0 = black, 1 = white, -2 = draw.
    pub fn get_winner(&self) -> i8 { self.winner }
    pub fn set_winner(&mut self, w: i8) { self.winner = w; }

    /// Get the full move log as a list of (colour, move_string) tuples.
    pub fn get_move_log(&self) -> Vec<(u8, String)> {
        self.alt_move_stack.clone()
    }

    /// Get PDN-formatted move strings (multi-jumps grouped).
    pub fn get_pdn_moves(&self) -> Vec<String> {
        let mut result: Vec<String> = Vec::new();
        let mut i = 0;
        while i < self.alt_move_stack.len() {
            let (_, ref m) = self.alt_move_stack[i];
            if m.contains('x') {
                let mut jumps: Vec<String> = Vec::new();
                jumps.push(m.clone());
                i += 1;
                while i < self.alt_move_stack.len() {
                    let (_, ref next) = self.alt_move_stack[i];
                    if next.contains('x') {
                        jumps.push(next.clone());
                        i += 1;
                    } else {
                        break;
                    }
                }
                if jumps.len() > 1 {
                    // Group consecutive jumps: "9x14" + "14x21" → "9x14x21"
                    let first_x = jumps[0].find('x').unwrap_or(jumps[0].len());
                    let mut grouped = jumps[0][..first_x].to_string();
                    for jmp in &jumps {
                        if let Some(x_pos) = jmp.find('x') {
                            grouped.push('x');
                            grouped.push_str(&jmp[x_pos + 1..]);
                        }
                    }
                    result.push(grouped);
                } else {
                    result.push(jumps[0].clone());
                }
            } else {
                result.push(m.clone());
                i += 1;
            }
        }
        result
    }

    // ── State queries ──

    pub fn get_active(&self) -> u8 { self.active }
    pub fn get_passive(&self) -> u8 { self.passive }
    pub fn get_turn_count(&self) -> u32 { self.turn_count }
    pub fn inc_turn_count(&mut self) { self.turn_count += 1; }
    pub fn get_no_eat_count(&self) -> u32 { self.no_eat_count }
    pub fn set_no_eat_count(&mut self, v: u32) { self.no_eat_count = v; }
    pub fn get_jump_flag(&self) -> bool { self.jump }
    pub fn get_pieces(&self) -> [u64; 2] { self.pieces }
    pub fn get_forward(&self) -> [u64; 2] { self.forward }
    pub fn get_backward(&self) -> [u64; 2] { self.backward }
    pub fn get_mandatory_jumps(&self) -> Vec<i64> { self.mandatory_jumps.clone() }
    pub fn swap_active(&mut self) {
        self.active = self.passive;
        self.passive = if self.active == BLACK { WHITE } else { BLACK };
    }

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

    // ── Clone / pickle ──

    pub fn copy(&self) -> Self { self.clone() }

    pub fn __getstate__(&self, py: Python<'_>) -> PyResult<PyObject> {
        let d = PyDict::new(py);
        d.set_item("forward", vec![self.forward[0], self.forward[1]])?;
        d.set_item("backward", vec![self.backward[0], self.backward[1]])?;
        d.set_item("pieces", vec![self.pieces[0], self.pieces[1]])?;
        d.set_item("active", self.active)?;
        d.set_item("passive", self.passive)?;
        d.set_item("empty", self.empty)?;
        d.set_item("jump", self.jump)?;
        d.set_item("turn_count", self.turn_count)?;
        d.set_item("no_eat_count", self.no_eat_count)?;
        d.set_item("winner", self.winner)?;
        d.set_item("mandatory_jumps", self.mandatory_jumps.clone())?;
        d.set_item("multiple_jump_stack", self.multiple_jump_stack.clone())?;
        let moves: Vec<(u8, String)> = self.moves.clone();
        d.set_item("moves", moves)?;
        Ok(d.into())
    }

    pub fn __setstate__(&mut self, state: Bound<'_, PyDict>) -> PyResult<()> {
        self.new_game();
        macro_rules! getk {
            ($d:expr, $k:expr) => { $d.get_item($k)?.expect(stringify!($k)) };
        }
        let fwd: Vec<u64> = getk!(state, "forward").extract()?;
        self.forward = [fwd[0], fwd[1]];
        let bwd: Vec<u64> = getk!(state, "backward").extract()?;
        self.backward = [bwd[0], bwd[1]];
        let pcs: Vec<u64> = getk!(state, "pieces").extract()?;
        self.pieces = [pcs[0], pcs[1]];
        self.active = getk!(state, "active").extract()?;
        self.passive = getk!(state, "passive").extract()?;
        self.empty = getk!(state, "empty").extract()?;
        self.jump = getk!(state, "jump").extract()?;
        self.turn_count = getk!(state, "turn_count").extract()?;
        self.no_eat_count = getk!(state, "no_eat_count").extract()?;
        self.winner = getk!(state, "winner").extract()?;
        self.mandatory_jumps = getk!(state, "mandatory_jumps").extract()?;
        self.multiple_jump_stack = getk!(state, "multiple_jump_stack").extract()?;
        self.moves = getk!(state, "moves").extract()?;
        Ok(())
    }
}

#[pymodule]
fn checkers_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CheckerBoard>()?;
    Ok(())
}
