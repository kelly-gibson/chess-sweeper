use std::sync::Arc;
use futures::executor::block_on;
use rand::Rng;
use rand::prelude::*;
use winit::window::{Window, WindowAttributes};
use winit::event_loop::EventLoop;
use winit::dpi::LogicalSize;
use std::collections::HashMap;
use winit::dpi::PhysicalSize;

mod renderer;
use renderer::Renderer;

mod rules;
use rules::{ChessRules, PieceType, Player, Position, BoardStateProvider, Piece, MoveHighlighter, GameState, CaptureInfo};

#[derive(Clone, Copy, PartialEq, Debug)]
enum CellState {
    Empty,                // 0.0
    Revealed,             // 0.5
    // Different mine types with different blast patterns
    SmallMine,            // -1.0  (Small blast radius)
    MediumMine,           // -1.1  (Medium blast radius)
    LargeMine,            // -1.2  (Large blast radius)
    HugeMine,             // -1.3  (Huge blast radius)
    Flagged,              // 1.5
    RevealedMine,         // -2.0
    Highlighted,          // 4.0
    // Player pieces (2.0-2.5)
    PlayerKing,           // 2.0
    PlayerQueen,          // 2.1
    PlayerBishop,         // 2.2
    PlayerKnight,         // 2.3
    PlayerRook,           // 2.4
    PlayerPawn,           // 2.5
    // Opponent pieces (3.0-3.5)
    OpponentKing,         // 3.0
    OpponentQueen,        // 3.1
    OpponentBishop,       // 3.2
    OpponentKnight,       // 3.3
    OpponentRook,         // 3.4
    OpponentPawn,         // 3.5
}

impl CellState {
    // Convert to float value for rendering
    fn to_value(&self) -> f32 {
        match self {
            CellState::SmallMine => -1.0,
            CellState::MediumMine => -1.1,
            CellState::LargeMine => -1.2,
            CellState::HugeMine => -1.3,
            CellState::Empty => 0.0,
            CellState::Revealed => 0.5,
            CellState::Flagged => 1.5,
            CellState::RevealedMine => -2.0,
            CellState::Highlighted => 4.0,
            // Player pieces
            CellState::PlayerKing => 2.0,
            CellState::PlayerQueen => 2.1,
            CellState::PlayerBishop => 2.2,
            CellState::PlayerKnight => 2.3,
            CellState::PlayerRook => 2.4,
            CellState::PlayerPawn => 2.5,
            // Opponent pieces
            CellState::OpponentKing => 3.0,
            CellState::OpponentQueen => 3.1,
            CellState::OpponentBishop => 3.2,
            CellState::OpponentKnight => 3.3,
            CellState::OpponentRook => 3.4,
            CellState::OpponentPawn => 3.5,
        }
    }
    
    // Convert from float value to cell state
    fn from_value(value: f32) -> Self {
        if value <= -1.5 {
            CellState::RevealedMine
        } else if value <= -1.25 {
            CellState::HugeMine
        } else if value <= -1.15 {
            CellState::LargeMine
        } else if value <= -1.05 {
            CellState::MediumMine
        } else if value <= -0.5 {
            CellState::SmallMine
        } else if value <= 0.25 {
            CellState::Empty
        } else if value <= 0.75 {
            CellState::Revealed
        } else if value <= 1.75 {
            CellState::Flagged
        } else if value < 2.6 {
            // Player pieces (2.0-2.5)
            let decimal = (value * 10.0).round() as i32 % 10;
            match decimal {
                0 => CellState::PlayerKing,
                1 => CellState::PlayerQueen,
                2 => CellState::PlayerBishop,
                3 => CellState::PlayerKnight,
                4 => CellState::PlayerRook,
                _ => CellState::PlayerPawn,
            }
        } else if value < 3.6 {
            // Opponent pieces (3.0-3.5)
            let decimal = (value * 10.0).round() as i32 % 10;
            match decimal {
                0 => CellState::OpponentKing,
                1 => CellState::OpponentQueen,
                2 => CellState::OpponentBishop,
                3 => CellState::OpponentKnight,
                4 => CellState::OpponentRook,
                _ => CellState::OpponentPawn,
            }
        } else {
            CellState::Highlighted
        }
    }
    
    // Helper method to check if a cell state is any type of mine
    fn is_mine(&self) -> bool {
        matches!(self, CellState::SmallMine | CellState::MediumMine | 
                       CellState::LargeMine | CellState::HugeMine)
    }
    
    // Get the blast radius for a mine type
    fn blast_radius(&self) -> usize {
        match self {
            CellState::SmallMine => 1,  // 1 tile radius (3x3 area)
            CellState::MediumMine => 2, // 2 tile radius (5x5 area)
            CellState::LargeMine => 3,  // 3 tile radius (7x7 area)
            CellState::HugeMine => 4,   // 4 tile radius (9x9 area)
            _ => 0, // Not a mine or unknown type
        }
    }
    
    // Get the armor penetration value for future implementation
    fn armor_penetration(&self) -> u8 {
        match self {
            CellState::SmallMine => 1,  // Low armor penetration
            CellState::MediumMine => 2, // Medium armor penetration
            CellState::LargeMine => 3,  // High armor penetration
            CellState::HugeMine => 4,   // Extreme armor penetration
            _ => 0, // Not a mine or unknown type
        }
    }
}

struct Board {
    cells: Vec<CellState>,
    width: usize,
    height: usize,
    game_over: bool,
    selected_piece: Option<(usize, usize)>,
    chess_rules: ChessRules,
    move_highlighter: MoveHighlighter,
    // Track piece types for each position
    piece_types: HashMap<Position, PieceType>,
    // Add game state for turn management
    game_state: GameState,
    // Store last captured piece for display
    last_capture: Option<CaptureInfo>,
    // Track highlighted cells separately
    highlights: Vec<bool>,
}

// Implement BoardStateProvider trait for Board
impl BoardStateProvider for Board {
    fn is_empty(&self, position: Position) -> bool {
        if position.x >= self.width || position.y >= self.height {
            return false;
        }
        
        let idx = position.y * self.width + position.x;
        // Consider mines as "empty" for chess rule purposes so pieces can move there
        // This prevents revealing mines through the highlighting system
        matches!(self.cells[idx], CellState::Empty | CellState::Revealed | CellState::Highlighted | 
                                  CellState::SmallMine | CellState::MediumMine | 
                                  CellState::LargeMine | CellState::HugeMine)
    }
    
    fn has_opponent_piece(&self, position: Position, player: Player) -> bool {
        if position.x >= self.width || position.y >= self.height {
            return false;
        }
        
        let idx = position.y * self.width + position.x;
        match (player, &self.cells[idx]) {
            (Player::Human, CellState::OpponentKing | CellState::OpponentQueen | 
                           CellState::OpponentBishop | CellState::OpponentKnight | 
                           CellState::OpponentRook | CellState::OpponentPawn) => true,
            (Player::Opponent, CellState::PlayerKing | CellState::PlayerQueen | 
                              CellState::PlayerBishop | CellState::PlayerKnight | 
                              CellState::PlayerRook | CellState::PlayerPawn) => true,
            _ => false,
        }
    }
    
    fn get_piece_at(&self, position: Position) -> Option<Piece> {
        if position.x >= self.width || position.y >= self.height {
            return None;
        }
        
        let idx = position.y * self.width + position.x;
        match self.cells[idx] {
            CellState::PlayerKing => Some(Piece { piece_type: PieceType::King, player: Player::Human }),
            CellState::PlayerQueen => Some(Piece { piece_type: PieceType::Queen, player: Player::Human }),
            CellState::PlayerBishop => Some(Piece { piece_type: PieceType::Bishop, player: Player::Human }),
            CellState::PlayerKnight => Some(Piece { piece_type: PieceType::Knight, player: Player::Human }),
            CellState::PlayerRook => Some(Piece { piece_type: PieceType::Rook, player: Player::Human }),
            CellState::PlayerPawn => Some(Piece { piece_type: PieceType::Pawn, player: Player::Human }),
            CellState::OpponentKing => Some(Piece { piece_type: PieceType::King, player: Player::Opponent }),
            CellState::OpponentQueen => Some(Piece { piece_type: PieceType::Queen, player: Player::Opponent }),
            CellState::OpponentBishop => Some(Piece { piece_type: PieceType::Bishop, player: Player::Opponent }),
            CellState::OpponentKnight => Some(Piece { piece_type: PieceType::Knight, player: Player::Opponent }),
            CellState::OpponentRook => Some(Piece { piece_type: PieceType::Rook, player: Player::Opponent }),
            CellState::OpponentPawn => Some(Piece { piece_type: PieceType::Pawn, player: Player::Opponent }),
            _ => None,
        }
    }
}

impl Board {
    fn new(width: usize, height: usize) -> Self {
        let mut cells = vec![CellState::Empty; width * height];
        let chess_rules = ChessRules::new(width, height);
        let move_highlighter = MoveHighlighter::new();
        let mut piece_types = HashMap::new(); // Keep for compatibility with BoardStateProvider

        // Define chess board dimensions within our larger grid
        // Standard chess is 8x8, so we'll place it centered in our 24x24 grid
        let chess_start_x = (width - 8) / 2; // Center the 8x8 chess board horizontally
        let chess_width = 8; // Standard chess board width
        
        // Place mines randomly (5% chance for easier gameplay on the large board)
        let mut rng = rand::rng();
        for y in 0..height {
            for x in 0..width {
                let i = y * width + x;
                
                // Don't place mines in the chess area
                let is_in_chess_area = 
                    x >= chess_start_x && x < chess_start_x + chess_width && 
                    (y < 2 || y >= height - 2); // Chess pieces are in first and last 2 rows
                
                if !is_in_chess_area && rng.random_bool(0.05) {
                    // Use the random float value to determine mine type
                    let random_value: f32 = rng.random();
                    let mine_type = if random_value < 0.25 {
                        CellState::SmallMine
                    } else if random_value < 0.5 {
                        CellState::MediumMine
                    } else if random_value < 0.75 {
                        CellState::LargeMine
                    } else {
                        CellState::HugeMine
                    };
                    cells[i] = mine_type;
                }
            }
        }
        
        // Place standard chess pieces for the player at the bottom of the board
        // Pawns in the second-to-last row
        for x in 0..8 {
            let actual_x = chess_start_x + x;
            cells[(height - 2) * width + actual_x] = CellState::PlayerPawn;
            piece_types.insert(Position::new(actual_x, height - 2), PieceType::Pawn);
        }
        
        // Main pieces in the last row according to standard chess arrangement
        // Rooks at corners
        cells[(height - 1) * width + chess_start_x] = CellState::PlayerRook; // Left rook
        cells[(height - 1) * width + (chess_start_x + 7)] = CellState::PlayerRook; // Right rook
        
        // Knights next to rooks
        cells[(height - 1) * width + (chess_start_x + 1)] = CellState::PlayerKnight; // Left knight
        cells[(height - 1) * width + (chess_start_x + 6)] = CellState::PlayerKnight; // Right knight
        
        // Bishops next to knights
        cells[(height - 1) * width + (chess_start_x + 2)] = CellState::PlayerBishop; // Left bishop
        cells[(height - 1) * width + (chess_start_x + 5)] = CellState::PlayerBishop; // Right bishop
        
        // Queen and King in the middle
        cells[(height - 1) * width + (chess_start_x + 3)] = CellState::PlayerQueen; // Queen
        cells[(height - 1) * width + (chess_start_x + 4)] = CellState::PlayerKing; // King
        
        // Place opponent pieces at the top of the board with the same standard arrangement
        // Pawns in the second row
        for x in 0..8 {
            let actual_x = chess_start_x + x;
            cells[1 * width + actual_x] = CellState::OpponentPawn;
            piece_types.insert(Position::new(actual_x, 1), PieceType::Pawn);
        }
        
        // Main pieces in the first row
        // Rooks at corners
        cells[0 * width + chess_start_x] = CellState::OpponentRook; // Left rook
        cells[0 * width + (chess_start_x + 7)] = CellState::OpponentRook; // Right rook
        
        // Knights next to rooks
        cells[0 * width + (chess_start_x + 1)] = CellState::OpponentKnight; // Left knight
        cells[0 * width + (chess_start_x + 6)] = CellState::OpponentKnight; // Right knight
        
        // Bishops next to knights
        cells[0 * width + (chess_start_x + 2)] = CellState::OpponentBishop; // Left bishop
        cells[0 * width + (chess_start_x + 5)] = CellState::OpponentBishop; // Right bishop
        
        // Queen and King in the middle
        cells[0 * width + (chess_start_x + 3)] = CellState::OpponentQueen; // Queen
        cells[0 * width + (chess_start_x + 4)] = CellState::OpponentKing; // King
        
        // Update piece_types map for the updated cell states
        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                let pos = Position::new(x, y);
                
                match cells[idx] {
                    CellState::PlayerKing => piece_types.insert(pos, PieceType::King),
                    CellState::PlayerQueen => piece_types.insert(pos, PieceType::Queen),
                    CellState::PlayerBishop => piece_types.insert(pos, PieceType::Bishop),
                    CellState::PlayerKnight => piece_types.insert(pos, PieceType::Knight),
                    CellState::PlayerRook => piece_types.insert(pos, PieceType::Rook),
                    CellState::PlayerPawn => piece_types.insert(pos, PieceType::Pawn),
                    CellState::OpponentKing => piece_types.insert(pos, PieceType::King),
                    CellState::OpponentQueen => piece_types.insert(pos, PieceType::Queen),
                    CellState::OpponentBishop => piece_types.insert(pos, PieceType::Bishop),
                    CellState::OpponentKnight => piece_types.insert(pos, PieceType::Knight),
                    CellState::OpponentRook => piece_types.insert(pos, PieceType::Rook),
                    CellState::OpponentPawn => piece_types.insert(pos, PieceType::Pawn),
                    _ => None
                };
            }
        }
        
        // Initialize game state with Human as the starting player
        let game_state = GameState::new(Player::Human);
        let last_capture = None;
        
        Self {
            cells,
            width,
            height,
            game_over: false,
            selected_piece: None,
            chess_rules,
            move_highlighter,
            piece_types,
            game_state,
            last_capture,
            highlights: vec![false; width * height],
        }
    }
    
    // Convert board to values for rendering
    fn to_values(&self) -> Vec<f32> {
        let mut values = Vec::with_capacity(self.width * self.height);
        
        for i in 0..self.cells.len() {
            if i < self.highlights.len() && self.highlights[i] {
                values.push(4.0); // Use 4.0 for highlighted cells
            } else {
                values.push(self.cells[i].to_value());
            }
        }
        
        values
    }
    
    // Select a piece at coordinates and show legal moves
    fn select_piece(&mut self, x: usize, y: usize) -> bool {
        if self.game_over || x >= self.width || y >= self.height {
            return false;
        }
        
        // Get current player from game state
        let current_player = match self.game_state.current_player() {
            Some(player) => player,
            None => return false, // Game is not in progress
        };
        
        let idx = y * self.width + x;
        let position = Position::new(x, y);
        
        // Check if there's a piece at this location belonging to the current player
        let is_current_player_piece = match current_player {
            Player::Human => matches!(self.cells[idx], 
                CellState::PlayerKing | CellState::PlayerQueen | 
                CellState::PlayerBishop | CellState::PlayerKnight | 
                CellState::PlayerRook | CellState::PlayerPawn),
            Player::Opponent => matches!(self.cells[idx], 
                CellState::OpponentKing | CellState::OpponentQueen | 
                CellState::OpponentBishop | CellState::OpponentKnight | 
                CellState::OpponentRook | CellState::OpponentPawn),
        };
        
        if is_current_player_piece {
            // Clear any previous highlights
            self.clear_highlights();
            
            // Store the selected piece position
            self.selected_piece = Some((x, y));
            
            // Get the piece type
            let piece_type = self.get_piece_at(position)
                .map(|piece| piece.piece_type)
                .unwrap_or(PieceType::Pawn); // Fallback if not found
            
            // Get valid moves for this piece
            let valid_moves = self.chess_rules.get_valid_moves(
                piece_type,
                current_player,
                position,
                self
            );
            
            eprintln!("Found {} valid moves for {} at ({}, {})", 
                     valid_moves.len(), format!("{:?}", piece_type), x, y);
            
            // Update the move highlighter
            self.move_highlighter.set_valid_moves(valid_moves);
            
            // Highlight all valid moves on the board
            self.highlight_valid_moves();
            
            // Debug: Count how many cells are highlighted
            let highlight_count = self.highlights.iter().filter(|&&h| h).count();
            eprintln!("Highlighted {} cells", highlight_count);
            
            eprintln!("Selected {} at ({}, {})", format!("{:?}", piece_type), x, y);
            return true;
        }
        
        false
    }
    
    // Move a selected piece to new coordinates
    fn move_piece(&mut self, to_x: usize, to_y: usize) -> bool {
        if self.game_over || to_x >= self.width || to_y >= self.height {
            return false;
        }
        
        // Get current player
        let current_player = match self.game_state.current_player() {
            Some(player) => player,
            None => return false, // Game not in progress
        };
        
        // Check if a piece is selected
        if let Some((from_x, from_y)) = self.selected_piece {
            // Check if the target is different from the starting point
            if from_x == to_x && from_y == to_y {
                return false;
            }
            
            // Check if the move is valid according to chess rules
            let to_pos = Position::new(to_x, to_y);
            if !self.move_highlighter.is_valid_move(to_pos) {
                eprintln!("Invalid move to ({}, {})", to_x, to_y);
                return false;
            }
            
            let from_pos = Position::new(from_x, from_y);
            let from_idx = from_y * self.width + from_x;
            let to_idx = to_y * self.width + to_x;
            
            // Get the piece type of the moving piece
            let piece = self.get_piece_at(from_pos).unwrap();
            let piece_type = piece.piece_type;
            
            // For sliding pieces (bishop, rook, queen), check for mines along the path
            // Knights are exempt as they jump over pieces and mines
            if piece_type != PieceType::Knight && 
               (piece_type == PieceType::Bishop || piece_type == PieceType::Rook || piece_type == PieceType::Queen) {
                // Calculate movement direction
                let dx = if to_x > from_x { 1 } else if to_x < from_x { -1 } else { 0 };
                let dy = if to_y > from_y { 1 } else if to_y < from_y { -1 } else { 0 };
                
                // Check each cell along the path, excluding start and end positions
                let mut check_x = from_x as i32 + dx;
                let mut check_y = from_y as i32 + dy;
                
                while (check_x != to_x as i32 || check_y != to_y as i32) && 
                      check_x >= 0 && check_x < self.width as i32 && 
                      check_y >= 0 && check_y < self.height as i32 {
                    
                    let check_idx = check_y as usize * self.width + check_x as usize;
                    
                    // If there's a mine along the path, detonate it
                    if self.cells[check_idx].is_mine() {
                        eprintln!("Mine detected along movement path at ({}, {})", check_x, check_y);
                        
                        // Get the moving piece info for the log
                        let moving_piece_info = format!("{:?} ({:?})", piece_type, current_player);
                        
                        // Get the mine type before removing the piece
                        let mine_type = self.cells[check_idx];
                        
                        // Remove the moving piece from its original position
                        self.cells[from_idx] = CellState::Empty;
                        self.piece_types.remove(&from_pos);
                        
                        // Clear selection and highlights
                        self.selected_piece = None;
                        self.clear_highlights();
                        
                        // Trigger the mine explosion
                        self.cells[check_idx] = CellState::RevealedMine;
                        self.create_explosion(check_x as usize, check_y as usize, mine_type);
                        
                        // Switch to next player's turn
                        if self.game_state.is_in_progress() {
                            self.game_state.next_turn();
                            eprintln!("BOOM! {} triggered a {:?} mine at ({}, {}) while moving and was destroyed!", 
                                     moving_piece_info, mine_type, check_x, check_y);
                            eprintln!("{}", self.game_state.status_message());
                        }
                        
                        return true;
                    }
                    
                    // Move to next cell in the path
                    check_x += dx;
                    check_y += dy;
                }
            }
            
            // Check if the target cell contains a mine (applies to all pieces)
            if self.cells[to_idx].is_mine() {
                // Get the mine type before clearing selection
                let mine_type = self.cells[to_idx];
                
                // Clear selection and highlights
                self.selected_piece = None;
                self.clear_highlights();
                
                // Trigger the mine explosion
                self.cells[to_idx] = CellState::RevealedMine;
                self.create_explosion(to_x, to_y, mine_type);
                
                // Switch to next player's turn
                if self.game_state.is_in_progress() {
                    self.game_state.next_turn();
                    eprintln!("BOOM! {} stepped on a {:?} mine at ({}, {})!", current_player.display_name(), mine_type, to_x, to_y);
                    eprintln!("{}", self.game_state.status_message());
                }
                
                return true;
            }
            
            // Check if the move is capturing an opponent's piece
            let captured_piece = if self.has_opponent_piece(to_pos, current_player) {
                let piece = self.get_piece_at(to_pos).unwrap();
                let capture_info = CaptureInfo::new(piece.piece_type, piece.player, to_pos);
                
                // Check if this is a king capture (win condition)
                if piece.piece_type == PieceType::King {
                    // Current player has won by capturing the king
                    self.game_state.set_winner(current_player);
                    self.game_over = true;
                    eprintln!("{} wins by capturing the opponent's king!", current_player.display_name());
                }
                
                Some(capture_info)
            } else {
                None
            };
            
            // Get the current cell state - this contains the piece type
            let current_cell_state = self.cells[from_idx];
            
            // Move the piece (preserving piece type)
            self.cells[to_idx] = current_cell_state;
            self.cells[from_idx] = CellState::Empty;
            
            // Update the piece type mapping
            self.piece_types.remove(&from_pos);
            
            // Get the piece type directly from the cell state
            let piece_type = match current_cell_state {
                CellState::PlayerKing => PieceType::King,
                CellState::PlayerQueen => PieceType::Queen,
                CellState::PlayerBishop => PieceType::Bishop,
                CellState::PlayerKnight => PieceType::Knight,
                CellState::PlayerRook => PieceType::Rook,
                CellState::PlayerPawn => PieceType::Pawn,
                CellState::OpponentKing => PieceType::King,
                CellState::OpponentQueen => PieceType::Queen,
                CellState::OpponentBishop => PieceType::Bishop,
                CellState::OpponentKnight => PieceType::Knight,
                CellState::OpponentRook => PieceType::Rook,
                CellState::OpponentPawn => PieceType::Pawn,
                _ => PieceType::Pawn, // Fallback
            };
            
            self.piece_types.insert(Position::new(to_x, to_y), piece_type);
            
            // Clear selection and highlights
            self.selected_piece = None;
            self.clear_highlights();
            
            // Store the capture info
            self.last_capture = captured_piece;
            
            // Log the move
            if let Some(capture) = &self.last_capture {
                eprintln!("Moved {:?} from ({}, {}) to ({}, {}) and captured {}", 
                          piece_type, from_x, from_y, to_x, to_y, capture.description());
            } else {
                eprintln!("Moved {:?} from ({}, {}) to ({}, {})", 
                          piece_type, from_x, from_y, to_x, to_y);
            }
            
            // Switch to next player's turn if game is still in progress
            if self.game_state.is_in_progress() {
                self.game_state.next_turn();
                eprintln!("{}", self.game_state.status_message());
            }
            
            return true;
        }
        
        false
    }
    
    // Clear all highlighted cells
    fn clear_highlights(&mut self) {
        self.highlights = vec![false; self.width * self.height];
    }
    
    // Highlight all valid moves
    fn highlight_valid_moves(&mut self) {
        let current_player = match self.game_state.current_player() {
            Some(player) => player,
            None => return, // Game not in progress
        };
        
        for y in 0..self.height {
            for x in 0..self.width {
                let pos = Position::new(x, y);
                if self.move_highlighter.is_valid_move(pos) {
                    let idx = y * self.width + x;
                    
                    // Highlight ALL valid move positions, regardless of cell type
                    // The piece movement logic will handle whether the move is legal
                    self.highlights[idx] = true;
                    
                    // Log the highlighted position for debugging
                    if self.has_opponent_piece(pos, current_player) {
                        eprintln!("Highlighted opponent piece at ({}, {})", x, y);
                    } else if self.is_empty(pos) {
                        eprintln!("Highlighted empty cell at ({}, {})", x, y);
                    } else {
                        eprintln!("Highlighted other cell at ({}, {})", x, y);
                    }
                }
            }
        }
    }
    
    // Reveal a cell - returns true if it was a mine (explosion, but doesn't end game)
    fn reveal(&mut self, x: usize, y: usize) -> bool {
        if self.game_over || x >= self.width || y >= self.height {
            return false;
        }
        
        // Get current player
        let current_player = match self.game_state.current_player() {
            Some(player) => player,
            None => return false, // Game not in progress
        };
        
        let idx = y * self.width + x;
        
        // Check if this cell is highlighted (valid move)
        if self.highlights[idx] {
            // Move to highlighted square if a piece is selected
            return self.move_piece(x, y);
        }
        
        match self.cells[idx] {
            mine_type if mine_type.is_mine() => {
                // Explosion! Destroy pieces in range instead of ending the game
                self.cells[idx] = CellState::RevealedMine; // Show the mine as revealed
                self.create_explosion(x, y, mine_type);
                
                // Switch to next player's turn after revealing a mine
                if self.game_state.is_in_progress() {
                    self.game_state.next_turn();
                    eprintln!("{}", self.game_state.status_message());
                }
                
                true // Return true to trigger a re-render
            },
            CellState::Empty => {
                // Reveal empty cell
                self.cells[idx] = CellState::Revealed;
                
                // Switch to next player's turn after revealing an empty cell
                if self.game_state.is_in_progress() {
                    self.game_state.next_turn();
                    eprintln!("{}", self.game_state.status_message());
                }
                
                true // Consider this a valid action that should end the turn
            },
            _ => {
                // For any other cell, check if it's a piece belonging to the current player
                let is_current_player_piece = match current_player {
                    Player::Human => matches!(self.cells[idx], 
                        CellState::PlayerKing | CellState::PlayerQueen | 
                        CellState::PlayerBishop | CellState::PlayerKnight | 
                        CellState::PlayerRook | CellState::PlayerPawn),
                    Player::Opponent => matches!(self.cells[idx], 
                        CellState::OpponentKing | CellState::OpponentQueen | 
                        CellState::OpponentBishop | CellState::OpponentKnight | 
                        CellState::OpponentRook | CellState::OpponentPawn),
                };
                
                if is_current_player_piece {
                    // Try to select the piece if it belongs to the current player
                    self.select_piece(x, y);
                    true
                } else {
                    false // Cannot reveal/interact with this cell
                }
            }
        }
    }
    
    // Create an explosion at the given coordinates, destroying pieces in range
    fn create_explosion(&mut self, x: usize, y: usize, mine_type: CellState) {
        // Get the blast radius from the mine type
        let blast_radius = mine_type.blast_radius();
        let armor_penetration = mine_type.armor_penetration();
        
        eprintln!("Creating explosion with blast radius {} and armor penetration {} from {:?}", 
                  blast_radius, armor_penetration, mine_type);
        
        // Find all cells within the blast radius
        for dy in -(blast_radius as isize)..=(blast_radius as isize) {
            for dx in -(blast_radius as isize)..=(blast_radius as isize) {
                // Calculate Manhattan distance (L1 norm) to consider diagonal distance more realistically
                let distance = dx.abs() + dy.abs();
                if distance > blast_radius as isize {
                    continue; // Skip cells outside the blast radius
                }
                
                // Calculate target coordinates
                let target_x = x as isize + dx;
                let target_y = y as isize + dy;
                
                // Skip if out of bounds
                if target_x < 0 || target_y < 0 || 
                   target_x >= self.width as isize || target_y >= self.height as isize {
                    continue;
                }
                
                let target_x = target_x as usize;
                let target_y = target_y as usize;
                let target_idx = target_y * self.width + target_x;
                
                // Check if there's a piece at this position
                match self.cells[target_idx] {
                    CellState::PlayerKing | CellState::PlayerQueen | 
                    CellState::PlayerBishop | CellState::PlayerKnight | 
                    CellState::PlayerRook | CellState::PlayerPawn => {
                        // Get the piece type for logging before removing it
                        let position = Position::new(target_x, target_y);
                        let piece_info = self.get_piece_at(position)
                            .map(|piece| format!("{:?} ({:?})", piece.piece_type, piece.player))
                            .unwrap_or_else(|| "Unknown piece".to_string());
                            
                        // Remove the piece from the board
                        self.cells[target_idx] = CellState::Revealed;
                        
                        // Remove the piece from the piece_types map
                        self.piece_types.remove(&Position::new(target_x, target_y));
                        
                        eprintln!("{} at ({}, {}) was destroyed by explosion!", piece_info, target_x, target_y);
                    },
                    CellState::OpponentKing | CellState::OpponentQueen | 
                    CellState::OpponentBishop | CellState::OpponentKnight | 
                    CellState::OpponentRook | CellState::OpponentPawn => {
                        // Get the piece type for logging before removing it
                        let position = Position::new(target_x, target_y);
                        let piece_info = self.get_piece_at(position)
                            .map(|piece| format!("{:?} ({:?})", piece.piece_type, piece.player))
                            .unwrap_or_else(|| "Unknown piece".to_string());
                            
                        // Remove the piece from the board
                        self.cells[target_idx] = CellState::Revealed;
                        
                        // Remove the piece from the piece_types map
                        self.piece_types.remove(&Position::new(target_x, target_y));
                        
                        eprintln!("{} at ({}, {}) was destroyed by explosion!", piece_info, target_x, target_y);
                    },
                    cell_type if cell_type.is_mine() => {
                        // Chain reaction - reveal this mine too! But don't create another explosion
                        self.cells[target_idx] = CellState::RevealedMine;
                        eprintln!("Chain reaction! {:?} mine at ({}, {}) was also revealed!", cell_type, target_x, target_y);
                    },
                    _ => {
                        // Reveal other cells but don't destroy them
                        if matches!(self.cells[target_idx], CellState::Empty | CellState::Flagged) {
                            self.cells[target_idx] = CellState::Revealed;
                        }
                    }
                }
            }
        }
        
        // If the selected piece was destroyed, clear the selection
        if let Some((sel_x, sel_y)) = self.selected_piece {
            let sel_idx = sel_y * self.width + sel_x;
            if !matches!(self.cells[sel_idx], CellState::PlayerKing | CellState::PlayerQueen | 
                          CellState::PlayerBishop | CellState::PlayerKnight | 
                          CellState::PlayerRook | CellState::PlayerPawn) {
                self.selected_piece = None;
                self.clear_highlights();
                eprintln!("Selected piece was destroyed!");
            }
        }
        
        eprintln!("BOOM! {:?} mine revealed at ({}, {}) with blast radius {}", 
                  mine_type, x, y, blast_radius);
    }
    
    // Toggle flag on a cell
    fn toggle_flag(&mut self, x: usize, y: usize) -> bool {
        if self.game_over || x >= self.width || y >= self.height {
            return false;
        }
        
        let idx = y * self.width + x;
        
        if self.cells[idx].is_mine() || matches!(self.cells[idx], CellState::Empty) {
                // Flag unrevealed cell
                self.cells[idx] = CellState::Flagged;
                true
        } else if matches!(self.cells[idx], CellState::Flagged) {
            // Unflag a flagged cell - restore the original mine type or empty state
            let value = self.cells[idx].to_value();
            self.cells[idx] = CellState::from_value(value); // This should restore the original mine type
            true
        } else if let Some(_) = self.selected_piece {
            // Cancel selection if right-clicking with a selection
            self.selected_piece = None;
            self.clear_highlights();
            true
        } else {
            false // Can't flag other types
        }
    }
    
    // Reveal all mines (game over) - modified to just reveal them, not end the game
    fn reveal_all_mines(&mut self) {
        for i in 0..self.cells.len() {
            if self.cells[i].is_mine() {
                self.cells[i] = CellState::RevealedMine;
            }
        }
    }
    
    // Reset the board for a new game
    fn reset(&mut self) {
        // Clear piece type mapping before recreating the board
        self.piece_types.clear();
        
        // Recreate the board (which will place pieces and mines correctly)
        let new_board = Board::new(self.width, self.height);
        *self = new_board;
    }
    
    // Check if player has won (all non-mine cells revealed)
    fn check_win(&self) -> bool {
        if self.game_over {
            return false;
        }
        
        // All non-mine cells should be revealed or be pieces
        for state in &self.cells {
            match state {
                CellState::Empty => {
                    // Found an unrevealed empty cell, so not won yet
                    return false;
                },
                _ => continue,
            }
        }
        
        true
    }
    
    // Render the game state using the provided renderer
    pub fn render(&mut self, renderer: &mut Renderer) {
        let mut grid_data = Vec::with_capacity(self.width * self.height);
        
        for y in 0..self.height {
            for x in 0..self.width {
                let idx = y * self.width + x;
                let cell = self.cells[idx];
                let is_highlighted = self.highlights[idx];
                
                // Use a distinctive value for highlighted cells
                if is_highlighted {
                    grid_data.push(4.0); // Use 4.0 for ALL highlighted cells
                } else {
                    // For non-highlighted cells, use the original cell values
                    grid_data.push(cell.to_value());
                }
            }
        }
        
        renderer.update_instances(&grid_data, self.width, self.height, Some(&self.piece_types));
    }
    
    // Update the get_game_state_message method to use the enum's status_message method
    fn get_game_state_message(&self) -> String {
        self.game_state.status_message()
    }
    
    // New method to get the last capture description if any
    fn get_last_capture_message(&self) -> Option<String> {
        self.last_capture.as_ref().map(|capture| {
            format!("Captured: {}", capture.description())
        })
    }
}

// Display game instructions in the console
fn display_instructions() {
    eprintln!("=== Chess-Sweeper - Game Instructions ===");
    eprintln!("- This is a fusion of Chess and Minesweeper");
    eprintln!("- White pieces start at the bottom, Black at the top");
    eprintln!("- Mines are randomly placed throughout the board");
    eprintln!("- Left click on your pieces to select them, then left click on a valid move location");
    eprintln!("- Right click to place/remove flags on suspected mines");
    eprintln!("- Right click on a selected piece to deselect it");
    eprintln!("- There are 4 types of mines with different blast radii:");
    eprintln!("  • Small Mine: 1 tile radius (3x3 area) - Low armor penetration");
    eprintln!("  • Medium Mine: 2 tile radius (5x5 area) - Medium armor penetration");
    eprintln!("  • Large Mine: 3 tile radius (7x7 area) - High armor penetration");
    eprintln!("  • Huge Mine: 4 tile radius (9x9 area) - Extreme armor penetration");
    eprintln!("- Armor penetration is prepared for future updates with piece toughness values");
    eprintln!("- BEWARE: Sliding pieces (bishops, rooks, queens) will detonate mines they move over!");
    eprintln!("  • Any piece that triggers a mine is destroyed in the explosion");
    eprintln!("- Knights are immune to mines they jump over (they only trigger mines they land on)");
    eprintln!("- Pawns and kings only trigger mines they land on");
    eprintln!("- Mine explosions have different blast radii based on mine type");
    eprintln!("- Mine explosions can trigger chain reactions with other mines");
    eprintln!("- Press 'R' to reset the game");
    eprintln!("- Press 'ESC' to deselect a piece or quit the game");
    eprintln!("- Press 'D' to show debug information");
    eprintln!("- Press 'H' to show these instructions again");
    eprintln!("==============================================");
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    
    // Display game instructions in the console
    display_instructions();
    
    let event_loop = EventLoop::new()?;
    
    // Create a larger window for the 24x24 grid
    let window = event_loop.create_window(
        winit::window::Window::default_attributes()
            .with_title("Chess-Sweeper - Press H for help")
            .with_inner_size(LogicalSize::new(1024, 768))
    )?;
    
    let window = Arc::new(window);
    
    // Create renderer and board with 24x24 grid
    let mut renderer = block_on(Renderer::new(&window));
    
    // Load the sprite sheet
    let sprite_path = "assets/sprites/chess_spritesheet.png";
    if let Ok(metadata) = std::fs::metadata(sprite_path) {
        eprintln!("Found sprite sheet: {} ({} bytes)", sprite_path, metadata.len());
        renderer.load_sprites(sprite_path);
    } else {
        eprintln!("Warning: Sprite sheet not found at {}. Will use colored blocks instead.", sprite_path);
        
        // Try to find individual sprite files
        let individual_path = "assets/sprites/chess_piece_2_white_king.png";
        if let Ok(metadata) = std::fs::metadata(individual_path) {
            eprintln!("Found individual piece image: {}", individual_path);
        } else {
            eprintln!("Warning: No piece images found. Using colored blocks for all pieces.");
        }
    }
    
    let mut board = Board::new(24, 24);
    let mut mouse_position = winit::dpi::PhysicalPosition::new(0.0, 0.0);
    
    // Set window title with initial turn info
    window.set_title(&format!("Chess-Sweeper - {}", board.get_game_state_message()));
    
    // Initial render
    board.render(&mut renderer);
    
    // Event loop with game mechanics
    event_loop.run(move |event, window_target| {
        use winit::event::{WindowEvent, ElementState, MouseButton};
        
        match event {
            winit::event::Event::WindowEvent { 
                window_id, 
                event
            } if window_id == window.id() => {
                match event {
                    WindowEvent::CloseRequested => {
                        window_target.exit();
                    },
                    WindowEvent::RedrawRequested => {
                        if let Err(e) = renderer.render() {
                            eprintln!("Render error: {:?}", e);
                            if matches!(e, wgpu::SurfaceError::Lost) {
                                renderer.resize(window.inner_size());
                            }
                        }
                    },
                    WindowEvent::Resized(size) => {
                        renderer.resize(size);
                    },
                    WindowEvent::CursorMoved { position, .. } => {
                        mouse_position = position;
                    },
                    WindowEvent::MouseInput { 
                        state: ElementState::Pressed, 
                        button, 
                        .. 
                    } => {
                        // Handle mouse clicks for gameplay
                        if board.game_over {
                            // Reset board on click after game over
                            board.reset();
                            window.set_title(&format!("Chess-Sweeper - {}", board.get_game_state_message()));
                            board.render(&mut renderer);
                            return;
                        }
                        
                        // Convert screen coordinates to grid coordinates
                        let screen_x = mouse_position.x as f32;
                        let screen_y = mouse_position.y as f32;
                        
                        if let Some((grid_x, grid_y)) = renderer.screen_to_grid(screen_x, screen_y) {
                            eprintln!("Clicked at grid position: ({}, {})", grid_x, grid_y);
                            
                            let changed = match button {
                                MouseButton::Left => {
                                    // Try to reveal or move
                                    board.reveal(grid_x, grid_y)
                                },
                                MouseButton::Right => {
                                    // Right click: toggle flag or deselect
                                    board.toggle_flag(grid_x, grid_y)
                                },
                                _ => false
                            };
                            
                            if changed {
                                // Update window title with current turn information
                                let title = match board.get_last_capture_message() {
                                    Some(capture_msg) => format!("Chess-Sweeper - {} - {}", 
                                                                board.get_game_state_message(), capture_msg),
                                    None => format!("Chess-Sweeper - {}", 
                                                   board.get_game_state_message()),
                                };
                                window.set_title(&title);
                                
                                // Update the renderer
                                board.render(&mut renderer);
                                
                                // Check for win condition
                                if board.check_win() {
                                    eprintln!("You win! All safe cells revealed.");
                                    board.game_over = true;
                                    window.set_title("Chess-sweeper - Game Over - All safe cells revealed!");
                                }
                            }
                        }
                    },
                    WindowEvent::KeyboardInput { 
                        event,
                        ..
                    } => {
                        use winit::keyboard::{Key, NamedKey};
                        
                        if event.state == ElementState::Pressed {
                            match event.logical_key {
                                Key::Named(NamedKey::Escape) => {
                                    // Clear selection on Escape
                                    if board.selected_piece.is_some() {
                                        board.selected_piece = None;
                                        board.clear_highlights();
                                        board.render(&mut renderer);
                                    } else {
                                    window_target.exit();
                                    }
                                },
                                Key::Character(ref c) if c == "r" || c == "R" => {
                                    // Reset game
                                    board.reset();
                                    board.render(&mut renderer);
                                    eprintln!("Game reset!");
                                },
                                Key::Character(ref c) if c == "d" || c == "D" => {
                                    // Debug - print renderer status
                                    renderer.debug_status();
                                },
                                Key::Character(ref c) if c == "h" || c == "H" => {
                                    // Display help instructions
                                    display_instructions();
                                },
                                _ => {}
                            }
                        }
                    },
                    _ => {},
                }
            },
            winit::event::Event::AboutToWait => {
                window.request_redraw();
            },
            _ => {},
        }
    })?;
    
    Ok(())
}
