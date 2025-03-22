// This file contains the chess rules and movement logic for the Chess-Sweeper game

use std::collections::HashSet;

/// Represents the different types of chess pieces
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PieceType {
    Pawn,
    Knight,
    Bishop,
    Rook,
    Queen,
    King,
}

/// Represents a player in the game
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Player {
    Human,    // Bottom player (white)
    Opponent, // Top player (Black)
}

impl Player {
    /// Get the opposing player
    pub fn opponent(&self) -> Self {
        match self {
            Player::Human => Player::Opponent,
            Player::Opponent => Player::Human,
        }
    }
    
    /// Get a display name for the player
    pub fn display_name(&self) -> &'static str {
        match self {
            Player::Human => "white",
            Player::Opponent => "Black",
        }
    }
}

/// Represents the current game state for turn management
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameState {
    InProgress(Player), // Game is in progress, with the current player's turn
    HumanWon,           // Human player has won
    OpponentWon,        // Opponent has won
    Draw,               // Game ended in a draw
}

impl GameState {
    /// Create a new game state with the specified starting player
    pub fn new(starting_player: Player) -> Self {
        GameState::InProgress(starting_player)
    }
    
    /// Get the current player, if the game is in progress
    pub fn current_player(&self) -> Option<Player> {
        match self {
            GameState::InProgress(player) => Some(*player),
            _ => None,
        }
    }
    
    /// Check if the game is still in progress
    pub fn is_in_progress(&self) -> bool {
        matches!(self, GameState::InProgress(_))
    }
    
    /// Switch to the next player's turn
    pub fn next_turn(&mut self) {
        if let GameState::InProgress(player) = self {
            *player = player.opponent();
        }
    }
    
    /// End the game with a winner
    pub fn set_winner(&mut self, winner: Player) {
        *self = match winner {
            Player::Human => GameState::HumanWon,
            Player::Opponent => GameState::OpponentWon,
        };
    }
    
    /// Set the game to a draw
    pub fn set_draw(&mut self) {
        *self = GameState::Draw;
    }

    /// Get a descriptive message about the current game state
    pub fn status_message(&self) -> String {
        match self {
            GameState::InProgress(player) => format!("{}'s Turn", player.display_name()),
            GameState::HumanWon => "white Wins!".to_string(),
            GameState::OpponentWon => "Black Wins!".to_string(),
            GameState::Draw => "Game Ended in Draw".to_string(),
        }
    }
}

/// Represents a chess piece with its type and owner
#[derive(Debug, Clone, Copy)]
pub struct Piece {
    pub piece_type: PieceType,
    pub player: Player,
}

/// Represents a position on the board
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Position {
    pub x: usize,
    pub y: usize,
}

impl Position {
    pub fn new(x: usize, y: usize) -> Self {
        Self { x, y }
    }
}

/// Chess rules engine
pub struct ChessRules {
    board_width: usize,
    board_height: usize,
}

impl ChessRules {
    /// Create a new chess rules engine
    pub fn new(board_width: usize, board_height: usize) -> Self {
        Self {
            board_width,
            board_height,
        }
    }

    /// Check if a position is within the board boundaries
    fn is_valid_position(&self, pos: Position) -> bool {
        pos.x < self.board_width && pos.y < self.board_height
    }

    /// Get all possible moves for a piece
    pub fn get_valid_moves(
        &self, 
        piece_type: PieceType, 
        player: Player, 
        position: Position, 
        board_state: &impl BoardStateProvider
    ) -> Vec<Position> {
        match piece_type {
            PieceType::Pawn => self.get_pawn_moves(player, position, board_state),
            PieceType::Knight => self.get_knight_moves(player, position, board_state),
            PieceType::Bishop => self.get_bishop_moves(player, position, board_state),
            PieceType::Rook => self.get_rook_moves(player, position, board_state),
            PieceType::Queen => self.get_queen_moves(player, position, board_state),
            PieceType::King => self.get_king_moves(player, position, board_state),
        }
    }

    /// Get pawn moves
    fn get_pawn_moves(
        &self, 
        player: Player, 
        position: Position, 
        board_state: &impl BoardStateProvider
    ) -> Vec<Position> {
        let mut moves = Vec::new();
        
        // Pawns move in different directions based on the player
        let (forward_direction, start_row) = match player {
            Player::Human => (-1i32, self.board_height - 2),    // Move up the board
            Player::Opponent => (1i32, 1),                      // Move down the board
        };
        
        // Forward move (1 square)
        let forward_pos = Position {
            x: position.x,
            y: (position.y as i32 + forward_direction) as usize,
        };
        
        if self.is_valid_position(forward_pos) && board_state.is_empty(forward_pos) {
            moves.push(forward_pos);
            
            // Initial two-square move
            if position.y == start_row {
                let double_forward_pos = Position {
                    x: position.x,
                    y: (position.y as i32 + 2 * forward_direction) as usize,
                };
                
                if self.is_valid_position(double_forward_pos) && board_state.is_empty(double_forward_pos) {
                    moves.push(double_forward_pos);
                }
            }
        }
        
        // Diagonal captures
        for dx in [-1, 1].iter() {
            let capture_pos = Position {
                x: (position.x as i32 + dx) as usize,
                y: (position.y as i32 + forward_direction) as usize,
            };
            
            if self.is_valid_position(capture_pos) && 
               board_state.has_opponent_piece(capture_pos, player) {
                moves.push(capture_pos);
            }
        }
        
        moves
    }

    /// Get knight moves
    fn get_knight_moves(
        &self, 
        player: Player, 
        position: Position, 
        board_state: &impl BoardStateProvider
    ) -> Vec<Position> {
        let mut moves = Vec::new();
        
        // Knight moves in an L-shape: 2 squares in one direction, then 1 square perpendicular
        let knight_offsets = [
            (-2, -1), (-2, 1), (-1, -2), (-1, 2),
            (1, -2), (1, 2), (2, -1), (2, 1),
        ];
        
        for (dx, dy) in knight_offsets.iter() {
            let new_x = position.x as i32 + dx;
            let new_y = position.y as i32 + dy;
            
            // Check bounds
            if new_x >= 0 && new_x < self.board_width as i32 && 
               new_y >= 0 && new_y < self.board_height as i32 {
                let target_pos = Position { x: new_x as usize, y: new_y as usize };
                
                // Knight can move to empty squares or capture opponent pieces
                if board_state.is_empty(target_pos) || 
                   board_state.has_opponent_piece(target_pos, player) {
                    moves.push(target_pos);
                }
            }
        }
        
        moves
    }

    /// Get bishop moves
    fn get_bishop_moves(
        &self, 
        player: Player, 
        position: Position, 
        board_state: &impl BoardStateProvider
    ) -> Vec<Position> {
        // Bishop moves diagonally
        self.get_sliding_moves(player, position, board_state, &[
            (-1, -1), (-1, 1), (1, -1), (1, 1)
        ])
    }

    /// Get rook moves
    fn get_rook_moves(
        &self, 
        player: Player, 
        position: Position, 
        board_state: &impl BoardStateProvider
    ) -> Vec<Position> {
        // Rook moves horizontally and vertically
        self.get_sliding_moves(player, position, board_state, &[
            (0, -1), (0, 1), (-1, 0), (1, 0)
        ])
    }

    /// Get queen moves
    fn get_queen_moves(
        &self, 
        player: Player, 
        position: Position, 
        board_state: &impl BoardStateProvider
    ) -> Vec<Position> {
        // Queen moves like a rook and bishop combined
        self.get_sliding_moves(player, position, board_state, &[
            (0, -1), (0, 1), (-1, 0), (1, 0),   // Horizontal and vertical
            (-1, -1), (-1, 1), (1, -1), (1, 1)  // Diagonal
        ])
    }

    /// Get king moves
    fn get_king_moves(
        &self, 
        player: Player, 
        position: Position, 
        board_state: &impl BoardStateProvider
    ) -> Vec<Position> {
        let mut moves = Vec::new();
        
        // King moves one square in any direction
        let directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ];
        
        for (dx, dy) in directions.iter() {
            let new_x = position.x as i32 + dx;
            let new_y = position.y as i32 + dy;
            
            // Check bounds
            if new_x >= 0 && new_x < self.board_width as i32 && 
               new_y >= 0 && new_y < self.board_height as i32 {
                let target_pos = Position { x: new_x as usize, y: new_y as usize };
                
                // King can move to empty squares or capture opponent pieces
                if board_state.is_empty(target_pos) || 
                   board_state.has_opponent_piece(target_pos, player) {
                    moves.push(target_pos);
                }
            }
        }
        
        moves
    }

    /// Helper method for generating moves for sliding pieces (bishop, rook, queen)
    fn get_sliding_moves(
        &self,
        player: Player,
        position: Position,
        board_state: &impl BoardStateProvider,
        directions: &[(i32, i32)],
    ) -> Vec<Position> {
        let mut moves = Vec::new();
        
        for (dx, dy) in directions {
            let mut current_x = position.x as i32;
            let mut current_y = position.y as i32;
            
            loop {
                current_x += dx;
                current_y += dy;
                
                // Check if we're still on the board
                if current_x < 0 || current_x >= self.board_width as i32 || 
                   current_y < 0 || current_y >= self.board_height as i32 {
                    break;
                }
                
                let target_pos = Position { x: current_x as usize, y: current_y as usize };
                
                if board_state.is_empty(target_pos) {
                    // Can move to this empty space
                    moves.push(target_pos);
                } else if board_state.has_opponent_piece(target_pos, player) {
                    // Can capture this opponent piece, but can't move further in this direction
                    moves.push(target_pos);
                    break;
                } else {
                    // Blocked by own piece
                    break;
                }
            }
        }
        
        moves
    }

    /// Detect the most likely piece type based on its position
    pub fn detect_piece_type(&self, player: Player, position: Position, board_width: usize) -> PieceType {
        // For the human player (bottom)
        if player == Player::Human {
            if position.y == board_width - 2 {
                // Second-to-last row contains pawns
                return PieceType::Pawn;
            } else if position.y == board_width - 1 {
                // Last row contains other pieces
                if position.x == 0 || position.x == board_width - 1 || position.x == 5 || position.x == board_width - 6 {
                    return PieceType::Rook;
                } else if position.x == 1 || position.x == board_width - 2 || position.x == 4 || position.x == board_width - 5 {
                    return PieceType::Knight;
                } else if position.x == 2 || position.x == board_width - 3 || position.x == 3 || position.x == board_width - 4 {
                    return PieceType::Bishop;
                } else if position.x == board_width / 2 - 1 {
                    return PieceType::Queen;
                } else if position.x == board_width / 2 {
                    return PieceType::King;
                }
            }
        } 
        // For the opponent (top)
        else {
            if position.y == 1 {
                // Second row contains pawns
                return PieceType::Pawn;
            } else if position.y == 0 {
                // First row contains other pieces
                if position.x == 0 || position.x == board_width - 1 || position.x == 5 || position.x == board_width - 6 {
                    return PieceType::Rook;
                } else if position.x == 1 || position.x == board_width - 2 || position.x == 4 || position.x == board_width - 5 {
                    return PieceType::Knight;
                } else if position.x == 2 || position.x == board_width - 3 || position.x == 3 || position.x == board_width - 4 {
                    return PieceType::Bishop;
                } else if position.x == board_width / 2 - 1 {
                    return PieceType::Queen;
                } else if position.x == board_width / 2 {
                    return PieceType::King;
                }
            }
        }
        
        // Default to Queen for pieces in other positions
        PieceType::Queen
    }
    
    /// Check if a move would result in capturing an opponent's king
    pub fn is_king_capture(&self, from: Position, to: Position, board_state: &impl BoardStateProvider) -> bool {
        // Check if there's a piece at the destination
        if let Some(captured_piece) = board_state.get_piece_at(to) {
            // Check if it's an opponent's king
            captured_piece.piece_type == PieceType::King
        } else {
            false
        }
    }
    
    /// Get the piece type at a position for the opponent
    pub fn get_opponent_piece_type(&self, position: Position, player: Player, board_state: &impl BoardStateProvider) -> Option<PieceType> {
        if let Some(piece) = board_state.get_piece_at(position) {
            if piece.player != player {
                return Some(piece.piece_type);
            }
        }
        None
    }
}

/// Trait for getting information about the board state
pub trait BoardStateProvider {
    /// Check if a position is empty
    fn is_empty(&self, position: Position) -> bool;
    
    /// Check if a position contains an opponent's piece
    fn has_opponent_piece(&self, position: Position, player: Player) -> bool;
    
    /// Get the piece at a position, if any
    fn get_piece_at(&self, position: Position) -> Option<Piece>;
}

/// Helper for highlighting available moves
pub struct MoveHighlighter {
    valid_moves: HashSet<Position>,
}

impl MoveHighlighter {
    pub fn new() -> Self {
        Self {
            valid_moves: HashSet::new(),
        }
    }
    
    pub fn set_valid_moves(&mut self, moves: Vec<Position>) {
        self.valid_moves.clear();
        for pos in moves {
            self.valid_moves.insert(pos);
        }
    }
    
    pub fn clear(&mut self) {
        self.valid_moves.clear();
    }
    
    pub fn is_valid_move(&self, position: Position) -> bool {
        self.valid_moves.contains(&position)
    }
}

/// Capture information for displaying to the player
#[derive(Debug, Clone)]
pub struct CaptureInfo {
    pub piece_type: PieceType,
    pub player: Player,
    pub position: Position,
}

impl CaptureInfo {
    pub fn new(piece_type: PieceType, player: Player, position: Position) -> Self {
        Self {
            piece_type,
            player,
            position,
        }
    }
    
    pub fn description(&self) -> String {
        format!("{}'s {} at ({}, {})", 
            self.player.display_name(),
            format!("{:?}", self.piece_type),
            self.position.x,
            self.position.y
        )
    }
} 