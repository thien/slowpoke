"""
    This module defines the CheckerBoard class.
    Built on top of Andrew Edwards's Checkers board. -- (almostimplemented.com)

    Andrew's code is licensed under the Apache License, 
    Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""
import numpy as np
import datetime
from termcolor import colored
from itertools import groupby

### CONSTANTS

# Black moves "forward", White moves "backward"
Black, White = 0, 1
empty = -1
blackKing = 2
whiteKing = 3
boringNoEatLimit = 50
# handles n/2 repeated move limits
repetitionLimits = 12
# The IBM704 had 36-bit words. Arthur Samuel used the extra bits to
# ensure that every normal move could be performed by flipping the
# original bit and the bit either 4 or 5 bits away, in the cases of
# moving right and left respectively.

unusedBits = 0b100000000100000000100000000100000000


def _set_bits(n):
    """Yield positions of set bits in n (LSB first), only looping over set bits.
    
    Much faster than bin(n)[::-1] enumerate for sparse bitboards.
    """
    while n:
        lsb = n & -n
        yield lsb.bit_length() - 1
        n ^= lsb


class CheckerBoard:
    """
    Initiates board via new_game().
    """
    __slots__ = (
        'forward', 'backward', 'pieces', 'active', 'passive',
        'empty', 'jump', 'mandatoryJumps', 'turnCount', 'multipleJumpStack',
        'state', 'winner', 'noEatCount', 'altMoveStack', 'moves', 'pdn',
        'blackPieces', 'whitePieces', '_history', 'AIBoardPos', 'is_over_called'
    )
    
    def __init__(self):
        self.forward = [None, None]
        self.backward = [None, None]
        self.pieces = [None, None]
        self.new_game()
        # initiate the state.
        self.updateState()

    """
    Initiates the PGN for the game (for export).
    """
    def init_pgn(self):
        currentDateTime = datetime.datetime.now()
        return {
            "Event" : "Some Event",
            "Date"  : currentDateTime.strftime("%y/%m/%d"),
            "Time"  : currentDateTime.strftime("%H:%M:%S"),
            "Result": "*",
            "FEN"   : "B:W21-32:B1-16",
            "Moves" : []
        }

    """
    Prints the PGN of the game.
    """
    def print_pgn(self):
        return self.pdn
    
    """
    Prints the PGN of the game.
    """
    def setID(self, game_id):
        self.pdn["_id"] = game_id

    def setColours(self, blackID, whiteID):
        self.pdn["Black"] = blackID
        self.pdn["White"] = whiteID
    
    """
    Resets current state to new game.
    """
    def new_game(self):
        
        self.active = Black
        self.passive = White

        self.forward[Black] = 0x1eff
        self.backward[Black] = 0
        self.pieces[Black] = self.forward[Black] | self.backward[Black]

        self.forward[White] = 0
        self.backward[White] = 0x7fbc00000
        self.pieces[White] = self.forward[White] | self.backward[White]

        self.empty = unusedBits ^ (2**36 - 1) ^ (self.pieces[Black] | self.pieces[White])

        self.jump = 0
        self.mandatoryJumps = []

        self.pdn = self.init_pgn()
        self.turnCount = 0
        self.multipleJumpStack = []
        self.state = []
        self.winner = None
        self.noEatCount = 0
        self.altMoveStack = []
        self.moves = []
        self._history = []  # For push/pop move operations

    """
    Updates the game state to reflect the effects of the input
    move.

    A legal move is represented by an integer with exactly two
    bits turned on: the old position and the new position.
    """
    def make_move(self, move, full_update=True):
        # Extract move string directly from move bits (no validation needed - caller ensures legality)
        move_abs = abs(move)
        # The move bit has exactly two 1-bits. Extract their positions.
        # For regular moves (0x11 or 0x21 shifted): bits are at i and i+4 (right) or i and i+5 (left)
        # For jumps (-0x101 or -0x401 shifted): bits are at i and i+8 (right) or i and i+10 (left)
        bits = [i for (i, b) in enumerate(bin(move_abs)[::-1]) if b == '1']
        src_bit = bits[0]
        dst_bit = bits[1]
        
        # Convert bit positions to square numbers (1-indexed, row-major)
        src = 1 + src_bit - src_bit//9
        dst = 1 + dst_bit - dst_bit//9
        moveString = f"{src}x{dst}" if move < 0 else f"{src}-{dst}"
        
        # perform move action below
        active = self.active
        passive = self.passive
        if move < 0:
            move *= -1
            takenPiece = int(1 << sum(i for (i, b) in enumerate(bin(move)[::-1]) if b == '1')//2)
            self.pieces[passive] ^= takenPiece
            if self.forward[passive] & takenPiece:
                self.forward[passive] ^= takenPiece
            if self.backward[passive] & takenPiece:
                self.backward[passive] ^= takenPiece
            self.jump = 1

        self.pieces[active] ^= move
        if self.forward[active] & move:
            self.forward[active] ^= move
        if self.backward[active] & move:
            self.backward[active] ^= move

        destination = move & self.pieces[active]
        self.empty = unusedBits ^ (2**36 - 1) ^ (self.pieces[Black] | self.pieces[White])
        
        # add to move stacks (for the bot to decide on historical moves)
        self.altMoveStack.append((active, moveString))
        self.moves.append((active, moveString))


        # if theres a jump, see if the user needs to make more jumps.
        if self.jump:
            self.noEatCount = 0
            self.mandatoryJumps = self.jumps_from(destination)
            # now put the previous position on cache.
            positions = moveString.split("x")
            if len(positions) >= 2:
                self.multipleJumpStack.append(positions[0])
                self.multipleJumpStack.append(positions[1])
            if self.mandatoryJumps:
                return self
        else:
            self.noEatCount += 1

        if active == Black and (destination & 0x780000000) != 0:
            self.backward[Black] |= destination
        elif active == White and (destination & 0xf) != 0:
            self.forward[White] |= destination

        # need to add the move to the list of moves.
        if len(self.multipleJumpStack) > 0:
            if len(self.multipleJumpStack) > 2:
                # concatenate the move attack into one string, removing duplicates
                jumpStacks = [x[0] for x in groupby(self.multipleJumpStack)]
                self.pdn["Moves"].append("x".join(jumpStacks))
            else:
                self.pdn["Moves"].append("x".join(self.multipleJumpStack))
            # empty the stack when done
            self.multipleJumpStack = []
        else:
            self.pdn["Moves"].append(moveString)
        
        # reset the number of jumps, switch players and continue.
        self.jump = 0
        self.active, self.passive = self.passive, self.active
        if full_update:
            self.updateState()
        else:
            self._update_rank()
        return self

    # """
    # Updates the game state to reflect the effects of the input
    # move.

    # A legal move is represented by an integer with exactly two
    # bits turned on: the old position and the new position.
    # """
    # def peek_move(self, move):
    #     B = self.copy()
    #     active = B.active
    #     passive = B.passive
    #     if move < 0:
    #         move *= -1
    #         takenPiece = int(1 << sum(i for (i, b) in enumerate(bin(move)[::-1]) if b == '1')/2)
    #         B.pieces[passive] ^= takenPiece
    #         if B.forward[passive] & takenPiece:
    #             B.forward[passive] ^= takenPiece
    #         if B.backward[passive] & takenPiece:
    #             B.backward[passive] ^= takenPiece
    #         B.jump = 1

    #     B.pieces[active] ^= move
    #     if B.forward[active] & move:
    #         B.forward[active] ^= move
    #     if B.backward[active] & move:
    #         B.backward[active] ^= move

    #     destination = move & B.pieces[active]
    #     B.empty = unusedBits ^ (2**36 - 1) ^ (B.pieces[Black] | B.pieces[White])

    #     if B.jump:
    #         B.mandatoryJumps = B.jumps_from(destination)
    #         if B.mandatoryJumps:
    #             return B

    #     if active == Black and (destination & 0x780000000) != 0:
    #         B.backward[Black] |= destination
    #     elif active == White and (destination & 0xf) != 0:
    #         B.forward[White] |= destination

    #     B.jump = 0
    #     B.active, B.passive = B.passive, B.active

    #     return B

    """
    These methods return an integer whose active bits are those squares
    that can make the move indicated by the method name.
    """
    def right_forward(self):
        return (self.empty >> 4) & self.forward[self.active]
    def left_forward(self):
        return (self.empty >> 5) & self.forward[self.active]
    def right_backward(self):
        return (self.empty << 4) & self.backward[self.active]
    def left_backward(self):
        return (self.empty << 5) & self.backward[self.active]
    def right_forward_jumps(self):
        return (self.empty >> 8) & (self.pieces[self.passive] >> 4) & self.forward[self.active]
    def left_forward_jumps(self):
        return (self.empty >> 10) & (self.pieces[self.passive] >> 5) & self.forward[self.active]
    def right_backward_jumps(self):
        return (self.empty << 8) & (self.pieces[self.passive] << 4) & self.backward[self.active]
    def left_backward_jumps(self):
        return (self.empty << 10) & (self.pieces[self.passive] << 5) & self.backward[self.active]

    def get_moves(self):
        if self.jump:
            return self.mandatoryJumps

        jumps = self.get_jumps()
        if jumps:
            return jumps

        rf = self.right_forward()
        lf = self.left_forward()
        rb = self.right_backward()
        lb = self.left_backward()

        moves = []
        for i in _set_bits(rf):
            moves.append(0x11 << i)
        for i in _set_bits(lf):
            moves.append(0x21 << i)
        for i in _set_bits(rb):
            moves.append(0x11 << (i - 4))
        for i in _set_bits(lb):
            moves.append(0x21 << (i - 5))
        return moves

    def get_jumps(self):
        rfj = self.right_forward_jumps()
        lfj = self.left_forward_jumps()
        rbj = self.right_backward_jumps()
        lbj = self.left_backward_jumps()

        moves = []

        if (rfj | lfj | rbj | lbj) != 0:
            for i in _set_bits(rfj):
                moves.append(-(0x101 << i))
            for i in _set_bits(lfj):
                moves.append(-(0x401 << i))
            for i in _set_bits(rbj):
                moves.append(-(0x101 << (i - 8)))
            for i in _set_bits(lbj):
                moves.append(-(0x401 << (i - 10)))

        return moves

    def jumps_from(self, piece):
        if self.active == Black:
            rfj = (self.empty >> 8) & (self.pieces[self.passive] >> 4) & piece
            lfj = (self.empty >> 10) & (self.pieces[self.passive] >> 5) & piece
            if piece & self.backward[self.active]:
                rbj = (self.empty << 8) & (self.pieces[self.passive] << 4) & piece
                lbj = (self.empty << 10) & (self.pieces[self.passive] << 5) & piece
            else:
                rbj = 0
                lbj = 0
        else:
            rbj = (self.empty << 8) & (self.pieces[self.passive] << 4) & piece
            lbj = (self.empty << 10) & (self.pieces[self.passive] << 5) & piece
            if piece & self.forward[self.active]:
                rfj = (self.empty >> 8) & (self.pieces[self.passive] >> 4) & piece
                lfj = (self.empty >> 10) & (self.pieces[self.passive] >> 5) & piece
            else:
                rfj = 0
                lfj = 0

        moves = []
        if (rfj | lfj | rbj | lbj) != 0:
            for i in _set_bits(rfj):
                moves.append(-(0x101 << i))
            for i in _set_bits(lfj):
                moves.append(-(0x401 << i))
            for i in _set_bits(rbj):
                moves.append(-(0x101 << (i - 8)))
            for i in _set_bits(lbj):
                moves.append(-(0x401 << (i - 10)))
        return moves

    # Returns true of the passed piece can be taken by the active player.
    def takeable(self, piece):
        active = self.active
        if (self.forward[active] & (piece >> 4)) != 0 and (self.empty & (piece << 4)) != 0:
            return True
        if (self.forward[active] & (piece >> 5)) != 0 and (self.empty & (piece << 5)) != 0:
            return True
        if (self.backward[active] & (piece << 4)) != 0 and (self.empty & (piece >> 4)) != 0:
            return True
        if (self.backward[active] & (piece << 5)) != 0 and (self.empty & (piece >> 5)) != 0:
            return True
        return False

    # returns true if there are no more possible moves to make
    def is_over(self):
        itHas = False
        # If there's enough moves where nobody is being jumped on, 
        # then call it a draw.
        if self.noEatCount == boringNoEatLimit:
            itHas = True
        else:
            # maybe someone has actually lost.
            if len(self.get_moves()) == 0:
                itHas = True
            else:
                # if they're still playing, check if there is 
                # threefold repetition.
                lastmoves = self.altMoveStack[-repetitionLimits:]
                if len(lastmoves) == repetitionLimits:
                    p1_list = lastmoves[0::2]
                    p2_list = lastmoves[1::2]
                    # count unique moves
                    p1_recent = set(p1_list)
                    p2_recent = set(p2_list)
                    if len(p1_recent) <4 and len(p2_recent) <4:
                        itHas = True      
        if itHas:
            self.checkWinner()
            return itHas
        else:
            return itHas

    def checkWinner(self):
        if self.noEatCount == boringNoEatLimit:
            self.winner = empty
            self.pdn["Winner"] = empty
            self.pdn["Result"] = "1/2-1/2"
        else:
            has_black = self.pieces[Black] != 0
            has_white = self.pieces[White] != 0
            if has_black and has_white:
                self.winner = empty
                self.pdn["Winner"] = empty
                self.pdn["Result"] = "1/2-1/2"
            else:
                if self.active == White:
                    self.winner = Black
                    self.pdn["Winner"] = Black
                    self.pdn["Result"] = "1-0"
                else:
                    self.winner = White
                    self.pdn["Winner"] = White
                    self.pdn["Result"] = "0-1"
        self.pdn['replay'] = self.moves

    # Prints winner message (when needed)
    def getWinnerMessage(self):
        if self.winner == Black:
            print ("Congrats Black, you win!")
        elif self.winner == empty:
            print ("It's a draw!")
        else:
            print ("Congrats White, you win!")

    # Returns the current player
    def current_player(self,board=None):
        if board:
            return board.current_player()
        else:
            if self.turnCount % 2 == 0:
                return Black
            else:
                return White

    # Returns a new board with the exact same state as the calling object.
    def copy(self):
        B = CheckerBoard()
        B.active = self.active
        B.backward = [x for x in self.backward]
        B.empty = self.empty
        B.forward = [x for x in self.forward]
        B.jump = self.jump
        B.mandatoryJumps = [x for x in self.mandatoryJumps]
        B.passive = self.passive
        B.pieces = [x for x in self.pieces]
        B.noEatCount = self.noEatCount
        # B.pdn = self.pdn
        B.altMoveStack = self.altMoveStack[-repetitionLimits+2:]
        return B

    def push_move(self, move):
        history_entry = (
            self.active,
            self.passive,
            self.forward[0], self.forward[1],
            self.backward[0], self.backward[1],
            self.pieces[0], self.pieces[1],
            self.empty,
            self.jump,
            tuple(self.mandatoryJumps),
            self.noEatCount,
            tuple(self.multipleJumpStack),
            self.turnCount,
            len(self.moves),
            len(self.altMoveStack),
            list(self.AIBoardPos),
        )
        self._history.append(history_entry)
        return self.make_move(move, full_update=False)

    def pop_move(self):
        if not self._history:
            return self
        entry = self._history.pop()
        self.active = entry[0]
        self.passive = entry[1]
        self.forward[0] = entry[2]
        self.forward[1] = entry[3]
        self.backward[0] = entry[4]
        self.backward[1] = entry[5]
        self.pieces[0] = entry[6]
        self.pieces[1] = entry[7]
        self.empty = entry[8]
        self.jump = entry[9]
        self.mandatoryJumps = list(entry[10])
        self.noEatCount = entry[11]
        self.multipleJumpStack = list(entry[12])
        self.turnCount = entry[13]
        del self.moves[entry[14]:]
        del self.altMoveStack[entry[15]:]
        self.AIBoardPos = list(entry[16])
        # No updateState() needed — AIBoardPos restored directly

    """
    Returns a list of possible moves that the player can choose to make.
    """
    def get_move_strings(self):
        # First check if we are in a jump sequence
        if self.jump and self.mandatoryJumps:
            # Convert mandatoryJumps to strings for display purposes
            jump_strings = []
            for move in self.mandatoryJumps:
                move_abs = abs(move)
                # Extract source and destination from the move bit
                bits = [(i, b) for (i, b) in enumerate(bin(move_abs)[::-1]) if b == '1']
                if len(bits) >= 2:
                    src_bit, dst_bit = bits[0][0], bits[1][0] if len(bits) > 1 else bits[0][0] + 4 if move_abs & 0x11 else bits[0][0] + 5
                    src = 1 + src_bit - src_bit//9
                    dst = 1 + dst_bit - dst_bit//9
                    jump_strings.append(f"{src}x{dst}")
            return jump_strings
        
        moves = []
        rfj = self.right_forward_jumps()
        lfj = self.left_forward_jumps()
        rbj = self.right_backward_jumps()
        lbj = self.left_backward_jumps()

        if (rfj | lfj | rbj | lbj) != 0:
            rfj = [(1 + i - i//9, 1 + (i + 8) - (i + 8)//9)
                        for (i, bit) in enumerate(bin(rfj)[::-1]) if bit == '1']
            lfj = [(1 + i - i//9, 1 + (i + 10) - (i + 8)//9)
                        for (i, bit) in enumerate(bin(lfj)[::-1]) if bit == '1']
            rbj = [(1 + i - i//9, 1 + (i - 8) - (i - 8)//9)
                        for (i, bit) in enumerate(bin(rbj)[::-1]) if bit ==  '1']
            lbj = [(1 + i - i//9, 1 + (i - 10) - (i - 10)//9)
                        for (i, bit) in enumerate(bin(lbj)[::-1]) if bit == '1']

            if self.active == Black:
                regular_moves = ["%ix%i" % (orig, dest) for (orig, dest) in rfj + lfj]
                reverse_moves = ["%ix%i" % (orig, dest) for (orig, dest) in rbj + lbj]
                moves = regular_moves + reverse_moves
            else:
                reverse_moves = ["%ix%i" % (orig, dest) for (orig, dest) in rfj + lfj]
                regular_moves = ["%ix%i" % (orig, dest) for (orig, dest) in rbj + lbj]
                moves = reverse_moves + regular_moves

        else:
            rf = self.right_forward()
            lf = self.left_forward()
            rb = self.right_backward()
            lb = self.left_backward()

            rf = [(1 + i - i//9, 1 + (i + 4) - (i + 4)//9)
                        for (i, bit) in enumerate(bin(rf)[::-1]) if bit == '1']
            lf = [(1 + i - i//9, 1 + (i + 5) - (i + 5)//9)
                        for (i, bit) in enumerate(bin(lf)[::-1]) if bit == '1']
            rb = [(1 + i - i//9, 1 + (i - 4) - (i - 4)//9)
                        for (i, bit) in enumerate(bin(rb)[::-1]) if bit ==  '1']
            lb = [(1 + i - i//9, 1 + (i - 5) - (i - 5)//9)
                        for (i, bit) in enumerate(bin(lb)[::-1]) if bit == '1']

            if self.active == Black:
                regular_moves = ["%i-%i" % (orig, dest) for (orig, dest) in rf + lf]
                reverse_moves = ["%i-%i" % (orig, dest) for (orig, dest) in rb + lb]
                moves = regular_moves + reverse_moves
            else:
                regular_moves = ["%i-%i" % (orig, dest) for (orig, dest) in rb + lb]
                reverse_moves = ["%i-%i" % (orig, dest) for (orig, dest) in rf + lf]
                moves = reverse_moves + regular_moves
        return moves


    """
    Fast state update for tree search. Only maintains AIBoardPos and turnCount.
    Skips PDN strings, display state, and piece-name formatting.
    """
    def _update_rank(self):
        blackKings = self.backward[Black]
        blackMen = self.forward[Black] ^ blackKings
        whiteKings = self.forward[White]
        whiteMen = self.backward[White] ^ whiteKings

        rank = [empty] * 32
        for i in range(4):
            base = 9 * i
            for j in range(8):
                cell = 1 << (base + j)
                idx = 8 * i + j
                if cell & blackMen:
                    rank[idx] = Black
                elif cell & whiteMen:
                    rank[idx] = White
                elif cell & blackKings:
                    rank[idx] = blackKing
                elif cell & whiteKings:
                    rank[idx] = whiteKing

        self.AIBoardPos = rank
        self.turnCount += 1

    """
    Returns a record of the positions of the pieces on the board.
    This also updates the FEN.
    """
    def updateState(self):

        # generate the PDN for the current board.
        def genPDN(blackPieces, whitePieces):
            Black_list = ','.join(blackPieces)
            White_list = ','.join(whitePieces)
            self.blackPieces = blackPieces
            self.whitePieces = whitePieces
            current = "B"
            if self.active == White:
                current = "W"
            li = current + ":W" + White_list + ":" + "B" + Black_list
            self.pdn["FEN"] = li

        # shorthand code for calculating the cell position.
        def cellPos(i,j):
            return 1 + j + 8*i

        blackPieces = []
        whitePieces = []

        if self.active == Black:
            blackKings = self.backward[self.active]
            blackMen = self.forward[self.active] ^ blackKings
            whiteKings = self.forward[self.passive]
            whiteMen = self.backward[self.passive] ^ whiteKings
        else:
            blackKings = self.backward[self.passive]
            blackMen = self.forward[self.passive] ^ blackKings
            whiteKings = self.forward[self.active]
            whiteMen = self.backward[self.active] ^ whiteKings

        state = [[None for _ in range(8)] for _ in range(4)]

        rank = []
        for i in range(4):
        
            for j in range(8):
                cell = 1 << (9*i + j)
                if cell & blackMen:
                    state[i][j] = Black
                    blackPieces.append(str(cellPos(i,j)))
                    rank.append(Black)
                elif cell & whiteMen:
                    state[i][j] = White
                    whitePieces.append(str(cellPos(i,j)))
                    rank.append(White)
                elif cell & blackKings:
                    state[i][j] = blackKing
                    blackPieces.append(("K" + str(cellPos(i,j))))
                    rank.append(blackKing)
                elif cell & whiteKings:
                    state[i][j] = whiteKing
                    whitePieces.append(("K" + str(cellPos(i,j))))
                    rank.append(whiteKing)
                else:
                    state[i][j] = empty
                    rank.append(empty)
        # print("Rank", rank)
        self.AIBoardPos = rank
        self.state = state
        self.turnCount += 1
        genPDN(blackPieces,whitePieces)

    """
    Returns the positions of the pieces for the AI.
    """
    def getBoardPos(self, colour):
        if colour == Black:
            return self.AIBoardPos
        else:
            # perform ugly position swap
            self.AIBoardPos.reverse()
            results = self.AIBoardPos
            self.AIBoardPos.reverse()
            return results

    """
    Same as above, but option to convert weights.
    """
    def getBoardPosWeighted(self, colour, weights):
        w = weights
        if colour == Black:
            rep = {Black: w['Black'], White: w['White'], empty: w['empty'],
                   blackKing: w['blackKing'], whiteKing: w['whiteKing']}
            return np.array([rep[n] for n in self.AIBoardPos], dtype=np.float32)
        else:
            rep = {Black: w['White'], White: w['Black'], empty: w['empty'],
                   blackKing: w['whiteKing'], whiteKing: w['blackKing']}
            return np.array([rep[n] for n in reversed(self.AIBoardPos)], dtype=np.float32)

    
    def generateASCIIBoard(self, blackPOV=True):
        """
        Param:
            blackPOV: True
            prints from the perspective of black if it's set to true.
            otherwise it will print to white's perspective; i.e it flips the board.
        """

        # generate ASCII board frame
        board = [None] * 17
        for i in range(9):
            board[2*i] = ["+", " - "] + ["-", " - "]*7 + ["+", "\n"]
            if i < 8:
              board[2*i + 1] = ["|", "   "] \
                             + [a for subl in [["|", "   "] for _ in range(7)] for a in subl] \
                             + ["|", "\n"]

        
        def paddingCheck(i,j):
            return ' ' if j + 8*i < 9 else ''

        # render the ASCII board content
        for i, chunk in enumerate(self.state):
            for j, cell in enumerate(chunk):
                # clean code writing for list indexes for the board.
                x = -1
                y = -1
                # calculate the positions for the indexes.
                if j < 4:
                    x = 2*(7 - 2*i) + 1
                    y = 2*(6 - 2*j) + 1
                else:
                    x = 2*(6 - 2*i) + 1
                    y = 2*(7 - 2*j) - 1
                
                piece = " "
          
                if cell == Black:
                    piece = colored("b", 'red', attrs=['reverse'])
                elif cell == White:
                    piece = colored("w", 'cyan', attrs=['reverse'])
                elif cell == blackKing:
                    piece = colored("B", 'red', attrs=['reverse'])
                elif cell == whiteKing:
                    piece = colored("W", 'cyan', attrs=['reverse'])

                # initiate the board with values.
                board[x][y] = piece + str(1 + j + 8*i) + (paddingCheck(i,j))

        # return "".join(map(lambda x: "".join(x), board))
        if blackPOV != True:
            # its white, reverse.
            board = board[::-1]
        return board
    
    def printBoard(self, blackPOV=True):
        return "".join(map(lambda x: "".join(x), self.generateASCIIBoard(blackPOV)))
    
    # Prints out the checkerboard. Mapped to default __str__ value; use if needed.
    def __str__(self):
        # print('\033c', end=None)
        return  "".join(map(lambda x: "".join(x), self.generateASCIIBoard()))
        # return self.generateASCIIBoard()