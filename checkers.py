"""
    This module defines the CheckerBoard class.
    Built on top of Andrew Edwards's Checkers board. -- (almostimplemented.com)
"""
import datetime
from termcolor import colored

### CONSTANTS

# Black moves "forward", White moves "backward"
Black, White = 0, 1

# The IBM704 had 36-bit words. Arthur Samuel used the extra bits to
# ensure that every normal move could be performed by flipping the
# original bit and the bit either 4 or 5 bits away, in the cases of
# moving right and left respectively.

unusedBits = 0b100000000100000000100000000100000000


class CheckerBoard:
    """
    Initiates board via new_game().
    """
    def __init__(self):
        self.forward = [None, None]
        self.backward = [None, None]
        self.pieces = [None, None]
        self.moves = []
        self.new_game()

    def init_pgn(self):
        currentDateTime = datetime.datetime.now()
        return {
            "Event" : "Some Event",
            "Date"  : currentDateTime.strftime("%y/%m/%d"),
            "Time"  : currentDateTime.strftime("%H:%M:%S"),
            "FEN"   : "B:W21-32:B1-16"
        }
    """
    Resets current state to new game.
    """
    def new_game(self):
        self.pgn = self.init_pgn()

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

    """
    Updates the game state to reflect the effects of the input
    move.

    A legal move is represented by an integer with exactly two
    bits turned on: the old position and the new position.
    """
    def make_move(self, move):
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

        if self.jump:
            self.mandatoryJumps = self.jumps_from(destination)
            if self.mandatoryJumps:
                return

        if active == Black and (destination & 0x780000000) != 0:
            self.backward[Black] |= destination
        elif active == White and (destination & 0xf) != 0:
            self.forward[White] |= destination

        self.jump = 0
        self.active, self.passive = self.passive, self.active

    """
    Updates the game state to reflect the effects of the input
    move.

    A legal move is represented by an integer with exactly two
    bits turned on: the old position and the new position.
    """
    def peek_move(self, move):
        B = self.copy()
        active = B.active
        passive = B.passive
        if move < 0:
            move *= -1
            takenPiece = int(1 << sum(i for (i, b) in enumerate(bin(move)[::-1]) if b == '1')/2)
            B.pieces[passive] ^= takenPiece
            if B.forward[passive] & takenPiece:
                B.forward[passive] ^= takenPiece
            if B.backward[passive] & takenPiece:
                B.backward[passive] ^= takenPiece
            B.jump = 1

        B.pieces[active] ^= move
        if B.forward[active] & move:
            B.forward[active] ^= move
        if B.backward[active] & move:
            B.backward[active] ^= move

        destination = move & B.pieces[active]
        B.empty = unusedBits ^ (2**36 - 1) ^ (B.pieces[Black] | B.pieces[White])

        if B.jump:
            B.mandatoryJumps = B.jumps_from(destination)
            if B.mandatoryJumps:
                return B

        if active == Black and (destination & 0x780000000) != 0:
            B.backward[Black] |= destination
        elif active == White and (destination & 0xf) != 0:
            B.forward[White] |= destination

        B.jump = 0
        B.active, B.passive = B.passive, B.active

        return B

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


    """
    Returns a list of all possible moves.

    A legal move is represented by an integer with exactly two
    bits turned on: the old position and the new position.

    Jumps are indicated with a negative sign.
    """
    def get_moves(self):
        # First check if we are in a jump sequence
        if self.jump:
            return self.mandatoryJumps

        # Next check if there are jumps
        jumps = self.get_jumps()
        if jumps:
            return jumps

        # If not, then find normal moves
        else:
            rf = self.right_forward()
            lf = self.left_forward()
            rb = self.right_backward()
            lb = self.left_backward()

            moves =  [0x11 << i for (i, bit) in enumerate(bin(rf)[::-1]) if bit == '1']
            moves += [0x21 << i for (i, bit) in enumerate(bin(lf)[::-1]) if bit == '1']
            moves += [0x11 << i - 4 for (i, bit) in enumerate(bin(rb)[::-1]) if bit == '1']
            moves += [0x21 << i - 5 for (i, bit) in enumerate(bin(lb)[::-1]) if bit == '1']
            return moves

    """
    Returns a list of all possible jumps.

    A legal move is represented by an integer with exactly two
    bits turned on: the old position and the new position.

    Jumps are indicated with a negative sign.
    """
    def get_jumps(self):
        rfj = self.right_forward_jumps()
        lfj = self.left_forward_jumps()
        rbj = self.right_backward_jumps()
        lbj = self.left_backward_jumps()

        moves = []

        if (rfj | lfj | rbj | lbj) != 0:
            moves += [-0x101 << i for (i, bit) in enumerate(bin(rfj)[::-1]) if bit == '1']
            moves += [-0x401 << i for (i, bit) in enumerate(bin(lfj)[::-1]) if bit == '1']
            moves += [-0x101 << i - 8 for (i, bit) in enumerate(bin(rbj)[::-1]) if bit == '1']
            moves += [-0x401 << i - 10 for (i, bit) in enumerate(bin(lbj)[::-1]) if bit == '1']

        return moves

    """
    Returns list of all possible jumps from the piece indicated.

    The argument piece should be of the form 2**n, where n + 1 is
    the square of the piece in question (using the internal numeric
    representaiton of the board).
    """
    def jumps_from(self, piece):
        if self.active == Black:
            rfj = (self.empty >> 8) & (self.pieces[self.passive] >> 4) & piece
            lfj = (self.empty >> 10) & (self.pieces[self.passive] >> 5) & piece
            if piece & self.backward[self.active]: # piece at square is a king
                rbj = (self.empty << 8) & (self.pieces[self.passive] << 4) & piece
                lbj = (self.empty << 10) & (self.pieces[self.passive] << 5) & piece
            else:
                rbj = 0
                lbj = 0
        else:
            rbj = (self.empty << 8) & (self.pieces[self.passive] << 4) & piece
            lbj = (self.empty << 10) & (self.pieces[self.passive] << 5) & piece
            if piece & self.forward[self.active]: # piece at square is a king
                rfj = (self.empty >> 8) & (self.pieces[self.passive] >> 4) & piece
                lfj = (self.empty >> 10) & (self.pieces[self.passive] >> 5) & piece
            else:
                rfj = 0
                lfj = 0

        moves = []
        if (rfj | lfj | rbj | lbj) != 0:
            moves += [-0x101 << i for (i, bit) in enumerate(bin(rfj)[::-1]) if bit == '1']
            moves += [-0x401 << i for (i, bit) in enumerate(bin(lfj)[::-1]) if bit == '1']
            moves += [-0x101 << i - 8 for (i, bit) in enumerate(bin(rbj)[::-1]) if bit == '1']
            moves += [-0x401 << i - 10 for (i, bit) in enumerate(bin(lbj)[::-1]) if bit == '1']
        return moves

    """
    Returns true of the passed piece can be taken by the active player.
    """
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

    """
    Returns true if there are no more possible moves to make.
    """
    def is_over(self):
        # probably the smallest function here.
        return len(self.get_moves()) == 0

    """
    Returns a new board with the exact same state as the calling object.
    """
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
        return B

    """
    Returns a list of possible moves that the player can choose to make.
    """
    def get_move_strings(self):
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
                regular_moves = ["%i to %i" % (orig, dest) for (orig, dest) in rfj + lfj]
                reverse_moves = ["%i to %i" % (orig, dest) for (orig, dest) in rbj + lbj]
                return regular_moves + reverse_moves
            else:
                reverse_moves = ["%i to %i" % (orig, dest) for (orig, dest) in rfj + lfj]
                regular_moves = ["%i to %i" % (orig, dest) for (orig, dest) in rbj + lbj]
                return reverse_moves + regular_moves

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
            regular_moves = ["%i to %i" % (orig, dest) for (orig, dest) in rf + lf]
            reverse_moves = ["%i to %i" % (orig, dest) for (orig, dest) in rb + lb]
            return regular_moves + reverse_moves
        else:
            regular_moves = ["%i to %i" % (orig, dest) for (orig, dest) in rb + lb]
            reverse_moves = ["%i to %i" % (orig, dest) for (orig, dest) in rf + lf]
            return reverse_moves + regular_moves

    """
    Checks for a winner.
    """
    def checkWinner(self):
        if self.active == White:
            print ("Congrats Black, you win!")
        else:
            print ("Congrats White, you win!")

    """
    Prints out ASCII art representation of board.
    """
    def __str__(self):
        empty = -1
        blackKing = 2
        whiteKing = 3

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
        for i in range(4):
            for j in range(8):
                cell = 1 << (9*i + j)
                if cell & blackMen:
                    state[i][j] = Black
                elif cell & whiteMen:
                    state[i][j] = White
                elif cell & blackKings:
                    state[i][j] = blackKing
                elif cell & whiteKings:
                    state[i][j] = whiteKing
                else:
                    state[i][j] = empty

        board = [None] * 17
        for i in range(9):
            board[2*i] = ["+", " - "] + ["+", " - "]*7 + ["+", "\n"]
            if i < 8:
              board[2*i + 1] = ["|", "   "] \
                             + [a for subl in [["|", "   "] for _ in range(7)] for a in subl] \
                             + ["|", "\n"]

        def cellPos(i,j):
            return 1 + j + 8*i
        def paddingCheck(i,j):
            return ' ' if j + 8*i < 9 else ''

        blackPieces = []
        whitePieces = []

        # render the ASCII board content
        for i, chunk in enumerate(state):
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
                king = False
                if cell == Black:
                    piece = colored("b", 'red', attrs=['reverse'])
                    blackPieces.append(str(cellPos(i,j)))
                elif cell == White:
                    piece = colored("w", 'cyan', attrs=['reverse'])
                    whitePieces.append(str(cellPos(i,j)))
                elif cell == blackKing:
                    piece = colored("B", 'red', attrs=['reverse'])
                    king = True
                    blackPieces.append(("K" + str(cellPos(i,j))))
                elif cell == whiteKing:
                    piece = colored("B", 'cyan', attrs=['reverse'])
                    king = True
                    whitePieces.append(("K" + str(cellPos(i,j))))

                # initiate the board with values.
                board[x][y] = piece + str(cellPos(i,j)) + (paddingCheck(i,j))

        # generate the PDN for the current board.
        def genPDN(blackPieces, whitePieces):
            Black_list = ','.join(blackPieces)
            White_list = ','.join(whitePieces)
            li = "W" + White_list + ":" + "B" + Black_list
            return li
        print(genPDN(blackPieces,whitePieces))

        return "".join(map(lambda x: "".join(x), board))
