import random
import copy

class TeekoPlayer:
    """ An object representation for an AI game player for the game Teeko.
    """
    board = [[' ' for j in range(5)] for i in range(5)]
    pieces = ['b', 'r']

    def __init__(self):
        """ Initializes a TeekoPlayer object by randomly selecting red or black as its
        piece color.
        """
        self.my_piece = random.choice(self.pieces)
        self.opp = self.pieces[0] if self.my_piece == self.pieces[1] else self.pieces[1]

    def is_drop_phase(self, state):
        ele_num = 0
        for row in range(len(state)):
            for cell in state[row]:
                if cell != ' ':
                    ele_num += 1
        return ele_num < 8
    
    def is_pos_valid(self, row, col):
            return row >= 0 and row <= 4 and col >= 0 and col <= 4

    def succ(self, state, next_player):
        successors = []
        dirs = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]

        drop_phase = self.is_drop_phase(state)

        # drop phase
        if drop_phase:
            for row in range(len(state)):
                for col in range(len(state[row])):
                    if state[row][col] == ' ':
                        tmp_board = copy.deepcopy(state)
                        tmp_board[row][col] = next_player
                        successors.append(tmp_board)
        # mave phase
        else:
            for row in range(len(state)):
                for col in range(len(state[row])):
                    if state[row][col] == next_player:
                        for dir in dirs:
                            new_row = row + dir[0]
                            new_col = col + dir[1]
                            if self.is_pos_valid(new_row, new_col) and state[new_row][new_col] == ' ':
                                tmp_board = copy.deepcopy(state)
                                tmp_board[row][col] = ' '
                                tmp_board[new_row][new_col] = next_player
                                successors.append(tmp_board)
        
        return successors
    
    def heuristic_game_value(self, state):
        score = self.game_value(state)
        score_table = [[4,6,5,6,4],
                       [6,10,10,10,6],
                       [5,10,12,10,5],
                       [6,10,10,10,6],
                       [4,6,5,6,4]]
        if score == 0:
            for i in range(5):
                for j in range(5):
                    if state[i][j] == self.my_piece:
                        score += score_table[i][j]
                        if self.is_pos_valid(i, j+1) and state[i][j] == state[i][j+1]:
                            score += 3 if (self.is_pos_valid(i, j+2) and state[i][j] == state[i][j+2]) else 2
                        if self.is_pos_valid(i+1, j) and state[i][j] == state[i+1][j]:
                            score += 3 if (self.is_pos_valid(i+2, j) and state[i][j] == state[i+2][j]) else 2
                        if self.is_pos_valid(i+1, j+1) and state[i][j] == state[i+1][j+1]:
                            score += 3 if (self.is_pos_valid(i+2, j+2) and state[i][j] == state[i+2][j+2]) else 2
                    elif state[i][j] == self.opp:
                        score -= score_table[i][j]
                        if self.is_pos_valid(i, j+1) and state[i][j] == state[i][j+1]:
                            score -= 3 if (self.is_pos_valid(i, j+2) and state[i][j] == state[i][j+2]) else 2
                        if self.is_pos_valid(i+1, j) and state[i][j] == state[i+1][j]:
                            score -= 3 if (self.is_pos_valid(i+2, j) and state[i][j] == state[i+2][j]) else 2
                        if self.is_pos_valid(i+1, j+1) and state[i][j] == state[i+1][j+1]:
                            score -= 3 if (self.is_pos_valid(i+2, j+2) and state[i][j] == state[i+2][j+2]) else 2
        
        # normalize
        score = score * 2 / 100

        return score

    def max_value(self, state, depth, player_turn, alpha, beta):
        # check if terminate
        terminal_state = self.game_value(state)
        if terminal_state != 0:
            return terminal_state
        elif depth == 0:
            return self.heuristic_game_value(state)
        elif player_turn == self.my_piece:
            for suc in self.succ(state, self.my_piece):
                alpha = max(alpha, self.max_value(suc, depth-1, self.opp, alpha, beta))
                if alpha >= beta:
                    return beta
            return alpha
            # return max(self.max_value(suc, depth-1, self.opp) for suc in self.succ(state, self.my_piece))
        else:
            for suc in self.succ(state, self.opp):
                beta = min(beta, self.max_value(suc, depth-1, self.my_piece, alpha, beta))
                if alpha >= beta:
                    return alpha
            return beta
            # return min(self.max_value(suc, depth-1, self.my_piece) for suc in self.succ(state, self.opp))

    def make_move(self, state):
        """ Selects a (row, col) space for the next move. You may assume that whenever
        this function is called, it is this player's turn to move.

        Args:
            state (list of lists): should be the current state of the game as saved in
                this TeekoPlayer object. Note that this is NOT assumed to be a copy of
                the game state and should NOT be modified within this method (use
                place_piece() instead). Any modifications (e.g. to generate successors)
                should be done on a deep copy of the state.

                In the "drop phase", the state will contain less than 8 elements which
                are not ' ' (a single space character).

        Return:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

        Note that without drop phase behavior, the AI will just keep placing new markers
            and will eventually take over the board. This is not a valid strategy and
            will earn you no points.
        """
        # detect drop phase
        # drop_phase = self.is_drop_phase(state)   

        # implement a minimax algorithm
        successors = self.succ(state, self.my_piece)
        next_move_index = -1
        max_score = -2
        for i in range(len(successors)):
            next_state = successors[i]
            next_score = self.max_value(next_state, 3, self.opp, -1.5, 1.5)
            if next_score > max_score:
                max_score = next_score
                next_move_index = i

        # get move
        move = []
        next_state = successors[next_move_index]
        row = -1
        col = -1
        for i in range(5):
            for j in range(5):
                if state[i][j] == self.my_piece and next_state[i][j] == ' ':
                    move.insert(0, (i, j))
                if state[i][j] == ' ' and next_state[i][j] == self.my_piece:
                    row = i
                    col = j

        # ensure the destination (row,col) tuple is at the beginning of the move list
        move.insert(0, (row, col))

        return move

    def opponent_move(self, move):
        """ Validates the opponent's next move against the internal board representation.
        You don't need to touch this code.

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.
        """
        # validate input
        if len(move) > 1:
            source_row = move[1][0]
            source_col = move[1][1]
            if source_row != None and self.board[source_row][source_col] != self.opp:
                self.print_board()
                print(move)
                raise Exception("You don't have a piece there!")
            if abs(source_row - move[0][0]) > 1 or abs(source_col - move[0][1]) > 1:
                self.print_board()
                print(move)
                raise Exception('Illegal move: Can only move to an adjacent space')
        if self.board[move[0][0]][move[0][1]] != ' ':
            raise Exception("Illegal move detected")
        # make move
        self.place_piece(move, self.opp)

    def place_piece(self, move, piece):
        """ Modifies the board representation using the specified move and piece

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

                This argument is assumed to have been validated before this method
                is called.
            piece (str): the piece ('b' or 'r') to place on the board
        """
        if len(move) > 1:
            self.board[move[1][0]][move[1][1]] = ' '
        self.board[move[0][0]][move[0][1]] = piece

    def print_board(self):
        """ Formatted printing for the board """
        for row in range(len(self.board)):
            line = str(row)+": "
            for cell in self.board[row]:
                line += cell + " "
            print(line)
        print("   A B C D E")

    def game_value(self, state):
        """ Checks the current board status for a win condition

        Args:
        state (list of lists): either the current state of the game as saved in
            this TeekoPlayer object, or a generated successor state.

        Returns:
            int: 1 if this TeekoPlayer wins, -1 if the opponent wins, 0 if no winner

        TODO: complete checks for diagonal and box wins
        """
        # check horizontal wins
        for row in state:
            for i in range(2):
                if row[i] != ' ' and row[i] == row[i+1] == row[i+2] == row[i+3]:
                    return 1 if row[i]==self.my_piece else -1

        # check vertical wins
        for col in range(5):
            for i in range(2):
                if state[i][col] != ' ' and state[i][col] == state[i+1][col] == state[i+2][col] == state[i+3][col]:
                    return 1 if state[i][col]==self.my_piece else -1

        # check \ diagonal wins
        for i in range(2):
            for j in range(2):
                if state[i][j] != ' ' and state[i][j] == state[i+1][j+1] == state[i+2][j+2] == state[i+3][j+3]:
                    return 1 if state[i][j]==self.my_piece else -1

        # check / diagonal wins
        for i in range(2):
            for j in range(3, 5):
                if state[i][j] != ' ' and state[i][j] == state[i+1][j-1] == state[i+2][j-2] == state[i+3][j-3]:
                    return 1 if state[i][j]==self.my_piece else -1

        # check box wins
        for i in range(4):
            for j in range(4):
                if state[i][j] != ' ' and state[i][j] == state[i][j+1] == state[i+1][j] == state[i+1][j+1]:
                    return 1 if state[i][j]==self.my_piece else -1

        return 0 # no winner yet

############################################################################
#
# THE FOLLOWING CODE IS FOR SAMPLE GAMEPLAY ONLY
#
############################################################################
def main():
    print('Hello, this is Samaritan')
    ai = TeekoPlayer()
    piece_count = 0
    turn = 0

    # drop phase
    while piece_count < 8 and ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved at "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                player_move = input("Move (e.g. B3): ")
                while player_move[0] not in "ABCDE" or player_move[1] not in "01234":
                    player_move = input("Move (e.g. B3): ")
                try:
                    ai.opponent_move([(int(player_move[1]), ord(player_move[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        piece_count += 1
        turn += 1
        turn %= 2

    # move phase - can't have a winner until all 8 pieces are on the board
    while ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved from "+chr(move[1][1]+ord("A"))+str(move[1][0]))
            print("  to "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                move_from = input("Move from (e.g. B3): ")
                while move_from[0] not in "ABCDE" or move_from[1] not in "01234":
                    move_from = input("Move from (e.g. B3): ")
                move_to = input("Move to (e.g. B3): ")
                while move_to[0] not in "ABCDE" or move_to[1] not in "01234":
                    move_to = input("Move to (e.g. B3): ")
                try:
                    ai.opponent_move([(int(move_to[1]), ord(move_to[0])-ord("A")),
                                    (int(move_from[1]), ord(move_from[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        turn += 1
        turn %= 2

    ai.print_board()
    if ai.game_value(ai.board) == 1:
        print("AI wins! Game over.")
    else:
        print("You win! Game over.")


if __name__ == "__main__":
    main()
