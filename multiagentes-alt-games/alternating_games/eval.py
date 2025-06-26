import numpy as np

def heuristic_nocca(board: "Board", player: int) -> float:
    """
    Heurística para Nocca Nocca.
    player: 0 (BLACK) o 1 (WHITE)
    Retorna un valor en [-1, 1]
    """
    opponent = board._opponent(player)
    player_pieces = np.argwhere(board.squares == player)

    if player_pieces.size == 0:
        return -1.0  # sin piezas = derrota
    if board.check_for_winner() == player:
        return 1.0
    if board.check_for_winner() == opponent:
        return -1.0

    # --- 1. Avance promedio
    rows = player_pieces[:, 0]
    if player == 0:  # BLACK quiere subir (hacia fila 0)
        advance = 1.0 - np.mean(rows) / (board.squares.shape[0] - 1)
    else:            # WHITE quiere bajar (hacia fila 7)
        advance = np.mean(rows) / (board.squares.shape[0] - 1)
    advance_score = 2 * advance - 1  # [-1,1]

    # --- 2. Piezas bloqueadas
    blocked = 0
    movable = 0
    for x, y, _ in player_pieces:
        stack = board.squares[x][y]
        top_idx = np.max(np.argwhere(stack != -1))  # índice más alto ocupado
        if stack[top_idx] != player:
            blocked += 1
        else:
            movable += 1
    total = blocked + movable
    block_score = 1.0 - blocked / (total + 1e-5)  # +1e-5 para evitar /0
    block_score = 2 * block_score - 1  # [-1, 1]

    # --- 3. Amenaza de victoria (pieza en penúltima fila)
    near_goal = False
    for x, _, _ in player_pieces:
        if player == 0 and x == 1: near_goal = True
        if player == 1 and x == 6: near_goal = True
    threat_score = 1.0 if near_goal else 0.0

    # --- Combinación ponderada
    value = (
        0.5 * advance_score +
        0.3 * block_score +
        0.2 * threat_score
    )

    return np.clip(value, -1.0, 1.0)

def default_eval(self, agent: int) -> float:
        if agent not in self.agents:
            raise ValueError(f"Agent {agent} is not part of the game.")

        if self.terminated():
            return self.rewards[agent]
    
        player = self.agent_name_mapping[agent]
        return 0. * player