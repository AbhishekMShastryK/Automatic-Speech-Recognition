import numpy as np

def compute_forced_alignment(S, input_lengths, Y, output_lengths):
    """
    Compute the forced alignment for CTC (Connectionist Temporal Classification).

    Args:
    S (np.array): Negative log-probabilities, shape [B, T, C]
    input_lengths (np.array): Actual lengths of utterances, shape [B]
    Y (np.array): Output token sequences, shape [B, L]
    output_lengths (np.array): Actual lengths of outputs, shape [B]

    Returns:
    best_costs (np.array): Costs of optimal alignments, shape [B]
    paths (np.array): Optimal alignments, shape [B, T]
    """

    B, T = S.shape[:2]  # B: batch size, T: max time steps
    best_costs = np.zeros(B, dtype=np.float32)
    paths = np.zeros((B, T), dtype=np.int32)

    for b in range(B):
        # Process each utterance in the batch
        utterance_length = input_lengths[b]  # Actual length of current utterance
        output_seq_length = output_lengths[b]  # Actual length of current output sequence
        current_output_seq = Y[b, :output_seq_length]  # Current output sequence

        # Prepare Y' by inserting blanks between labels and at the beginning and end
        Y_prime = np.zeros(2 * output_seq_length + 1, dtype=np.int32)
        Y_prime[1::2] = current_output_seq  # Odd indices are filled with labels from current_output_seq

        # Initialize alpha (costs) and backpointers
        alpha = np.full((utterance_length, 2 * output_seq_length + 1), np.inf, dtype=np.float32)
        alpha[0, 0] = S[b, 0, 0]  # Cost for blank (token 0) at first timestep
        alpha[0, 1] = S[b, 0, current_output_seq[0]]  # Cost for first label at first timestep
        
        # backpointers stores the index of the previous state that led to the current state
        # It's used for reconstructing the optimal alignment path
        backpointers = np.zeros((utterance_length, 2 * output_seq_length + 1), dtype=np.int32)

        # Fill alpha table and backpointers
        for t in range(1, utterance_length):
            for l in range(2 * output_seq_length + 1):
                if l == 0:
                    # Only option for blank is to stay on blank
                    alpha[t, l] = alpha[t-1, l] + S[b, t, 0]
                    backpointers[t, l] = l
                else:
                    if Y_prime[l] == 0 or (l >= 2 and Y_prime[l] == Y_prime[l-2]):
                        # For blank or repeated label, consider staying or moving one step
                        prev_costs = [alpha[t-1, l], alpha[t-1, l-1]]
                        alpha[t, l] = min(prev_costs) + S[b, t, Y_prime[l]]
                        backpointers[t, l] = l - np.argmin(prev_costs)
                    else:
                        # For new label, consider staying, moving one, or moving two steps
                        prev_costs = [alpha[t-1, l], alpha[t-1, l-1], alpha[t-1, l-2]]
                        alpha[t, l] = min(prev_costs) + S[b, t, Y_prime[l]]
                        backpointers[t, l] = l - np.argmin(prev_costs)

        # Find best cost at final timestep (utterance_length - 1)
        best_costs[b] = min(alpha[utterance_length-1, -1], alpha[utterance_length-1, -2])

        # Backtrack to find the optimal alignment A*
        optimal_alignment = []
        l = 2 * output_seq_length if alpha[utterance_length-1, -1] < alpha[utterance_length-1, -2] else 2 * output_seq_length - 1
        for t in range(utterance_length-1, -1, -1):
            optimal_alignment.append(Y_prime[l])
            l = backpointers[t, l]

        # Store the reversed path (optimal_alignment)
        paths[b, :utterance_length] = optimal_alignment[::-1]

    return best_costs, paths