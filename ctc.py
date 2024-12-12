import numpy as np

INF_COST = 1e34


def sequence_mask(lengths, maxlen):
    """Computes sequence mask.

    Args:
        lengths: [batch]
        maxlen: int, output sequence dimension

    Returns:
        Output binary mask of the shape [batch, maxlen].
    """

    return np.arange(maxlen)[None, :] < lengths[:, None]


def reverse_sequence_by_length(seq, lengths, flip_dim=1):
    """Reverse input sequences according to their lengths.

    Args:
        seq: [batch, t, ...]
        lengths: [batch], non-padded lengths along reverse dimension
        flip_dim: int, the dimension along which to flip.

    Returns:
        Tensor with the same dimension as input seq, but with sequences reversed in flip_dim.
    """

    ndims = seq.ndim
    # Assume dimension 0 is batch.
    dim_permute = [0] + [flip_dim] + list(range(1, flip_dim)) + list(range(flip_dim + 1, ndims))
    seq_permute = np.transpose(seq, dim_permute)

    output = []
    for i, l in enumerate(lengths):
        x = np.concatenate([np.flip(seq_permute[i, :l], [0]), seq_permute[i, l:]])
        output.append(x)
    output = np.stack(output)

    dim_permute = [0] + list(range(2, flip_dim + 1)) + [1] + list(range(flip_dim + 1, ndims))
    return np.transpose(output, dim_permute)


def compute_forced_alignment(costs, input_lens, labels, output_lens, blank_id=0):
    """Computes CTC forced alignment, which is the optimal token sequence for the given label sequenc
e.

    The algorithm is similar to that of computing alpha values, except that logaddexp is replaced wit
h maximum in
    computing the dynamic programming table.

    Args:
        costs: [batch, t_in, num_classes]
        input_lens: [batch]
        labels: [batch, t_out]
        output_lens: [batch]
        blank_id: int, token id of blank.

    Returns:
        CTC optimal path costs of shape [batch], optimal alignments of shape [batch, t_in].
    """

    B, Tin, C = costs.shape
    _, Tout = labels.shape
    L = 2 * Tout + 1
    alpha = INF_COST * np.ones([B, Tin, L], dtype=costs.dtype)
    # Save moves at each node: 0 for horizontal moves, 1 for diagonal moves, 2 for moves between distinct non-blanks.
    choices = np.zeros([B, Tin, L], dtype=np.int32)

    # t=0, l=0, blank
    # update has shape [batch]
    alpha[:, 0, 0] = costs[:, 0, blank_id]
    # t=0, l=1, first non-blank
    alpha[:, 0, 1] = np.take_along_axis(costs[:, 0, :], labels[:, 0, None], 1).squeeze(1)

    for t in range(1, Tin):

        # Horizontal moves between blanks.
        # shape [batch, 1]
        costs_blank = costs[:, t, blank_id, None]
        # Update has shape [batch, t_out + 1].
        alpha[:, t, 0::2] = alpha[:, t - 1, 0::2] + costs_blank

        # Horizontal moves between non-blanks.
        # shape [batch, t_out]
        costs_nonblank = np.take_along_axis(costs[:, t, :], labels, 1)
        # Update has shape [batch, t_out].
        alpha[:, t, 1::2] = alpha[:, t - 1, 1::2] + costs_nonblank

        # Diagonal moves, blank to non-blank.
        alpha_diagonal = alpha[:, t - 1, 0:-1:2] + costs_nonblank
        mask = np.greater(alpha[:, t, 1::2], alpha_diagonal)
        float_mask = mask.astype(np.float32)
        # Update has shape [batch, t_out].
        alpha[:, t, 1::2] = alpha[:, t, 1::2] * (1.0 - float_mask) + alpha_diagonal * float_mask
        # Update choice from 0 (default) to 1.
        int_mask = mask.astype(np.int32)
        choices[:, t, 1::2] = choices[:, t, 1::2] * (1 - int_mask) + 1 * int_mask

        # Diagonal moves, non-blank to blank.
        # Update has shape [batch, t_out].
        alpha_diagonal = alpha[:, t - 1, 1::2] + costs_blank
        mask = np.greater(alpha[:, t, 2::2], alpha_diagonal)
        float_mask = mask.astype(np.float32)
        alpha[:, t, 2::2] = alpha[:, t, 2::2] * (1.0 - float_mask) + alpha_diagonal * float_mask
        # Update choice from 0 (default) to 1.
        int_mask = mask.astype(np.int32)
        choices[:, t, 2::2] = choices[:, t, 2::2] * (1 - int_mask) + 1 * int_mask

        # Diagonal moves, non-blank to distinct non-blank.
        # Check if previous non-blank equals current non-blank.
        # shape [batch, t_out-1]
        distinct_mask = (labels[:, 1:] != labels[:, :-1]).astype(np.float32)
        costs_nonblank_distinct = costs_nonblank[:, 1:] * distinct_mask + INF_COST * (1.0 - distinct_mask)
        # Update has shape [batch, t_out - 1].
        alpha_diagonal = alpha[:, t - 1, 1:-3:2] + costs_nonblank_distinct
        mask = np.greater(alpha[:, t, 3::2], alpha_diagonal)
        float_mask = mask.astype(np.float32)
        alpha[:, t, 3::2] = alpha[:, t, 3::2] * (1.0 - float_mask) + alpha_diagonal * float_mask
        # Update choice to 2.
        int_mask = mask.astype(np.int32)
        choices[:, t, 3::2] = choices[:, t, 3::2] * (1 - int_mask) + 2 * int_mask

    # All labels including blanks.
    all_tokens = np.ones([B, L], dtype=np.int32) * blank_id
    all_tokens[:, 1::2] = labels
    input_seq_mask = sequence_mask(input_lens, Tin).astype(np.int32)

    # Reverse in time for back tracking.
    alpha_reversed = reverse_sequence_by_length(alpha, input_lens)
    # shape [batch, t_in, L], set out-of-bound choices to 0.
    choices_reversed = reverse_sequence_by_length(choices * input_seq_mask[:, :, None], input_lens)

    # shape [batch, 1]
    output_lens = output_lens[:, None]
    alpha_blank = np.take_along_axis(alpha_reversed[:, 0, :], 2 * output_lens, 1)
    alpha_nonblank = np.take_along_axis(alpha_reversed[:, 0, :], 2 * output_lens - 1, 1)

    # costs of best paths.
    blank_vs_non_mask = alpha_blank < alpha_nonblank
    # shape [batch, 1], indexing into all_tokens of length L.
    last_index = np.where(blank_vs_non_mask, 2 * output_lens, 2 * output_lens - 1)
    last_token = np.take_along_axis(all_tokens, last_index, 1)
    # Back pointer.
    last_choice = np.take_along_axis(choices_reversed[:, 0, :], last_index, 1)

    blank_vs_non_mask = blank_vs_non_mask.astype(np.float32)
    best_costs = alpha_blank * blank_vs_non_mask + alpha_nonblank * (1 - blank_vs_non_mask)
    best_costs = best_costs.squeeze(1)

    # Best path contains one token per input step.
    paths = [last_token]

    for t in range(1, Tin):
        # choice gives the change of index in all tokens.
        last_index = last_index - last_choice
        last_token = np.take_along_axis(all_tokens, last_index, 1)
        paths.append(last_token)
        last_choice = np.take_along_axis(choices_reversed[:, t, :], last_index, 1)

    paths = np.concatenate(paths, 1)
    paths = reverse_sequence_by_length(paths, input_lens)
    # Set out-of-bound tokens to blank.
    paths = paths * input_seq_mask + blank_id * (1 - input_seq_mask)

    return best_costs, paths
