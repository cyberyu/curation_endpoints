'''
Taken from FairSeq:https://github.com/pytorch/fairseq with slight modifications applied
'''
import torch.nn as nn

DEFAULT_MAX_SOURCE_POSITIONS = 1e5
DEFAULT_MAX_TARGET_POSITIONS = 1e5

def LSTM(input_size, hidden_size, **kwargs):
    m = nn.LSTM(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.1, 0.1)
    return m

class LSTMEncoder(nn.Module):
    """LSTM encoder."""
    def __init__(
        self, embed_dim=768, hidden_size=768, num_layers=1,
        dropout_in=0.1, dropout_out=0.1, bidirectional=True, padding_value=0.,
        max_source_positions=DEFAULT_MAX_SOURCE_POSITIONS
    ):
        super(LSTMEncoder, self).__init__()
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.max_source_positions = max_source_positions

        self.dropout_in = nn.Dropout(dropout_in)
        self.dropout_out = nn.Dropout(dropout_out)

        self.lstm = LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_out if num_layers > 1 else 0.,
            bidirectional=bidirectional,
        )

        self.padding_value = padding_value

        self.output_units = hidden_size
        if bidirectional:
            self.output_units *= 2

    def forward(self, src_embs, src_lengths):
        '''
        :param src_embs: Should be (bsz, seqlen, emb_dim)
        :param src_lengths:
        :return:
        '''
        bsz, seqlen, emb_sz = src_embs.size()

        x = self.dropout_in(src_embs)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # pack embedded source tokens into a PackedSequence
        packed_x = nn.utils.rnn.pack_padded_sequence(x, src_lengths.data.tolist(), enforce_sorted=False)

        # apply LSTM
        if self.bidirectional:
            state_size = 2 * self.num_layers, bsz, self.hidden_size
        else:
            state_size = self.num_layers, bsz, self.hidden_size
        h0 = x.new_zeros(*state_size)
        c0 = x.new_zeros(*state_size)
        packed_outs, (final_hiddens, final_cells) = self.lstm(packed_x, (h0, c0))

        # unpack outputs and apply dropout
        x, _ = nn.utils.rnn.pad_packed_sequence(packed_outs, padding_value=self.padding_value)
        x = self.dropout_out(x)
        assert list(x.size()) == [seqlen, bsz, self.output_units]

        if self.bidirectional:

            def combine_bidir(outs):
                out = outs.view(self.num_layers, 2, bsz, -1).transpose(1, 2).contiguous()
                return out.view(self.num_layers, bsz, -1)

            final_hiddens = combine_bidir(final_hiddens)
            final_cells = combine_bidir(final_cells)

        return x, final_hiddens, final_cells


    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return self.max_source_positions

