import config
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch_scatter import scatter_max
from data_utils import UNK_ID

INF = 1e12


class Encoder(nn.Module):
    def __init__(self, embeddings, entity, relation, vocab_size, embedding_size, hidden_size, num_layers, dropout):
        super(Encoder, self).__init__()
        graph_size = config.graph_vector_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.tag_embedding = nn.Embedding(3, 3)
        self.entity_embedding = nn.Embedding(config.entity_size, config.graph_vector_size)
        self.relation_embedding = nn.Embedding(config.relation_size + 1, config.graph_vector_size)
        lstm_input_size = embedding_size + 3 + 2 * config.graph_vector_size

        if embeddings is not None:
            self.embedding = nn.Embedding(vocab_size, embedding_size). \
                from_pretrained(embeddings, freeze=config.freeze_embedding)

        if entity is not None:
            self.entity_embedding = nn.Embedding(config.entity_size, config.graph_vector_size). \
                from_pretrained(entity, freeze=True)
        if relation is not None:
            self.relation_embedding = nn.Embedding(config.relation_size + 1, config.graph_vector_size). \
                from_pretrained(relation, freeze=True)

        self.num_layers = num_layers
        if self.num_layers == 1:
            dropout = 0.0
        self.lstm = nn.LSTM(lstm_input_size, hidden_size, dropout=dropout,
                            num_layers=num_layers, bidirectional=True, batch_first=True)
        self.linear_trans = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.update_layer = nn.Linear(4 * hidden_size, 2 * hidden_size, bias=False)
        self.gate = nn.Linear(4 * hidden_size, 2 * hidden_size, bias=False)

        self.head_graph = nn.Linear(graph_size, graph_size, bias=False)
        self.rela_graph = nn.Linear(graph_size, graph_size, bias=False)
        self.tail_graph = nn.Linear(graph_size, graph_size, bias=False)

    def gated_self_attn(self, queries, memories, mask):
        # queries: [b, t, d]
        # memories: [b, t, d]
        # mask: [b, t]
        energies = torch.matmul(queries, memories.transpose(1, 2))  # [b, t, t]
        mask = mask.unsqueeze(1)
        energies = energies.masked_fill(mask == 0, value=-1e12)

        scores = F.softmax(energies, dim=2)
        context = torch.matmul(scores, queries)
        inputs = torch.cat([queries, context], dim=2)
        f_t = torch.tanh(self.update_layer(inputs))
        g_t = torch.sigmoid(self.gate(inputs))
        updated_output = g_t * f_t + (1 - g_t) * queries

        return updated_output

    def graph_stat_attention(self, embed_mask, cs_seq, mask):
        # ent_emb: [batch_size, srq_len, graph_dim]
        # cs_seq: [batch_size, srq_len ,max_com_num, 3]
        # mask: [batch_size, srq_len , max_com_num]
        mask = mask.bool()
        head_embedding = self.entity_embedding(cs_seq[:, :, :, 0])
        head_embedding = head_embedding.masked_fill(embed_mask, value=0)
        tail_embedding = self.entity_embedding(cs_seq[:, :, :, 2])
        tail_embedding = tail_embedding.masked_fill(embed_mask, value=0)
        rela_embedding = self.relation_embedding(cs_seq[:, :, :, 1])
        rela_embedding = rela_embedding.masked_fill(embed_mask, value=0)
        # [batch_size, srq_len, max_com_num, gd]
        b, t, m, gd = head_embedding.size()
        graph_embedding = torch.cat((head_embedding, tail_embedding), dim=-1)  # [b, t, m, 2 * gd]
        # print(graph_embedding.size())
        head_tail = self.head_graph(head_embedding) + self.tail_graph(tail_embedding)
        # print(head_tail.size())
        beta = torch.tanh(head_tail)
        # [b, t, m, gd]
        rel = self.rela_graph(rela_embedding).view(b * t * m, 1, gd)  # [btm,1,gd]
        beta = torch.matmul(rel, beta.view(b * t * m, gd, 1)).view(b, t, m)  # [b,t,m]
        beta = beta.masked_fill(mask, value=-1e12)
        alpha = F.softmax(beta, dim=2).view(b * t, 1, m)  # [bt, 1, m]
        graph_embedding = torch.matmul(alpha, graph_embedding.view(b * t, m, 2 * gd)).view(b, t,
                                                                                           2 * gd)  # [b, t, 2 * gd]
        return graph_embedding

    def forward(self, src_seq, src_len, tag_seq, cs_seq, mask_seq, embed_mask):
        # print(cs_seq[:, :, :, 0])
        total_length = src_seq.size(1)
        embedded = self.embedding(src_seq)
        tag_embedded = self.tag_embedding(tag_seq)
        # insert commonsense
        # ent_embedded = self.entity_embedding(src_seq)
        com_embedded = self.graph_stat_attention(embed_mask, cs_seq, mask_seq)

        embedded = torch.cat((embedded, tag_embedded, com_embedded), dim=2)
        packed = pack_padded_sequence(embedded,
                                      src_len,
                                      batch_first=True,
                                      enforce_sorted=False)
        outputs, states = self.lstm(packed)  # states : tuple of [4, b, d]
        outputs, _ = pad_packed_sequence(outputs,
                                         batch_first=True,
                                         total_length=total_length)  # [b, t, d]
        h, c = states

        # self attention
        mask = torch.sign(src_seq)
        memories = self.linear_trans(outputs)
        outputs = self.gated_self_attn(outputs, memories, mask)

        _, b, d = h.size()
        h = h.view(2, 2, b, d)  # [n_layers, bi, b, d]
        h = torch.cat((h[:, 0, :, :], h[:, 1, :, :]), dim=-1)

        c = c.view(2, 2, b, d)
        c = torch.cat((c[:, 0, :, :], c[:, 1, :, :]), dim=-1)
        concat_states = (h, c)

        return outputs, concat_states


class Decoder(nn.Module):
    def __init__(self, embeddings, vocab_size,
                 embedding_size, hidden_size, num_layers, dropout):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        if embeddings is not None:
            self.embedding = nn.Embedding(vocab_size, embedding_size). \
                from_pretrained(embeddings, freeze=config.freeze_embedding)

        if num_layers == 1:
            dropout = 0.0
        self.encoder_trans = nn.Linear(hidden_size, hidden_size)
        self.reduce_layer = nn.Linear(
            embedding_size + hidden_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True,
                            num_layers=num_layers, bidirectional=False, dropout=dropout)
        self.concat_layer = nn.Linear(2 * hidden_size, hidden_size)
        self.logit_layer = nn.Linear(hidden_size, vocab_size)

    @staticmethod
    def attention(query, memories, mask):
        # query : [b, 1, d]
        # memory : [b, t, d]
        energy = torch.matmul(query, memories.transpose(1, 2))  # [b, 1, t]
        energy = energy.squeeze(1).masked_fill(mask == 0, value=-1e12)
        attn_dist = F.softmax(energy, dim=1).unsqueeze(dim=1)  # [b, 1, t]
        context_vector = torch.matmul(attn_dist, memories)  # [b, 1, d]

        return context_vector, energy

    def get_encoder_features(self, encoder_outputs):
        return self.encoder_trans(encoder_outputs)

    def forward(self, trg_seq, ext_src_seq, init_states, encoder_outputs, encoder_mask):
        # trg_seq : [b,t]
        # init_states : [2,b,d]
        # encoder_outputs : [b,t,d]
        # init_states : a tuple of [2, b, d]
        device = trg_seq.device
        batch_size, max_len = trg_seq.size()

        hidden_size = encoder_outputs.size(-1)
        memories = self.get_encoder_features(encoder_outputs)
        logits = []
        # init decoder hidden states and context vector
        prev_states = init_states
        prev_context = torch.zeros((batch_size, 1, hidden_size))
        prev_context = prev_context.to(device)
        for i in range(max_len):
            y_i = trg_seq[:, i].unsqueeze(1)  # [b, 1]
            embedded = self.embedding(y_i)  # [b, 1, d]
            lstm_inputs = self.reduce_layer(
                torch.cat([embedded, prev_context], 2))
            output, states = self.lstm(lstm_inputs, prev_states)
            # encoder-decoder attention
            context, energy = self.attention(output, memories, encoder_mask)
            concat_input = torch.cat((output, context), dim=2).squeeze(dim=1)
            logit_input = torch.tanh(self.concat_layer(concat_input))
            logit = self.logit_layer(logit_input)  # [b, |V|]

            # maxout pointer network
            if config.use_pointer:
                num_oov = max(torch.max(ext_src_seq - self.vocab_size + 1), 0)
                zeros = torch.zeros((batch_size, num_oov), device=device)
                extended_logit = torch.cat([logit, zeros], dim=1)
                out = torch.zeros_like(extended_logit) - INF
                out, _ = scatter_max(energy, ext_src_seq, out=out)
                out = out.masked_fill(out == -INF, 0)
                logit = extended_logit + out
                logit = logit.masked_fill(logit == 0, -INF)

            logits.append(logit)
            # update prev state and context
            prev_states = states
            prev_context = context

        logits = torch.stack(logits, dim=1)  # [b, t, |V|]

        return logits

    def decode(self, y, ext_x, prev_states, prev_context, encoder_features, encoder_mask):
        # forward one step lstm
        # y : [b]
        embedded = self.embedding(y.unsqueeze(1))
        lstm_inputs = self.reduce_layer(torch.cat([embedded, prev_context], 2))
        output, states = self.lstm(lstm_inputs, prev_states)

        context, energy = self.attention(output,
                                         encoder_features,
                                         encoder_mask)
        concat_input = torch.cat((output, context), 2).squeeze(1)
        logit_input = torch.tanh(self.concat_layer(concat_input))
        logit = self.logit_layer(logit_input)  # [b, |V|]

        if config.use_pointer:
            batch_size = y.size(0)
            num_oov = max(torch.max(ext_x - self.vocab_size + 1), 0)
            zeros = torch.zeros((batch_size, num_oov), device=config.device)
            extended_logit = torch.cat([logit, zeros], dim=1)
            out = torch.zeros_like(extended_logit) - INF
            out, _ = scatter_max(energy, ext_x, out=out)
            out = out.masked_fill(out == -INF, 0)
            logit = extended_logit + out
            logit = logit.masked_fill(logit == -INF, 0)
            # forcing UNK prob 0
            logit[:, UNK_ID] = -INF

        return logit, states, context


class Seq2seq(nn.Module):
    def __init__(self, embedding=None, ent_embedding=None, rel_embedding=None):
        super(Seq2seq, self).__init__()
        self.encoder = Encoder(embedding,
                               ent_embedding,
                               rel_embedding,
                               config.vocab_size,
                               config.embedding_size,
                               config.hidden_size,
                               config.num_layers,
                               config.dropout)
        self.decoder = Decoder(embedding,
                               config.vocab_size,
                               config.embedding_size,
                               2 * config.hidden_size,
                               config.num_layers,
                               config.dropout)

    def forward(self, src_seq, tag_seq, cs_seq, mask_seq, embed_mask, ext_src_seq, trg_seq):
        enc_mask = torch.sign(src_seq)
        src_len = torch.sum(enc_mask, 1)
        enc_outputs, enc_states = self.encoder(src_seq, src_len, tag_seq, cs_seq, mask_seq, embed_mask)
        sos_trg = trg_seq[:, :-1].contiguous()

        logits = self.decoder(sos_trg, ext_src_seq,
                              enc_states, enc_outputs, enc_mask)
        return logits
