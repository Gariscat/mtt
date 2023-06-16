'''
    Reference: Homework 3 - CSCI-SHU 376 Natural Language Processing by Prof. Yik-Cheung (Wilson) Tam
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def gelu_fast(x):
    return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))

def gelu_new(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

def mish(x):
    return x * torch.tanh(torch.nn.functional.softplus(x))

def linear_act(x):
    return x

ACT2FN = {
    "relu": F.relu,
    "silu": F.silu,
    "swish": F.silu,
    "gelu": F.gelu,
    "tanh": torch.tanh,
    "gelu_new": gelu_new,
    "gelu_fast": gelu_fast,
    "mish": mish,
    "linear": linear_act,
    "sigmoid": torch.sigmoid,
}

# Transformer config
class TransformerConfig:
  def __init__(self,
    input_size=128,
    vocab_size=24,
    hidden_size=192,
    num_hidden_layers=3,
    num_attention_heads=3,
    #intermediate_size=3072,
    intermediate_size=192,
    hidden_act="gelu",
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    max_position_embeddings=128,
    initializer_range=0.02,
    layer_norm_eps=1e-12,
    pad_token_id=0,
    **kwargs
  ):
    self.input_size = input_size
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads
    self.hidden_act = hidden_act
    self.intermediate_size = intermediate_size
    self.hidden_dropout_prob = hidden_dropout_prob
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.max_position_embeddings = max_position_embeddings
    self.initializer_range = initializer_range
    self.layer_norm_eps = layer_norm_eps
    self.pad_token_id = pad_token_id

## config = TransformerConfig()

class Embeddings(nn.Module):
    """
    Input: Input ids of shape (B,N) where B is batch size, N is sequence length
    Return: word embedding and position embedding
    This corresponds to the purpose box and position encoding in Slide 17 in transformers.ppt
    """

    def __init__(self, config):
        super().__init__()

        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

        # TODO:
        # Require additional 2 extra tokens [PAD] and [SEP]
        ### self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.word_embeddings = nn.Linear(config.input_size, config.hidden_size)

        # TODO:
        # position embeddings. Refer to config for the maximum number of positions we want
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # TODO:
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # TODO: use config.hidden_dropout_prob
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def forward(self, input_vec):
        """
        input_vec: a batch of padded sentences of shape (B, N, D) where N is the max sequence length, D is the length of one input vector
        """
        input_shape = input_vec.size()
        seq_length = input_shape[1]

        # TODO: get position embeddings
        position_ids = self.position_ids[:, :seq_length]  # This will return the position IDS for each word in a batch
        position_embeds = self.position_embeddings(position_ids)

        # TODO: get word embeddings
        word_embeds = self.word_embeddings(input_vec)

        # TODO: Sum of word embedding and position embedding to obtain composite embeddings
        embeddings = position_embeds + word_embeds

        # TODO: Apply layer normalization
        embeddings = self.layer_norm(embeddings)

        # TODO: Apply dropout after layer normalization
        embeddings = self.dropout(embeddings)

        return embeddings


class MultiHeadAttention(nn.Module):
    """
    Q, K, V Multi-head attention
    This corresponds to the orange box in Slide 17 in transformers.ppt
    """

    def __init__(self, config):
        super().__init__()
        # We implement the new Transformer++ here
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size

        # this is for dropping out attention_weights
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        # Q, K, V projections for all heads
        # TODO
        # (Head, H, Head_Dim)
        self.register_parameter('Q', nn.Parameter(torch.rand(self.num_attention_heads, self.attention_head_size,
                            self.attention_head_size)))
        self.register_parameter('K', nn.Parameter(torch.rand(self.num_attention_heads, self.attention_head_size,
                            self.attention_head_size)))
        self.register_parameter('V', nn.Parameter(torch.rand(self.num_attention_heads, self.attention_head_size,
                            self.attention_head_size)))

        self.register_parameter('W', nn.Parameter(torch.rand(self.attention_head_size)))  # (Head_Dim, )
        self.out = nn.Linear(self.attention_head_size, self.attention_head_size)

    def _transpose_for_scores(self, x):
        """
        You may need this function for multi-head attention.
        Input: 3D shape (B, N, hidden_size)
        Reshape the last dimension (hidden feature vector) into (head, head_dimension)
        Output: 4D shape (B, head, N, head_dim), where hidden_size = head * head_dim
        """
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)

        # Before: (B, N, Head, Head_Dim) with axis index 0,1,2,3
        # After: (B, Head, N, Head_Dim), i.e. multiple transformer blocks per head taking in input x
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        """
        hidden_states: input embeddings tensor of shape (B, N, H)
        """
        # TODO
        B, N, H = hidden_states.shape
        context = None
        #           (Head, B*N, Head_Dim)                      -> (Head, B, N, Head_Dim)
        qv = torch.matmul(hidden_states.reshape(-1, H), self.Q).reshape(self.num_attention_heads, B, N, -1)
        kv = torch.matmul(hidden_states.reshape(-1, H), self.K).reshape(self.num_attention_heads, B, N, -1)
        vv = torch.matmul(hidden_states.reshape(-1, H), self.V).reshape(self.num_attention_heads, B, N, -1)
        d_k = math.sqrt(self.attention_head_size)
        attention_weights = torch.matmul(qv, kv.transpose(-1, -2)) / d_k  # (Head, B, N, N)
        if attention_mask is not None:
            attention_weights = attention_weights + attention_mask.transpose(0,
                                                                             1)  # attention_mask is batch_first while attention_weights is not
        attention_weights = torch.nn.Softmax(dim=-1)(attention_weights)
        attention_weights = self.dropout(attention_weights)
        head_values = torch.matmul(attention_weights, vv)  # (Head, B, N, Head_Dim)
        # head weights: w_i
        head_weights = torch.matmul(head_values, self.W)  # (Head, B, N)
        head_weights = nn.Softmax(dim=0)(head_weights)  # softmax along the "head" axis
        context = (head_values * head_weights.unsqueeze(-1)).sum(0)  # (B, N, Head_Dim)
        context = self.out(context)  # (B, N, H)
        # return tuple of values
        outputs = (context, attention_weights, head_weights)

        return outputs


class AddAndNormalize(nn.Module):
    """
    Add the input tensor and apply layer normalize.
    See slide 19 in transformers.ppt
    We have linear projection, dropout, and layer norm here!
    """

    def __init__(self, config):
        super().__init__()

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # TODO
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, input_tensor):
        """
        Input: 1. hidden_states from MultiHeadAttention, 2. Input tensor of transformer to form a highway
        """
        # TODO: apply linear projection
        # 1 line of code
        hidden_states = self.dense(hidden_states)

        # TODO: apply dropout here
        # 1 line of code
        hidden_states = self.dropout(hidden_states)

        # TODO: apply residual connection and layer normalization
        # 1 line of code
        hidden_states = self.layer_norm(hidden_states + input_tensor)

        return hidden_states


class FeedForward(nn.Module):
    """
    Introduce non-linearity
    This corresponds to the Feedforward Layer box in slide 19 in transformers.ppt
    You do not need to modify this class.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states


class Transformer(nn.Module):
    """
    A Transformer++ block with attentive head
    Slide 19 in transformers.ppt
    """

    def __init__(self, config):
        super().__init__()
        # create the 4 components in a transformer block
        self.attention = MultiHeadAttention(config)
        self.add_and_normalize0 = AddAndNormalize(config)
        self.feed_forward = FeedForward(config)
        self.add_and_normalize1 = AddAndNormalize(config)

    def forward(self, hidden_states, attention_mask=None):
        """
        Given the hidden_states tensor, you feed them to multi-head attention,
        followed by add_and_normalize, followed by feed_forward, followed by another
        add_and_normalize
        """
        # TODO: construct the transformer block from the 4 components from __init__(.)
        # 4 lines of code
        input_tensor = hidden_states
        hidden_states = self.add_and_normalize0(self.attention(hidden_states, attention_mask)[0], input_tensor)
        input_tensor = hidden_states
        hidden_states = self.add_and_normalize1(self.feed_forward(hidden_states), input_tensor)

        return hidden_states


class TransformerLM(nn.Module):
    """
    TransformerLM consists of input embedding layer followed by stacks of Transformer blocks
    Use Causal masking (can only access the word histories from the left)
    This corresponds to the full picture in Slide 17 in transformers.ppt
    """

    def __init__(self, config):
        super().__init__()

        # save config
        self.config = config

        # create WE and PE layer
        self.embeddings = Embeddings(config)

        # create a stack of transformer blocks
        self.layers = nn.ModuleList([Transformer(config) for _ in range(config.num_hidden_layers)])

        # TODO: create an output layer for next word prediction
        # Hint: Use config to access necessary variables to create the output projection
        self.dense_pitch = nn.Linear(config.hidden_size, config.vocab_size)
        self.dense_value = nn.Linear(config.hidden_size, 1)

        # init weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _make_causal_mask(self, input_vec, dtype, pad_token_id):
        batch_size, seq_length, _ = input_vec.size()
        ### seq_ids = torch.arange(seq_length).to(device)
        seq_ids = torch.arange(seq_length).to(input_vec.device)
        x = seq_ids[None, None, :].repeat(batch_size, seq_length, 1)
        y = seq_ids[None, :, None]
        causal_mask = (x <= y).to(dtype)
        ### attention_mask = torch.ne(input_vec, pad_token_id)
        attention_mask = 1
        # TODO: You need to make use of causal mask and attention mask to create a final mask
        # 1 line of code
        ### mask = causal_mask * attention_mask.unsqueeze(-1)
        mask = causal_mask * attention_mask

        mask = mask.to(dtype)

        # TODO: convert mask into log domain so that it can be applied to logits before nn.Softmax
        # log(0) is approximated by -10000.0
        # 1 line of code
        mask = torch.fmax(torch.log(mask), torch.ones_like(mask) * (-10000))

        # Return Shape: (B, num_head, SequenceLength I, SequenceLength J)
        # Will broadcast at axis=1 denoting number of attention heads
        return mask[:, None, :, :]

    def forward(self, input_vec, attention_mask=None):
        # obtain WE+PE
        hidden_states = self.embeddings(input_vec)

        # create 1. causal masking and 2. sentence length mask (to exclude the padding positions from attention)
        if attention_mask is None:
            # Shape: (B, Head, N, N)
            ### attention_mask = self._make_causal_mask(input_vec, hidden_states.dtype, config.pad_token_id).to(device)
            attention_mask = self._make_causal_mask(input_vec, hidden_states.dtype, self.config.pad_token_id).to(input_vec.device)

        for i, layer_module in enumerate(self.layers):
            hidden_states = layer_module(hidden_states, attention_mask=attention_mask)

        # TODO: project hidden size to output vocabulary size
        pitch_logits = self.dense_pitch(hidden_states)
        value_estimates = self.dense_value(hidden_states)

        return pitch_logits, value_estimates, hidden_states