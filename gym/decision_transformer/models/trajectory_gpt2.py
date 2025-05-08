import torch
import torch.nn as nn

class GPT2Model(nn.Module):
    """
    Simplified GPT2 model compatible with decision_transformer.
    """
    def __init__(self, config):
        super().__init__()
        
        # Add n_ctx attribute to config if it doesn't exist
        if not hasattr(config, 'n_ctx'):
            config.n_ctx = getattr(config, 'max_position_embeddings', 1024)
            
        self.config = config
        self.h = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        
        self.use_layers = None
        
    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.wte(input_ids)
            
        hidden_states = inputs_embeds
        
        # Apply transformer blocks
        for block in self.h:
            hidden_states = block(hidden_states, attention_mask)
            
        # Apply final layer norm
        hidden_states = self.ln_f(hidden_states)
        
        # Return as dictionary to match Hugging Face's format
        return {
            'last_hidden_state': hidden_states,
            'past_key_values': None,
            'hidden_states': None,
            'attentions': None,
        }
        
    def set_layers(self, num_layers):
        """Set the number of layers to use."""
        if num_layers is not None:
            num_layers -= 1
        self.use_layers = num_layers


class TransformerBlock(nn.Module):
    """Simplified transformer block."""
    def __init__(self, config):
        super().__init__()
        self.attn = SelfAttention(config)
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop)
        )
        
    def forward(self, x, attention_mask=None):
        # Self-attention
        a = self.attn(self.ln_1(x), attention_mask)
        x = x + a
        
        # Feed-forward network
        m = self.mlp(self.ln_2(x))
        x = x + m
        
        return x


class SelfAttention(nn.Module):
    """Simplified self-attention mechanism."""
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = nn.Dropout(config.attn_pdrop)
        
        # Ensure n_embd is divisible by n_head
        assert self.n_embd % self.n_head == 0
        
        # Create query, key, value projections
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        
    def _attn(self, q, k, v, attention_mask=None):
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (float(v.size(-1)) ** 0.5)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Convert 2D attention_mask to 4D for broadcasting
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -10000.0
            scores = scores + attention_mask
        
        # Apply softmax to get attention weights
        weights = nn.functional.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)
        
        # Apply attention weights to values
        output = torch.matmul(weights, v)
        
        return output
    
    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_shape)
    
    def split_heads(self, x):
        new_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, x, attention_mask=None):
        # Project query, key, value
        x = self.c_attn(x)
        q, k, v = x.chunk(3, dim=-1)
        
        # Split into multiple heads
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)
        
        # Apply self-attention
        a = self._attn(q, k, v, attention_mask)
        
        # Merge heads back together
        a = self.merge_heads(a)
        
        # Output projection
        a = self.c_proj(a)
        a = self.resid_dropout(a)
        
        return a
