import torch

from einops import rearrange
from torch import nn


class CausalSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    # key, value, query에 대한 선형변환 layer 초기화.
    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)

    # 이 드롭아웃은 트랜스포머의 원래 구현에 따라 normalized attention scores에 적용된다.
    # 다소 이례적이지만, 경험적으로 이것이 더 나은 성능을 제공한다고 알려져 있다.
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

  def transform(self, x, linear_layer):
    # hidden_state (x) 를 사영하기 위해 k, v, q의 해당 linear_layer가 사용된다.
    proj = linear_layer(x)
    # 다음으로, 프로젝션에 대해 여러 헤드를 생성해야 한다. 
    # 이는 은닉 상태를 self.num_attention_heads로 분할하며, 
    # 각 헤드는 self.attention_head_size 크기를 갖도록 한다.
    proj = rearrange(proj, 'b t (h d) -> b t h d', h=self.num_attention_heads)
    # 적절히 전치하여 크기 [bs, num_attention_heads, seq_len, attention_head_size]인 프로젝션을 얻는다.
    proj = rearrange(proj, 'b t h d -> b h t d')
    return proj

  def attention(self, key, query, value, attention_mask):
    """
    key, value: [bs, num_heads, key_len, head_dim]
      - prefix를 쓰면 key_len = prefix_len + seq_len
    query: [bs, num_heads, query_len, head_dim]
      - query_len = seq_len
    attention_mask: [bs, 1, 1, key_len]
      - prefix 위치는 0, padding 위치는 -10000
    """
    bs, num_heads, query_len, head_dim = query.size()
    key_len = key.size(2)
    prefix_len = key_len - query_len

    # 1. QK^T score 계산
    score = query @ key.transpose(-1, -2)
    score = score / (head_dim ** 0.5)

    # 2. causal mask 생성
    # prefix 부분은 모든 query token이 attend 가능해야 하므로 mask하지 않는다.
    causal_mask = torch.triu(
      torch.ones(query_len, query_len, device=score.device, dtype=torch.bool),
      diagonal=1,
    )

    if prefix_len > 0:
      prefix_mask = torch.zeros(
        query_len, prefix_len, device=score.device, dtype=torch.bool
      )
      causal_mask = torch.cat([prefix_mask, causal_mask], dim=-1)

    # causal_mask: [query_len, key_len]
    score = score.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), -10000.0)

    # 3. padding mask 적용
    score = score + attention_mask

    # 4. softmax + dropout
    attention_prob = torch.softmax(score, dim=-1)
    attention_prob = self.dropout(attention_prob)

    # 5. value에 attention 가중치 적용
    context = attention_prob @ value

    # 6. multi-head 다시 합치기: [bs, query_len, hidden_size]
    context = context.transpose(1, 2).contiguous()
    context = context.view(bs, query_len, num_heads * head_dim)

    return context
    

  def forward(self, hidden_states, attention_mask, prefix_key_value=None):
    """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    prefix_key_value:
      None 또는 (prefix_key, prefix_value)
      prefix_key/value: [bs, num_heads, prefix_len, head_dim]
    output: [bs, seq_len, hidden_state]
    """
    key_layer = self.transform(hidden_states, self.key)
    value_layer = self.transform(hidden_states, self.value)
    query_layer = self.transform(hidden_states, self.query)

    if prefix_key_value is not None:
      prefix_key, prefix_value = prefix_key_value

      # 기존 key/value 앞에 prefix key/value를 붙인다.
      key_layer = torch.cat([prefix_key, key_layer], dim=2)
      value_layer = torch.cat([prefix_value, value_layer], dim=2)

      # attention_mask도 prefix 길이만큼 앞에 0을 붙인다.
      # prefix token들은 padding이 아니므로 항상 attend 가능.
      bs = attention_mask.size(0)
      prefix_len = prefix_key.size(2)
      prefix_attention_mask = torch.zeros(
        bs, 1, 1, prefix_len,
        dtype=attention_mask.dtype,
        device=attention_mask.device,
      )
      attention_mask = torch.cat([prefix_attention_mask, attention_mask], dim=-1)

    attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
    return attn_value
