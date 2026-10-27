"""One batched SDPA over bounded selected K/V, without stacking full histories."""
import torch
import torch.nn.functional as F


def attend_rows(q, rows, selections, scale):
    batch, _, hq, d = q.shape
    hkv = rows[0].keys.shape[1]
    width = max(ids.shape[-1] for ids, valid in selections)
    keys, values, masks = [], [], []
    heads = torch.arange(hkv, device=q.device)[:, None]
    for row, (ids, valid) in zip(rows, selections):
        safe = torch.where(valid, ids, 0)
        if hasattr(row, 'layout'):
            safe = row.layout.map(safe)
        k = row.keys[0][heads, safe].masked_fill(~valid[..., None], 0)
        v = row.values[0][heads, safe].masked_fill(~valid[..., None], 0)
        pad = width-ids.shape[-1]
        keys.append(F.pad(k, (0,0,0,pad)))
        values.append(F.pad(v, (0,0,0,pad)))
        masks.append(F.pad(valid, (0,pad)))
    # The batch dimension is an independent attention axis. This never pads
    # KV history to the longest prompt: only the selected O(B*K*D) values.
    output = F.scaled_dot_product_attention(
        q[:, 0].reshape(batch, hkv, hq//hkv, d), torch.stack(keys), torch.stack(values),
        attn_mask=torch.stack(masks)[:, :, None], scale=scale, dropout_p=0.0)
    return output.reshape(batch, 1, hq, d)
