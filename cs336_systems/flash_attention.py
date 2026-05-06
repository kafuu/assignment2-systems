import torch
import math
import triton.language as tl
import triton

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    STORE_OUTPUT_AS_BF16: tl.constexpr, 
    ):
    """
    Q: Batch Nq(sequence) d_attn
    K: Batch Nk(sequence) d_attn
    V: Batch Nk(seq) d_v
    
    S = QKT/sqrt(D) : Batch Nq Nk
    O = softmax(S)*V : Batch Nq d_v
    L = sum(softmax(S))(按行) : Batch Nq
       
    """
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    ) #存储结果
    
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    ) #存储sum
    
    Q_tile = tl.load(Q_block_ptr,boundary_check=(0,1),padding_option="zero")
    
    Max = tl.full((Q_TILE_SIZE,),
                  -float("inf"), 
                  dtype=tl.float32)
    
    Sum = tl.zeros((Q_TILE_SIZE,),
                   dtype=tl.float32)

    Out_tile = tl.zeros((Q_TILE_SIZE,D),
                        dtype=tl.float32)
    
    for ks in range(tl.cdiv(N_KEYS,K_TILE_SIZE)):
        K_tile = tl.load(K_block_ptr,boundary_check=(0,1),padding_option="zero")
        V_tile = tl.load(V_block_ptr,boundary_check=(0,1),padding_option="zero")        
        S = tl.dot(Q_tile, tl.trans(K_tile)) * scale
        
        if IS_CAUSAL:
            Q_idx = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)[:,None]
            K_idx = ks * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)[None,:]
            S = tl.where(K_idx > Q_idx, tl.full(S.shape,-1e6, dtype= tl.float32), S)

        Max_new = tl.maximum(Max,tl.max(S,axis=-1))
        alpha = tl.exp(Max - Max_new)
        exped_dmax_S = tl.exp(S - Max_new[:,None])

        Sum = alpha * Sum + tl.sum(exped_dmax_S,axis=-1)
        Out_tile = alpha[:, None] * Out_tile + tl.dot(
            exped_dmax_S,
            V_tile.to(tl.float32),
        )
        Max = Max_new
        
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE,0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE,0))
        
    Out_tile = Out_tile / Sum[:, None]    
    logSumExp = tl.log(Sum) + Max
    
    if STORE_OUTPUT_AS_BF16:
        tl.store(
            O_block_ptr,
            value=Out_tile.to(tl.bfloat16),
            boundary_check=(0, 1),
        )
    else:
        tl.store(
            O_block_ptr,
            value=Out_tile,
            boundary_check=(0, 1),
        )
    tl.store(L_block_ptr,value=logSumExp,boundary_check=(0,))
        
def flash_backward1(Q, K, V, O, dO, L, block_size=16, causal=False):
    N, D = Q.shape
    scale = 1.0 / math.sqrt(D)

    dQ = torch.zeros_like(Q)
    dK = torch.zeros_like(K)
    dV = torch.zeros_like(V)

    # D vector: [B, H, N]
    Dvec = torch.sum(dO * O, dim=-1)

    for i in range(0, N, block_size):
        i_end = min(i + block_size, N)

        Qi = Q[ i:i_end, :]       # [B,H,Br,D]
        dOi = dO[ i:i_end, :]
        Li = L[ i:i_end]          # [B,H,Br]
        Di = Dvec[ i:i_end]       # [B,H,Br]

        dQi = torch.zeros_like(Qi)

        for j in range(0, N, block_size):
            j_end = min(j + block_size, N)

            Kj = K[j:j_end, :]   # [B,H,Bc,D]
            Vj = V[j:j_end, :]
            Bc = j_end - j

            # scores
            S = scale * torch.matmul(Qi, Kj.transpose(-2, -1))   # [B,H,Br,Bc]

            # causal mask
            if causal:
                q_idx = torch.arange(i, i_end, device=Q.device)
                k_idx = torch.arange(j, j_end, device=Q.device)
                mask = q_idx[:, None] >= k_idx[None, :]           # [Br,Bc]
                S = torch.where(mask, S, torch.full_like(S, float('-inf')))

            # reconstruct P from L
            P = torch.exp(S - Li[..., None])   # [B,H,Br,Bc]
            if causal:
                # exp(-inf) 本身就是 0，这句可不要
                P = torch.nan_to_num(P, nan=0.0)

            # dV += P^T @ dO
            dV[j:j_end, :] += torch.matmul(P.transpose(-2, -1), dOi)

            # dP = dO @ V^T
            dP = torch.matmul(dOi, Vj.transpose(-2, -1))   # [B,H,Br,Bc]

            # dS = P * (dP - D_i)
            dS = P * (dP - Di[..., :, None])

            # dQ += scale * dS @ K
            dQi += scale * torch.matmul(dS, Kj)

            # dK += scale * dS^T @ Q
            dK[j:j_end, :] += scale * torch.matmul(dS.transpose(-2, -1), Qi)

        dQ[i:i_end, :] = dQi

    return dQ, dK, dV
        
        
def flash_backward(Q, K, V, O, dO, L, block_size=16, causal=False):
    """
    PyTorch tiled backward for FlashAttention.

    Q, K, V, O, dO: [B, N, D]
    L: [B, N], logsumexp from forward
    """
    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()
    O = O.contiguous()
    dO = dO.contiguous()
    L = L.contiguous()

    B, Nq, D = Q.shape
    _, Nk, _ = K.shape
    scale = 1.0 / math.sqrt(D)

    # 用 fp32 累加更稳，最后再 cast 回输入 dtype
    dQ = torch.zeros_like(Q, dtype=torch.float32)
    dK = torch.zeros_like(K, dtype=torch.float32)
    dV = torch.zeros_like(V, dtype=torch.float32)

    Q_f = Q.float()
    K_f = K.float()
    V_f = V.float()
    O_f = O.float()
    dO_f = dO.float()
    L_f = L.float()

    # Delta_i = sum_j P_ij * dP_ij = dO_i dot O_i
    # shape: [B, Nq]
    Delta = torch.sum(dO_f * O_f, dim=-1)

    for i in range(0, Nq, block_size):
        i_end = min(i + block_size, Nq)

        Qi = Q_f[:, i:i_end, :]        # [B, Br, D]
        dOi = dO_f[:, i:i_end, :]      # [B, Br, D]
        Li = L_f[:, i:i_end]           # [B, Br]
        Deltai = Delta[:, i:i_end]     # [B, Br]

        dQi = torch.zeros_like(Qi, dtype=torch.float32)

        for j in range(0, Nk, block_size):
            j_end = min(j + block_size, Nk)

            Kj = K_f[:, j:j_end, :]    # [B, Bc, D]
            Vj = V_f[:, j:j_end, :]    # [B, Bc, D]

            # S_ij = Q_i K_j^T / sqrt(D)
            S = scale * torch.matmul(Qi, Kj.transpose(-2, -1))  # [B, Br, Bc]

            if causal:
                q_idx = torch.arange(i, i_end, device=Q.device)[:, None]
                k_idx = torch.arange(j, j_end, device=Q.device)[None, :]
                mask = k_idx <= q_idx
                S = torch.where(mask, S, torch.full_like(S, float("-inf")))

            # 由 forward 保存的 logsumexp L 重构 P
            # P_ij = exp(S_ij - logsumexp_i)
            P = torch.exp(S - Li[:, :, None])  # [B, Br, Bc]

            if causal:
                P = torch.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)

            # dV_j += P_ij^T dO_i
            dV[:, j:j_end, :] += torch.matmul(P.transpose(-2, -1), dOi)

            # dP_ij = dO_i V_j^T
            dP = torch.matmul(dOi, Vj.transpose(-2, -1))  # [B, Br, Bc]

            # softmax backward:
            # dS_ij = P_ij * (dP_ij - Delta_i)
            dS = P * (dP - Deltai[:, :, None])  # [B, Br, Bc]

            # dQ_i += scale * dS_ij K_j
            dQi += scale * torch.matmul(dS, Kj)

            # dK_j += scale * dS_ij^T Q_i
            dK[:, j:j_end, :] += scale * torch.matmul(dS.transpose(-2, -1), Qi)

        dQ[:, i:i_end, :] = dQi

    return dQ.to(Q.dtype), dK.to(K.dtype), dV.to(V.dtype)
    
@triton.jit
def flash_bwd_preprocess_kernel(
    O_ptr, dO_ptr, Delta_ptr,
    stride_ob, stride_oq, stride_od,
    stride_dob, stride_doq, stride_dod,
    stride_db, stride_dq,
    N_QUERIES: tl.constexpr,
    D: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    q_block = tl.program_id(0)
    b = tl.program_id(1)

    q_offsets = q_block * BLOCK_Q + tl.arange(0, BLOCK_Q)
    d_offsets = tl.arange(0, BLOCK_D)

    mask = (q_offsets[:, None] < N_QUERIES) & (d_offsets[None, :] < D)

    O = tl.load(
        O_ptr + b * stride_ob + q_offsets[:, None] * stride_oq + d_offsets[None, :] * stride_od,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    dO = tl.load(
        dO_ptr + b * stride_dob + q_offsets[:, None] * stride_doq + d_offsets[None, :] * stride_dod,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    delta = tl.sum(O * dO, axis=1)

    tl.store(
        Delta_ptr + b * stride_db + q_offsets * stride_dq,
        delta,
        mask=q_offsets < N_QUERIES,
    )

@triton.jit
def flash_bwd_dkdv_kernel(
    Q_ptr, K_ptr, V_ptr,
    dO_ptr, L_ptr, Delta_ptr,
    dK_ptr, dV_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_dob, stride_doq, stride_dod,
    stride_lb, stride_lq,
    stride_db, stride_dq,
    stride_dkb, stride_dkk, stride_dkd,
    stride_dvb, stride_dvk, stride_dvd,
    N_QUERIES: tl.constexpr,
    N_KEYS: tl.constexpr,
    scale,
    D: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    STORE_GRAD_AS_BF16: tl.constexpr,
):
    k_block = tl.program_id(0)
    b = tl.program_id(1)

    k_offsets = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    d_offsets = tl.arange(0, BLOCK_D)

    K = tl.load(
        K_ptr + b * stride_kb + k_offsets[:, None] * stride_kk + d_offsets[None, :] * stride_kd,
        mask=(k_offsets[:, None] < N_KEYS) & (d_offsets[None, :] < D),
        other=0.0,
    )

    V = tl.load(
        V_ptr + b * stride_vb + k_offsets[:, None] * stride_vk + d_offsets[None, :] * stride_vd,
        mask=(k_offsets[:, None] < N_KEYS) & (d_offsets[None, :] < D),
        other=0.0,
    )

    dK_acc = tl.zeros((BLOCK_K, BLOCK_D), dtype=tl.float32)
    dV_acc = tl.zeros((BLOCK_K, BLOCK_D), dtype=tl.float32)

    for q_start in range(0, N_QUERIES, BLOCK_Q):
        q_offsets = q_start + tl.arange(0, BLOCK_Q)

        Q = tl.load(
            Q_ptr + b * stride_qb + q_offsets[:, None] * stride_qq + d_offsets[None, :] * stride_qd,
            mask=(q_offsets[:, None] < N_QUERIES) & (d_offsets[None, :] < D),
            other=0.0,
        )

        dO = tl.load(
            dO_ptr + b * stride_dob + q_offsets[:, None] * stride_doq + d_offsets[None, :] * stride_dod,
            mask=(q_offsets[:, None] < N_QUERIES) & (d_offsets[None, :] < D),
            other=0.0,
        )

        L = tl.load(
            L_ptr + b * stride_lb + q_offsets * stride_lq,
            mask=q_offsets < N_QUERIES,
            other=0.0,
        ).to(tl.float32)

        Delta = tl.load(
            Delta_ptr + b * stride_db + q_offsets * stride_dq,
            mask=q_offsets < N_QUERIES,
            other=0.0,
        ).to(tl.float32)

        S = tl.dot(Q, tl.trans(K)) * scale

        valid_mask = (q_offsets[:, None] < N_QUERIES) & (k_offsets[None, :] < N_KEYS)

        if IS_CAUSAL:
            causal_mask = k_offsets[None, :] <= q_offsets[:, None]
            valid_mask = valid_mask & causal_mask

        S = tl.where(valid_mask, S, -float("inf"))

        P = tl.exp(S - L[:, None])

        dV_acc += tl.dot(
            tl.trans(P),
            dO.to(tl.float32),
        )

        dP = tl.dot(
            dO,
            tl.trans(V),
        )

        dS = P * (dP - Delta[:, None])

        dK_acc += tl.dot(
            tl.trans(dS),
            Q.to(tl.float32),
        ) * scale

    if STORE_GRAD_AS_BF16:
        tl.store(
            dK_ptr + b * stride_dkb + k_offsets[:, None] * stride_dkk + d_offsets[None, :] * stride_dkd,
            dK_acc.to(tl.bfloat16),
            mask=(k_offsets[:, None] < N_KEYS) & (d_offsets[None, :] < D),
        )

        tl.store(
            dV_ptr + b * stride_dvb + k_offsets[:, None] * stride_dvk + d_offsets[None, :] * stride_dvd,
            dV_acc.to(tl.bfloat16),
            mask=(k_offsets[:, None] < N_KEYS) & (d_offsets[None, :] < D),
        )
    else:
        tl.store(
            dK_ptr + b * stride_dkb + k_offsets[:, None] * stride_dkk + d_offsets[None, :] * stride_dkd,
            dK_acc,
            mask=(k_offsets[:, None] < N_KEYS) & (d_offsets[None, :] < D),
        )

        tl.store(
            dV_ptr + b * stride_dvb + k_offsets[:, None] * stride_dvk + d_offsets[None, :] * stride_dvd,
            dV_acc,
            mask=(k_offsets[:, None] < N_KEYS) & (d_offsets[None, :] < D),
        )

@triton.jit
def flash_bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr,
    dO_ptr, L_ptr, Delta_ptr,
    dQ_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_dob, stride_doq, stride_dod,
    stride_lb, stride_lq,
    stride_db, stride_dq,
    stride_dqb, stride_dqq, stride_dqd,
    N_QUERIES: tl.constexpr,
    N_KEYS: tl.constexpr,
    scale,
    D: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    STORE_GRAD_AS_BF16: tl.constexpr,
):
    q_block = tl.program_id(0)
    b = tl.program_id(1)

    q_offsets = q_block * BLOCK_Q + tl.arange(0, BLOCK_Q)
    d_offsets = tl.arange(0, BLOCK_D)

    Q = tl.load(
        Q_ptr + b * stride_qb + q_offsets[:, None] * stride_qq + d_offsets[None, :] * stride_qd,
        mask=(q_offsets[:, None] < N_QUERIES) & (d_offsets[None, :] < D),
        other=0.0,
    )

    dO = tl.load(
        dO_ptr + b * stride_dob + q_offsets[:, None] * stride_doq + d_offsets[None, :] * stride_dod,
        mask=(q_offsets[:, None] < N_QUERIES) & (d_offsets[None, :] < D),
        other=0.0,
    )

    L = tl.load(
        L_ptr + b * stride_lb + q_offsets * stride_lq,
        mask=q_offsets < N_QUERIES,
        other=0.0,
    ).to(tl.float32)

    Delta = tl.load(
        Delta_ptr + b * stride_db + q_offsets * stride_dq,
        mask=q_offsets < N_QUERIES,
        other=0.0,
    ).to(tl.float32)

    dQ_acc = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)

    for k_start in range(0, N_KEYS, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)

        K = tl.load(
            K_ptr + b * stride_kb + k_offsets[:, None] * stride_kk + d_offsets[None, :] * stride_kd,
            mask=(k_offsets[:, None] < N_KEYS) & (d_offsets[None, :] < D),
            other=0.0,
        )

        V = tl.load(
            V_ptr + b * stride_vb + k_offsets[:, None] * stride_vk + d_offsets[None, :] * stride_vd,
            mask=(k_offsets[:, None] < N_KEYS) & (d_offsets[None, :] < D),
            other=0.0,
        )

        S = tl.dot(Q, tl.trans(K)) * scale

        valid_mask = (q_offsets[:, None] < N_QUERIES) & (k_offsets[None, :] < N_KEYS)

        if IS_CAUSAL:
            causal_mask = k_offsets[None, :] <= q_offsets[:, None]
            valid_mask = valid_mask & causal_mask

        S = tl.where(valid_mask, S, -float("inf"))

        P = tl.exp(S - L[:, None])

        dP = tl.dot(
            dO,
            tl.trans(V),
        )

        dS = P * (dP - Delta[:, None])

        dQ_acc += tl.dot(
            dS,
            K.to(tl.float32),
        ) * scale

    if STORE_GRAD_AS_BF16:
        tl.store(
            dQ_ptr + b * stride_dqb + q_offsets[:, None] * stride_dqq + d_offsets[None, :] * stride_dqd,
            dQ_acc.to(tl.bfloat16),
            mask=(q_offsets[:, None] < N_QUERIES) & (d_offsets[None, :] < D),
        )
    else:
        tl.store(
            dQ_ptr + b * stride_dqb + q_offsets[:, None] * stride_dqq + d_offsets[None, :] * stride_dqd,
            dQ_acc,
            mask=(q_offsets[:, None] < N_QUERIES) & (d_offsets[None, :] < D),
        )
    
    
class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        Q = Q.contiguous()
        K = K.contiguous()
        V = V.contiguous()

        B, N_QUERIES, D = Q.shape
        _, N_KEYS, _ = K.shape

        Q_TILE_SIZE = 16
        K_TILE_SIZE = 16
        scale = 1.0 / math.sqrt(D)

        # 如果你当前 kernel 假设 Dv == D，这里先保持和原来一致
        O = torch.empty_like(Q)
        L = torch.empty((B, N_QUERIES), device=Q.device, dtype=torch.float32)

        grid = (triton.cdiv(N_QUERIES, Q_TILE_SIZE), B)

        flash_fwd_kernel[grid](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_QUERIES, N_KEYS,
            scale,
            D=D,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
            IS_CAUSAL=is_causal,
            STORE_OUTPUT_AS_BF16=(Q.dtype == torch.bfloat16),
        )

        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal

        return O

    @staticmethod
    def backward(ctx, dO):
        L, Q, K, V, O = ctx.saved_tensors

        dO = dO.contiguous()

        B, N_QUERIES, D = Q.shape
        _, N_KEYS, _ = K.shape

        assert V.shape[-1] == D, "当前 backward 版本假设 Dv == D"

        BLOCK_Q = 16
        BLOCK_K = 16
        BLOCK_D = 1 << (D - 1).bit_length()

        scale = 1.0 / math.sqrt(D)

        dQ = torch.empty_like(Q)
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)

        Delta = torch.empty(
            (B, N_QUERIES),
            device=Q.device,
            dtype=torch.float32,
        )

        grid_q = (triton.cdiv(N_QUERIES, BLOCK_Q), B)
        grid_k = (triton.cdiv(N_KEYS, BLOCK_K), B)

        flash_bwd_preprocess_kernel[grid_q](
            O, dO, Delta,
            O.stride(0), O.stride(1), O.stride(2),
            dO.stride(0), dO.stride(1), dO.stride(2),
            Delta.stride(0), Delta.stride(1),
            N_QUERIES=N_QUERIES,
            D=D,
            BLOCK_Q=BLOCK_Q,
            BLOCK_D=BLOCK_D,
        )

        flash_bwd_dkdv_kernel[grid_k](
            Q, K, V,
            dO, L, Delta,
            dK, dV,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            dO.stride(0), dO.stride(1), dO.stride(2),
            L.stride(0), L.stride(1),
            Delta.stride(0), Delta.stride(1),
            dK.stride(0), dK.stride(1), dK.stride(2),
            dV.stride(0), dV.stride(1), dV.stride(2),
            N_QUERIES=N_QUERIES,
            N_KEYS=N_KEYS,
            scale=scale,
            D=D,
            BLOCK_Q=BLOCK_Q,
            BLOCK_K=BLOCK_K,
            BLOCK_D=BLOCK_D,
            IS_CAUSAL=ctx.is_causal,
            STORE_GRAD_AS_BF16=(Q.dtype == torch.bfloat16),
        )

        flash_bwd_dq_kernel[grid_q](
            Q, K, V,
            dO, L, Delta,
            dQ,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            dO.stride(0), dO.stride(1), dO.stride(2),
            L.stride(0), L.stride(1),
            Delta.stride(0), Delta.stride(1),
            dQ.stride(0), dQ.stride(1), dQ.stride(2),
            N_QUERIES=N_QUERIES,
            N_KEYS=N_KEYS,
            scale=scale,
            D=D,
            BLOCK_Q=BLOCK_Q,
            BLOCK_K=BLOCK_K,
            BLOCK_D=BLOCK_D,
            IS_CAUSAL=ctx.is_causal,
            STORE_GRAD_AS_BF16=(Q.dtype == torch.bfloat16),
        )

        return dQ, dK, dV, None

class FlashAttnTorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """
        Torch版flash atten实现
        
        Q: Batch Nq(sequence) d_attn
        K: Batch Nk(sequence) d_attn
        V: Batch Nk(seq) d_v
        
        S = QKT/sqrt(D) : Batch Nq Nk
        O = softmax(S)*V : Batch Nq d_v
        L = sum(softmax(S))(按行) : Batch Nq
        
        tile:{
            Q:16 * d_attn
            K:16 * d_attn
        }
        
        Flash Attn流程：
            1. 取Q tile(16 * d_attn) 为 Qi
            3. M = Q tile * K tile (16 * 16)
            4. P = softmax(M) * V tile (16 * d_v)
            5. P = [(exp(M)/exp(max)) / (sum(exp)/exp(max))] * V tile
            6. 记max为m，记sum(exp)为l，记(exp(M)/exp(max)) * V tile为acc
                2. 取K tile(16 * d_attn) V tile(16 * d_v) 为 Kj , Vj
                7. 旧max要被新max取代，l与acc要重新缩放，缩放倍数为alpha = exp(m - m_new)
                8. l = alpha * l + 新的sum(exp(M-新的max))
                9. acc = alpha * acc + M * N tile
            
        """
        B, Nq, D = Q.shape
        _, Nk, _ = K.shape
        scale = 1.0 / math.sqrt(D)

        Bq = 16
        Bk = 16

        O = torch.empty((B, Nq, V.shape[-1]), device=Q.device, dtype=Q.dtype)
        L = torch.empty((B, Nq), device=Q.device, dtype=torch.float32)

        for b in range(B):
            for qs in range(0, Nq, Bq):
                Qi = Q[b, qs:qs+Bq].float()   # [bq, d]

                m = torch.full((Qi.shape[0],), float("-inf"), device=Q.device) # max: [bq]
                l = torch.zeros((Qi.shape[0],), device=Q.device) # sum: [bq]
                acc = torch.zeros((Qi.shape[0], V.shape[-1]), device=Q.device) # out tile: [bq, d_v]

                for ks in range(0, Nk, Bk):
                    Kj = K[b, ks:ks+Bk].float() # [bk, d]
                    Vj = V[b, ks:ks+Bk].float() # [bk, d_v]

                    S = Qi @ Kj.transpose(0, 1) * scale   # [bq, bk]

                    if is_causal:   
                        q_idx = torch.arange(qs, qs + Qi.shape[0], device=Q.device)[:, None] #[ bq]
                        k_idx = torch.arange(ks, ks + Kj.shape[0], device=Q.device)[None, :] # [bk]
                        S = torch.where(k_idx > q_idx, S.new_full((), -1e6), S) # mask: [bq, bk]
                        

                    m_new = torch.maximum(m, S.max(dim=-1).values) # [bq]
                    alpha = torch.exp(m - m_new) #换到新坐标系
                    p_tilde = torch.exp(S - m_new[:, None]) 

                    l = alpha * l + p_tilde.sum(dim=-1)
                    acc = alpha[:, None] * acc + p_tilde @ Vj
                    m = m_new

                O[b, qs:qs+Qi.shape[0]] = (acc / l[:, None]).to(Q.dtype)
                L[b, qs:qs+Qi.shape[0]] = m + torch.log(l)

        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO):
        L, Q, K, V, O = ctx.saved_tensors

        dQ, dK, dV = flash_backward(
            Q, K, V, O, dO, L,
            block_size=16,
            causal=ctx.is_causal
        )

        return dQ, dK, dV, None