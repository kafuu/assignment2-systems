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
    
    for _ in range(tl.cdiv(N_KEYS,K_TILE_SIZE)):
        K_tile = tl.load(K_block_ptr,boundary_check=(0,1),padding_option="zero")
        V_tile = tl.load(V_block_ptr,boundary_check=(0,1),padding_option="zero")        
        S:tl.tensor = Q_tile @ tl.trans(K_tile,(0,1)) * scale

        Max_new = tl.maximum(Max,S.max(axis=-1))
        alpha = tl.exp(Max - Max_new)
        exped_dmax_S = tl.exp(S - Max_new[:,None])

        Sum = alpha * Sum + exped_dmax_S.sum(axis=-1)
        Out_tile = alpha[:,None] * Out_tile + exped_dmax_S @ V_tile
        Max = Max_new
        
        K_block_ptr.advance((K_TILE_SIZE,0))
        V_block_ptr.advance((K_TILE_SIZE,0))
        
    Out_tile = Out_tile / Sum[:, None]    
    logSumExp = tl.log(Sum) + Max
    
    tl.store(O_block_ptr,value=Out_tile,boundary_check=(0,1))
    tl.store(L_block_ptr,value=logSumExp,boundary_check=(0,))
        
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
        )

        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal

        return O

    @staticmethod
    def backward(ctx, dO):
        raise NotImplementedError

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
        raise NotImplementedError