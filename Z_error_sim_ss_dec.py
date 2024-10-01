import numpy as np
from scipy.sparse import csc_matrix
from ldpc import bposd_decoder

def get_BPOSD_failures(code, par, p, iters, z):
    # par = [bp_iters, osd_sweeps]
    n = code.N
    m = len(code.hx)

    # Construct single-shot Tanner graph
    H_ss = np.concatenate((code.hx,np.identity(m,dtype=int)), axis=1).astype(int)

    # Sparsify matrices
    H_ss = csc_matrix(H_ss)
    hx = csc_matrix(code.hx)
    
    bposd_ss = bposd_decoder(H_ss, error_rate=p, max_iter=par[0], bp_method='ms', osd_method='osd_cs', osd_order=par[1])
    bposd_hx = bposd_decoder(hx, error_rate=p, max_iter=par[0], bp_method='ms', osd_method='osd_cs', osd_order=par[1])
    
    failures = 0
    for i in range(iters):        
        residual_error = np.zeros(n+m, dtype=int)
        for t in range(z):
            error = residual_error ^ (np.random.rand(n+m)<p).astype(int)
            syndrome = H_ss @ error % 2
            bposd_ss.decode(syndrome)
            residual_error[:n] = bposd_ss.osdw_decoding[:n] ^ error[:n]

        final_error = (np.random.rand(n) < p).astype(int) ^ residual_error[:n]
        final_syndrome = hx @ final_error % 2
        bposd_hx.decode(final_syndrome)
        final_state = bposd_hx.osdw_decoding ^ final_error
        
        if (code.lx@final_state%2).any():
            failures += 1

    return failures