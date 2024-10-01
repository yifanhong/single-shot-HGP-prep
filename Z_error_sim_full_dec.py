import numpy as np
from scipy.sparse import csc_matrix
from ldpc import bposd_decoder

def get_BPOSD_failures(code, par, p, iters, z):
    # par = [bp_iters, osd_sweeps]
    n = code.N
    m = len(code.hx)
    
    # Construct bulk 3D Tanner graph
    H_dec = np.kron(np.identity(z,dtype=int), code.hx)
    H_dec = np.concatenate((H_dec,np.zeros([m*z,m*z],dtype=int)), axis=1)
    for j in range(m*z):
        H_dec[j,n*z+j] = 1
        if m+j < m*z:
            H_dec[m+j,n*z+j] = 1

    # Sparsify matrices
    H_dec = csc_matrix(H_dec)
    hx = csc_matrix(code.hx)
    
    bposd_bulk = bposd_decoder(H_dec, error_rate=p, max_iter=par[0], bp_method='ms', osd_method='osd_cs', osd_order=par[1])
    bposd_hx = bposd_decoder(hx, error_rate=p, max_iter=par[0], bp_method='ms', osd_method='osd_cs', osd_order=par[1])
    
    failures = 0
    for i in range(iters):        
        init_error = (np.random.rand(H_dec.shape[1]) < p).astype(int)
        init_syndrome = H_dec @ init_error % 2
        bposd_bulk.decode(init_syndrome)
        residual_error = bposd_bulk.osdw_decoding ^ init_error
        boundary_error = np.sum(np.reshape(residual_error[:n*z], [z,n]), axis=0) % 2
        final_error = (np.random.rand(n) < p).astype(int) ^ boundary_error
        final_syndrome = hx @ final_error % 2
        bposd_hx.decode(final_syndrome)
        final_state = bposd_hx.osdw_decoding ^ final_error
        
        if (code.lx@final_state%2).any():
            failures += 1

    return failures