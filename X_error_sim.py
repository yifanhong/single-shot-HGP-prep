import numpy as np
from scipy.sparse import csc_matrix
from ldpc import bposd_decoder
from ldpc.codes import rep_code

def get_BPOSD_failures(code, par, p, q, iters, z):
    # par = [bp_iters, osd_sweeps]
    n = code.N
    m = len(code.hz)
    
    # Construct '3'-dimensional code with metachecks
    H_rep = rep_code(z)
    row1 = np.concatenate((np.kron(code.hz,np.identity(z,dtype=int)), np.zeros([m*z,m*(z-1)],dtype=int)), axis=1)
    row2 = np.concatenate((np.kron(np.identity(n,dtype=int),H_rep), np.kron(code.hx.T,np.identity(z-1,dtype=int))), axis=1)
    Hz = np.concatenate((row1,row2), axis=0)
    Mz = np.concatenate((np.kron(np.identity(m,dtype=int),H_rep), np.kron(code.hz,np.identity(z-1,dtype=int))), axis=1)

    # Sparsify matrices
    hz = csc_matrix(code.hz)
    Hz = csc_matrix(Hz)
    Mz = csc_matrix(Mz)
    
    # Define decoders
    if z > 1:
        bposd_Mz = bposd_decoder(Mz, error_rate=q, max_iter=par[0], bp_method='ms', osd_method='osd_cs', osd_order=par[1])
    bposd_Hz = bposd_decoder(Hz, error_rate=p, max_iter = par[0], bp_method = 'ms', osd_method = 'osd_cs', osd_order = par[1])
    bposd_hz = bposd_decoder(hz, error_rate=p, max_iter = par[0], bp_method = 'ms', osd_method = 'osd_cs', osd_order = par[1])
    
    failures = 0
    for i in range(iters):
        init_syndrome_err = (np.random.rand(Hz.shape[0]) < q).astype(int)
        if z > 1:
            init_metasyndrome = Mz @ init_syndrome_err % 2
            bposd_Mz.decode(init_metasyndrome)
            repaired_syndrome = bposd_Mz.osdw_decoding ^ init_syndrome_err
        else:
            repaired_syndrome = init_syndrome_err
        bposd_Hz.decode(repaired_syndrome)
        init_correction = bposd_Hz.osdw_decoding
        boundary_correction = init_correction[:n*z:z]
        final_error = (np.random.rand(n) < p).astype(int) ^ boundary_correction
        final_syndrome = hz @ final_error % 2
        bposd_hz.decode(final_syndrome)
        final_state = bposd_hz.osdw_decoding ^ final_error
        
        if (code.lz@final_state%2).any():
            failures += 1

    return failures