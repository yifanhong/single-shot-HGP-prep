import numpy as np
from ldpc import bposd_decoder
from scipy.sparse import csc_matrix

def get_BPOSD_failures(code, par, p, q, iters, cycles):
    # par = [bp_iters, osd_sweeps]
    n = code.N
    m = len(code.hz)
    
    # Construct decoding (difference) graph
    if cycles > 0:
        H = np.kron(np.eye(cycles,dtype=int), code.hz)
        H = np.concatenate((H,np.zeros([m*cycles,m*(cycles+1)],dtype=int)), axis=1)
        for j in range(m*cycles):
            H[j,n*cycles+j] = 1
            H[j,n*cycles+m+j] = 1
        
        H = csc_matrix(H)
    Hz = csc_matrix(code.hz)
    
    meas_err_probs = np.ones(m) * q
    phys_err_probs = np.ones(n) * p
    
    if cycles > 0:
        channel_probs= np.concatenate((np.tile(phys_err_probs,cycles), np.tile(meas_err_probs,cycles+1)))
        bpd = bposd_decoder(H, channel_probs = channel_probs, max_iter = par[0], bp_method = 'ms', osd_method = 'osd_cs', osd_order = par[1])
    bpd_clean = bposd_decoder(Hz, channel_probs = phys_err_probs, max_iter = par[0], bp_method = 'ms', osd_method = 'osd_cs', osd_order = par[1])
    
    failures = 0
    for i in range(iters):
        init_syndrome_error = (np.random.rand(m) < q).astype(int)
        if cycles > 0:
            phys_errors = (np.random.rand(cycles,n) < p).astype(int)
            phys_errors_cum = np.cumsum(phys_errors, axis=0) % 2
            syndrome = (Hz @ phys_errors_cum.T).T % 2
            syndrome_error = (np.random.rand(cycles,m) < q).astype(int)
            noisy_syndrome = syndrome ^ syndrome_error
            noisy_syndrome[1:,:] = noisy_syndrome[1:,:] ^ noisy_syndrome[:-1,:]     # Convert to difference syndrome
            noisy_syndrome[0] = noisy_syndrome[0] ^ init_syndrome_error
            bpd.decode(np.ravel(noisy_syndrome))
            bpd_output = np.reshape(bpd.osdw_decoding[n*cycles:], [cycles+1,m])
            init_repaired_syndrome = bpd_output[0] ^ init_syndrome_error
        else:
            init_repaired_syndrome = init_syndrome_error

        bpd_clean.decode(init_repaired_syndrome)
        # Perform QEC with clean syndromes and check if logical error happened
        final_error = bpd_clean.osdw_decoding ^ (np.random.rand(n) < p).astype(int)
        final_syndrome = Hz @ final_error % 2
        bpd_clean.decode(final_syndrome)
        final_state = bpd_clean.osdw_decoding ^ final_error
        if (code.lz@final_state%2).any():
            failures += 1
                
    return failures