def speck_bits_encrypt(P, K, nr):
    # P: (N, 32) bits -> x,y uint16
    x = np.packbits(P[:, :16], axis=1, bitorder='big').view(np.uint16).byteswap().flatten()
    y = np.packbits(P[:, 16:], axis=1, bitorder='big').view(np.uint16).byteswap().flatten()
    # K: (N, 64) bits -> 4 words uint16 (k0..k3)
    kwords = [np.packbits(K[:, i*16:(i+1)*16], axis=1, bitorder='big')
                .view(np.uint16).byteswap().flatten() for i in range(4)]
    ks = expand_keys(np.array(kwords), nr)
    cx, cy = encryption((x, y), ks)
    # convert cx,cy (uint16) trở lại bit array (N,32)
    ...
    return C_bits