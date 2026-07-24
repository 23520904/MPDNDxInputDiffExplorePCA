def make_train_data(n, nr, a=POS_DELTAS, b=NEG_DELTAS, related_key=False, feature_mode='full', use_gpu=True, to_float32=True):
    """Fixed-key / Polytopic Quadruple data generator utility."""
    # Use a safe batch size for data generation to avoid OOM
    gen_batch = min(n, 100000)
    
    generator = PolytopicQuadrupleGenerator(
        encryption_function=encrypt_wrapper,
        plain_bits=32,
        key_bits=64,
        nr=nr,
        pos_diffs=a,
        neg_diffs=b,
        related_key=related_key,
        feature_mode=feature_mode,
        n_samples=n,
        batch_size=gen_batch,
        use_gpu=use_gpu,
        to_float32=to_float32
    )
    return generator[0]