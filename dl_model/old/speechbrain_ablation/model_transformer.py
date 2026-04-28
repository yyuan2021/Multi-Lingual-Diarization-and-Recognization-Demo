from dl_model.old.speechbrain_ablation.shared import TransformerPairStudent

MODEL_NAME = "transformer"


def build_model(args):
    return TransformerPairStudent(
        sample_rate=args.sr,
        n_mels=args.n_mels,
        emb_dim=args.emb_dim,
        dropout=args.dropout,
        time_mask_max=args.time_mask_max,
        freq_mask_max=args.freq_mask_max,
        use_specaugment=True,
    )

