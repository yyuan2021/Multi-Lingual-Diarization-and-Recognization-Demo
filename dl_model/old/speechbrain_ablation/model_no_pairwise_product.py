from dl_model.old.speechbrain_ablation.shared import TDNNPairStudent

MODEL_NAME = "no_pairwise_product"


def build_model(args):
    return TDNNPairStudent(
        sample_rate=args.sr,
        n_mels=args.n_mels,
        channels=tuple(args.student_channels),
        emb_dim=args.emb_dim,
        dropout=args.dropout,
        time_mask_max=args.time_mask_max,
        freq_mask_max=args.freq_mask_max,
        use_dilation=True,
        use_stats_pooling=True,
        use_pairwise_product=False,
        use_specaugment=True,
    )

