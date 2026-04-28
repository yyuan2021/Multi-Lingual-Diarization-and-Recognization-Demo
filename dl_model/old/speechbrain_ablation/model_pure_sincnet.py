# Pure SincNet without TDNN backend.
from dl_model.old.speechbrain_ablation.shared import SincNetPairStudent

MODEL_NAME = "pure_sincnet"


def build_model(args):
    return SincNetPairStudent(
        sample_rate=args.sr,
        emb_dim=args.emb_dim,
        dropout=args.dropout,
        sinc_channels=64,
    )
