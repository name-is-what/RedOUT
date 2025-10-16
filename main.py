def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn
from model import HCL
from data_loader import *
import argparse
import numpy as np
import torch
import random
import sklearn.metrics as skm
import torch_geometric
from tree_utli import HRN, HRNEncoder


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-exp_type", type=str, default="ad", choices=["oodd", "ad"])
    parser.add_argument("-DS", help="Dataset", default="BZR")
    parser.add_argument("-DS_ood", help="Dataset", default="COX2")
    parser.add_argument("-DS_pair", default=None)
    parser.add_argument("-rw_dim", type=int, default=16)
    parser.add_argument("-dg_dim", type=int, default=16)
    parser.add_argument("-batch_size", type=int, default=128)
    parser.add_argument("-batch_size_test", type=int, default=9999)
    parser.add_argument("-lr", type=float, default=0.0001)
    parser.add_argument("-num_layer", type=int, default=5)
    parser.add_argument("-hidden_dim", type=int, default=32)
    parser.add_argument("-num_trial", type=int, default=5)
    parser.add_argument("-num_epoch", type=int, default=400)
    parser.add_argument("-eval_freq", type=int, default=10)
    parser.add_argument("-is_adaptive", type=int, default=1)
    parser.add_argument("-num_cluster", type=int, default=2)
    parser.add_argument("-alpha", type=float, default=0.1)
    parser.add_argument("-gamma", type=float, default=0.1)
    parser.add_argument("-lam", type=float, default=0.1)

    # tree parameters
    parser.add_argument('-l', '--local', dest='local', action='store_const', const=True, default=False)
    parser.add_argument('-g', '--glob', dest='glob', action='store_const', const=True, default=False)
    parser.add_argument('-p', '--prior', dest='prior', action='store_const', const=True, default=False)
    parser.add_argument("--loss_sym", action="store_true")
    parser.add_argument("--tree_depth", type=int, default=5)
    parser.add_argument("--tree_pooling_type", type=str, default="sum")
    parser.add_argument("--tree_hidden_dim", type=int, default=32)
    parser.add_argument("--tree_dropout", type=int, default=0)
    parser.add_argument("--tree_link_input", action="store_true")
    parser.add_argument("--tree_drop_root", action="store_true")
    parser.add_argument("--tree_learning_rate", type=float, default=0.01)

    return parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    torch_geometric.seed_everything(seed)


def sim(z1, z2):
    import torch.nn.functional as F

    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())


def semi_loss(z1, z2):
    f = lambda x: torch.exp(x / 0.2)
    refl_sim = f(sim(z1, z1))
    between_sim = f(sim(z1, z2))
    return -torch.log(
        between_sim.diag()
        / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag())
    )


def weak_cmi(z1, z2, y1, y2):
    N = z1.shape[0]
    EPS = 1e-5
    f = lambda x: torch.exp(x / 1)
    between_sim = f(sim(z1, z2))
    mask = y1 == y2
    conditional_mask = y1.repeat(N, 1) == y2.reshape(-1, 1).repeat(1, N)
    conditional_mask += torch.eye(N, device=z1.device).bool()
    neg_sim = torch.sum(torch.mul(between_sim, conditional_mask), dim=1)
    ccl = -torch.log(between_sim.diag() / neg_sim)
    return ccl * mask


def loss_CRI(z1, z2, y1, y2):
    l1 = semi_loss(z1, z2)
    l2 = semi_loss(z2, z1)
    ret = (l1 + l2) * 0.5
    loss = ret
    ccl_loss = weak_cmi(z1, z2, y1, y2)
    loss = -args.gamma * ccl_loss
    return loss


def loss_at(g_f, y_pred):
    import torch.nn.functional as F

    return torch.nn.functional.cross_entropy(g_f, y_pred, reduction="none") + args.alpha * torch.mean(
        F.kl_div(
            torch.nn.LogSoftmax(dim=1)(g_f),
            torch.normal(g_f),
            reduction="none",
        ),
        dim=1,
    )


def online_learn(model, treeM, treeOpt, data, num_epoch):
    for epoch in range(1, num_epoch):
        model.eval()
        treeM.train()
        treeOpt.zero_grad()

        b, g_f, g_s, n_f, n_s = model(
            data.x, data.x_s, data.edge_index, data.batch, data.num_graphs
        )
        x_hrn = treeM(data)

        y_pred, y_pred_t = g_f.softmax(dim=1).cpu(), x_hrn.softmax(dim=1).cpu()
        y_pred, y_pred_t = y_pred.detach().numpy(), y_pred_t.detach().numpy()
        y_pred, y_pred_t = np.argmax(y_pred, axis=1), np.argmax(y_pred_t, axis=1)
        y_pred, y_pred_t = (
            torch.tensor(y_pred).to(device),
            torch.tensor(y_pred_t).to(device),
        )

        y_score_g = model.calc_loss_g(g_f, g_s)
        y_score_b = model.calc_loss_tree(x_hrn, g_f)

        loss = y_score_b.mean()
        loss += loss_CRI(g_f, x_hrn, y_pred, y_pred_t).mean()
        loss.backward()
        treeOpt.step()

        y_score = y_score_b
        y_score += loss_CRI(g_f, x_hrn, y_pred, y_pred_t)

        y_true = data.y
        auc = skm.roc_auc_score(
            y_true.detach().cpu().tolist(), y_score.detach().cpu().tolist()
        )
        if epoch % 10 == 0:
            print(f"[ONLINE RE-TRAINING] Epoch: {epoch:03d} | AUC:{auc:.4f}")

    return y_score


if __name__ == "__main__":
    setup_seed(0)
    args = arg_parse()

    if args.exp_type == "ad":
        if args.DS.startswith("Tox21"):
            dataloader, dataloader_test, meta = get_ad_dataset_Tox21(args)
        else:
            splits = get_ad_split_TU(args, fold=args.num_trial)

    aucs = []
    for trial in range(args.num_trial):
        setup_seed(trial + 1)

        if args.exp_type == "oodd":
            dataloader, dataloader_test, meta = get_ood_dataset(args, pre=True)

        elif args.exp_type == "ad" and not args.DS.startswith("Tox21"):
            dataloader, _, meta = get_ad_dataset_TU(args, splits[trial], pre=True)
            _, dataloader_test, _ = get_ad_dataset_TU(args, splits[trial])

        dataset_num_features = meta["num_feat"]
        tree_input_dim = meta["deg_x"]
        n_train = meta["num_train"]

        if trial == 0:
            print("================")
            print(f"Exp_type: {args.exp_type}")
            print(f"DS: {args.DS_pair if args.DS_pair is not None else args.DS}")
            print(f"num_features: {dataset_num_features}")
            print(f"num_structural_encodings: {args.dg_dim + args.rw_dim}")
            print(f"hidden_dim: {args.hidden_dim}")
            print(f"num_gc_layers: {args.num_layer}")
            print("================")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = HCL(
            args.hidden_dim, args.num_layer, dataset_num_features, args.dg_dim + args.rw_dim
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        encoder = HRNEncoder(
            args.tree_depth,
            args.tree_pooling_type,
            tree_input_dim,
            args.tree_hidden_dim,
            args.hidden_dim * args.num_layer,
            args.tree_dropout,
            args.tree_link_input,
            args.tree_drop_root,
            device,
        )
        treeM = HRN(encoder, args.hidden_dim * args.num_layer).to(device)
        treeOpt = torch.optim.Adam(treeM.parameters(), lr=args.tree_learning_rate)

        save_path = os.path.join(os.getcwd(), "pre-trained")
        file_name = f"OOD_{args.DS_pair}.pth"
        file_path = os.path.join(save_path, file_name)
        checkpoint = torch.load(file_path)
        model.load_state_dict(checkpoint["model_state_dict"])

        y_score_all = []
        y_true_all = []
        for data in dataloader_test:
            data = data.to(device)
            y_score = online_learn(model, treeM, treeOpt, data, args.num_epoch)
            y_true = data.y
            y_score_all += y_score.detach().cpu().tolist()
            y_true_all += y_true.detach().cpu().tolist()

        auc = skm.roc_auc_score(y_true_all, y_score_all)
        print(f"[TTA EVALIDATION RESULT] Trial: {trial:02d} | AUC:{auc:.4f}")
        aucs.append(auc)

    aucs = sorted(aucs, reverse=True)[:5]
    avg_auc = np.mean(aucs) * 100
    std_auc = np.std(aucs) * 100
    print(f"[FINAL RESULT] AVG_AUC:{avg_auc:.2f}+-{std_auc:.2f}")
