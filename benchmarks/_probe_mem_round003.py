"""Probe: _Subspace GPU diag 各 n_str 的 t1 缓冲区尺寸, 验证 dim=5e5 OOM 机制。

theory §5 风险 2 预测 12,12 峰值 ~2GB; 实测 dim=5e5 GPU diag 371.5s (疑似 OOM→CPU 回退)。
本 probe 直接计算 sigma_selected_ci_gpu 内部各 t1 缓冲区的 numpy 尺寸 (不跑 GPU),
确认是否超 17GB RTX 5080 显存。
"""
import numpy as np
from pyscf.fci import cistring, selected_ci
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

norb, nelec = 12, (6, 6)
full = np.array(cistring.make_strings(range(norb), nelec[0]), dtype=np.int64)
nn = norb * (norb - 1) // 2
npair = norb * (norb + 1) // 2
print(f"norb={norb} nelec={nelec} nn={nn} npair={npair}")

for n_str in [100, 224, 317, 400, 708, 908]:
    sa = full[:n_str]
    dd_a = selected_ci.des_des_linkstr(sa, norb, nelec[0], True)
    dd_b = selected_ci.des_des_linkstr(sa, norb, nelec[1], True)
    cd_a = selected_ci.cre_des_linkstr(sa, norb, nelec[0], True)
    cd_b = selected_ci.cre_des_linkstr(sa, norb, nelec[1], True)
    nb = len(sa)
    # mask 非零连接数 (与 _links_tril 一致)
    def nconn(link_index):
        f = link_index.reshape(-1, 4)
        return int((f[:, 3] != 0).sum())
    nca_aa = nconn(dd_a)
    nca_bb = nconn(dd_b)
    nca_ba = nconn(cd_a)
    nca_bb2 = nconn(cd_b)
    t1_aaaa = nca_aa * nn * nb * 8 / 1e9
    t1_bbaa = len(sa) * npair * nb * 8 / 1e9
    total = t1_aaaa + 2 * t1_bbaa  # aaaa alpha + beta 各一个 t1, bbaa 一个 t1
    print(f"n_str={n_str:4d} dim={len(sa)*nb:8d} "
          f"nconn dd_a={nca_aa:9d} dd_b={nca_bb:9d} cd_a={nca_ba:9d} cd_b={nca_bb2:9d} "
          f"| t1_aaaa={t1_aaaa:8.2f}GB t1_bbaa={t1_bbaa:6.3f}GB peak~={total:8.2f}GB")
