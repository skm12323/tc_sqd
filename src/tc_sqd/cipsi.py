"""CIPSI refinement --- Configuration Interaction by Perturbative Selection.

从用户提供的**种子 det 集合**（如 UCJ 辅助采样 ``ucj_assisted_configurations``
或 S+D 激发）出发，迭代做 PT2 筛选的生成集扩展，将子空间自动补全到近 FCI 精度。

定位（方向 B）：
    UCJ-SQD 用少量采样 shots 达化学精度；若需要更高精度（FCI 级），CIPSI 从
    UCJ 种子出发只需 1-2 轮即补全到全空间（种子字符串已覆盖全空间大部分）。
    代价是对角化维度 = 字符串乘积（可达全空间），与 HCI 近全空间相当——CIPSI
    是**高精度 refine 层**，不是"少量 det"路线。

算法（每轮）：
    1. 当前子空间对角化（复用 solve_sci 的稳健路径: dim≤1000 numpy eigh, 否则 eigsh）
    2. 取 |c| > ``dom_thresh`` 的主导 dets，枚举单/双激发连接 -> 候选 dets
    3. 扩展空间上 ``contract_2e`` 一次得 <a|H|Psi>（pyscf 矩阵元，免手写符号问题）
    4. PT2_a = <a|H|Psi>^2 / (E_gs - E_a)，按 |PT2| 加入 top 候选
    5. 重复至空间达全空间 / PT2 < ``pt2_floor`` / 无新候选 / ``max_iter``

空间表示与 det 计数口径：
    本实现沿用 SQD 库的字符串乘积表示（对角化维度 = n_str_a × n_str_b）。
    闭壳层 (n_a==n_b) 时 α/β 合并为同一字符串集合（与
    ``bitstring_matrix_to_ci_strs`` 默认一致）；开壳层用独立 α/β 集合。
    注意"采样 det 数"（bsm 行）远小于对角化维度——两者口径不同。
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np
from pyscf.fci import cistring, selected_ci, direct_spin1
from pyscf import ao2mo
from scipy.sparse.linalg import eigsh, LinearOperator

from .configuration_recovery import recover_configurations
from .fermion import bitstring_matrix_to_ci_strs, SCIState
from .diagnostics import extrapolate_energy_variance, extrapolate_ev_pt2
from .tail_sampling import discover_tail_pool

__all__ = ["solve_cipsi", "solve_sqd_active", "solve_sqd_adaptive", "solve_hci",
           "solve_sqd_ev", "solve_sqd_distill", "eigenvector_importance_sample"]


# --------------------------------------------------------------------------- #
#  连接枚举 (Slater 行列式对之间的单/双激发目标)
# --------------------------------------------------------------------------- #
def _occ_bits(s: int, norb: int) -> list:
    return [i for i in range(norb) if (s >> i) & 1]


def _excited_dets(a: int, b: int, norb: int):
    """det (α=a, β=b) 的所有单/双激发目标 det 集合 {(α', β')}。"""
    oa, ob = _occ_bits(a, norb), _occ_bits(b, norb)
    va = [v for v in range(norb) if not (a >> v) & 1]
    vb = [v for v in range(norb) if not (b >> v) & 1]
    out = set()
    # 单激发 (α / β)
    for i in oa:
        for v in va:
            out.add((a ^ (1 << i) ^ (1 << v), b))
    for i in ob:
        for v in vb:
            out.add((a, b ^ (1 << i) ^ (1 << v)))
    # 双激发 αα / ββ
    for p in range(len(oa)):
        for q in range(p + 1, len(oa)):
            i, j = oa[p], oa[q]
            for r in range(len(va)):
                for s in range(r + 1, len(va)):
                    u, v = va[r], va[s]
                    out.add((a ^ (1 << i) ^ (1 << j) ^ (1 << u) ^ (1 << v), b))
    for p in range(len(ob)):
        for q in range(p + 1, len(ob)):
            i, j = ob[p], ob[q]
            for r in range(len(vb)):
                for s in range(r + 1, len(vb)):
                    u, v = vb[r], vb[s]
                    out.add((a, b ^ (1 << i) ^ (1 << j) ^ (1 << u) ^ (1 << v)))
    # 双激发 αβ
    for i in oa:
        for u in va:
            for j in ob:
                for v in vb:
                    out.add((a ^ (1 << i) ^ (1 << u), b ^ (1 << j) ^ (1 << v)))
    return out


def _single_excited_strings(s: int, norb: int):
    """字符串 s 的所有单激发目标字符串集合 (occ→virt, 激发阶 +1)。

    与 :func:`_excited_dets` 的差别: 只做单激发、且是**字符串级** (单个自旋扇区),
    供 round_008 三激发定向注入对全部已选字符串迭代爬激发阶用
    (单激发图连通 ⇒ 迭代到 fixpoint = 全空间)。
    """
    occ = [i for i in range(norb) if (s >> i) & 1]
    virt = [v for v in range(norb) if not (s >> v) & 1]
    out = set()
    for i in occ:
        for v in virt:
            out.add(s ^ (1 << i) ^ (1 << v))
    return out


# --------------------------------------------------------------------------- #
#  warm-start v0 构造 (round_010): 旧解态投影到新子空间作 ARPACK 初始向量
# --------------------------------------------------------------------------- #
def _project_v0(sa_old, sb_old, c2d_old, sa_new, sb_new):
    """旧解态投影到新子空间作 eigsh 的 ``v0`` (round_010 迭代减少)。

    solve_sqd_active 的收敛循环里子空间**单调增长** (旧字符串 ⊂ 新字符串),
    上一轮解态投影到新子空间 (旧字符串索引映射 + 新字符串振幅**置零** +
    归一化) 与新基态余弦 ≥ √(1−w_new) ≈ 0.99+, 是高质量 Krylov 初猜。

    - prune 收缩等场景旧串可能已不在新集合 -> searchsorted 越界/错配必须
      mask (交集投影, 正常轮次 mask 全 True 零成本)。
    - 新字符串振幅**置零** (不均匀小值): 第一次 matvec H·v0 自然生成新 det
      方向的修正分量; 均匀小值反而注入随机噪声压低余弦。
    - 全零 (‖P·c_old‖ < 1e-300) 返回 None -> 调用方不传 v0 (随机, 现状行为)。

    返回展平归一化 float64 向量 (长度 nA_new × nB_new) 或 None。O(dim)。
    """
    sa_old = np.asarray(sa_old, dtype=np.int64)
    sb_old = np.asarray(sb_old, dtype=np.int64)
    c2d_old = np.asarray(c2d_old, dtype=np.float64)
    ia = np.searchsorted(sa_new, sa_old)          # 旧字符串在新数组中的行号候选
    ib = np.searchsorted(sb_new, sb_old)
    # 交集 mask: 越界或 searchsorted 停在插入点 (值不匹配) 的都剔除
    ma = (ia < len(sa_new)) & (sa_new[np.minimum(ia, len(sa_new) - 1)] == sa_old)
    mb = (ib < len(sb_new)) & (sb_new[np.minimum(ib, len(sb_new) - 1)] == sb_old)
    v0 = np.zeros((len(sa_new), len(sb_new)))
    v0[np.ix_(ia[ma], ib[mb])] = c2d_old[np.ix_(np.where(ma)[0], np.where(mb)[0])]
    n = float(np.linalg.norm(v0))
    return None if n < 1e-300 else (v0 / n).ravel()


# --------------------------------------------------------------------------- #
#  子空间对角化 (与 solve_sci 相同的稳健路径)
# --------------------------------------------------------------------------- #
class _Subspace:
    """字符串集合 (α, β) 的子空间对角化, 提供 <a|H|Psi> 的 PT2 矩阵元。"""

    def __init__(self, h1e, eri, norb, nelec, *,
                 backend: str = "cpu",
                 gpu_eigsh_mode: str = "hybrid",
                 warm_start: bool = False):
        self.h1e = np.asarray(h1e)
        self.eri = np.asarray(eri)
        self.norb = norb
        self.nelec = nelec
        self.h2e = direct_spin1.absorb_h1e(self.h1e, self.eri, norb, nelec, 0.5)
        self.h2e = ao2mo.restore(1, self.h2e, norb)
        self.myci = selected_ci.SCI()
        # GPU 后端: 无 cupy/GPU 时静默回退 CPU (绝不 raise, round_003 §6.4)
        self.backend = backend
        # round_005 三模式旋钮 (仅 backend=="gpu" 时 diag 读; CPU 路径完全不触及):
        #   "hybrid"       (默认, 新): scipy.sparse.linalg.eigsh + GPU matvec (sigma)。
        #                   绕开 cupyx ARPACK 收敛停滞 (round_003/004 实证 #3), 引擎换回
        #                   scipy 继承 ~700-811 N_matvec, matvec 仍走 GPU (5-15× per-mv)。
        #   "cupyx"        (诊断/调参/方向 B 基线): 原 cupyx.eigsh + maxiter 护栏 (方向 C,
        #                   stall -> ArpackNoConvergence -> except 回退 CPU, 不再挂死)。
        #   "cpu_fallback" (逃生舱): scipy eigsh + contract_2e (GPU 不参与 matvec),
        #                   隔离 "慢在 eigsh 还是 init" 用。默认 backend="cpu" 不读此参数。
        self.gpu_eigsh_mode = gpu_eigsh_mode
        # round_010 warm-start v0: 默认 False = 不读不写缓存, 三处 eigsh 逐字
        # 一致 (零回归)。True 时缓存上次成功 diag 的 (sa, sb, c2d), 下次 diag 经
        # _project_v0 投影作 eigsh v0 —— 只减少 ARPACK 迭代, 收敛值不变
        # (E diff ≤ 1e-10, round_010 P1)。缓存生命周期 = 实例生命周期:
        # solve_sqd_adaptive 换基重建 _Subspace -> 缓存自动失效 (旧基 c2d 对
        # 新基是错误初猜, 结构上排除跨基污染)。
        self.warm_start = warm_start
        self._warm = None                  # (sa, sb, c2d) 上次成功 diag 的缓存
        self.last_n_mv = 0                 # 最近一次 diag 的 matvec 次数 (诊断仪表)
        # round_004 方式 C: 实例级 eri 缓存 (cupy 常驻 GPU, 一次)。
        # _eri1_aaaa/_bbaa 与 self.h2e 同生命周期 (init 固定); solve_sqd_adaptive 换基
        # 重建 _Subspace (cipsi.py:675) -> 新 h1e/eri -> 新 self.h2e -> 新缓存, 语义自洽。
        # CPU 路径不预算 (None -> diag 不读, 零开销); GPU 路径预算两个小 cupy 数组
        # (norb=12: eri1_aaaa=(66,66)≈35KB, eri1_bbaa=(78,78)≈49KB, 合计 ~84KB)。
        self._eri1_aaaa = None
        self._eri1_bbaa = None
        if backend == "gpu":
            from .noise import has_gpu
            if not has_gpu():
                self.backend = "cpu"
            else:
                # 方式 C: 用 self.h2e (== sigma_selected_ci_gpu 内部重算的 h2e, 逐字
                # 相同) 预算 eri1_aaaa/_bbaa, cp.asarray 一次常驻 GPU。sigma 内部
                # 喂 cp.matmul 直接用, 不再 cp.asarray。
                from .selected_ci_gpu import _selci_eri_aaaa, _selci_eri_bbaa
                import cupy as cp
                self._eri1_aaaa = cp.asarray(_selci_eri_aaaa(self.h2e, norb))
                self._eri1_bbaa = cp.asarray(_selci_eri_bbaa(self.h2e, norb, nelec))

    def diag(self, str_a, str_b):
        """对角化 (str_a, str_b) 子空间, 返回 (E_gs, c2d, sa, sb)。"""
        sa = np.asarray(sorted(str_a), dtype=np.int64)
        sb = np.asarray(sorted(str_b), dtype=np.int64)
        nA, nB = len(sa), len(sb)
        dim = nA * nB

        # round_010 warm-start v0: 上次成功 diag 的解态投影到本子空间 (§1.1)。
        # 默认 warm_start=False 时既不读也不写缓存, v0 恒 None -> kw={} ->
        # 三处 eigsh 调用与改动前逐字一致 (零回归)。dim≤1000 走 dense eigh
        # (分支 ①), 不需要 v0, 不触及。
        v0 = None
        if self.warm_start and self._warm is not None and dim > 1000:
            sa_o, sb_o, c2d_o = self._warm
            v0 = _project_v0(sa_o, sb_o, c2d_o, sa, sb)
        # v0=None 必须不传该 kwarg (scipy 显式 v0=None 与省略语义可能不同):
        kw = {"v0": v0} if v0 is not None else {}

        # round_010 仪表: 本次 diag 的 matvec 次数 (每 diag 清零, matvec 闭包自增;
        # P0'/P2 验收锚)。纯整数自增, 不触数值路径。
        self.last_n_mv = 0

        def _counted(fn):
            def _mv(v):
                self.last_n_mv += 1
                return fn(v)
            return _mv

        # round_004 方式 B: hop / link 懒构到 _build_cpu_hop 闭包。
        # GPU 成功路径完全不付 _all_linkstr_index (CPU 合并索引) 的构建税;
        # 仅分支 ① (dim≤1000)、分支 ②-CPU (默认)、分支 ②-GPU 的 except 回退
        # 三处调用 -> 拿到 (link, hop) 二元组 (闭包内 nonlocal 缓存)。
        link = None
        hop = None

        def _build_cpu_hop():
            nonlocal link, hop
            link = selected_ci._all_linkstr_index((sa, sb), self.norb, self.nelec)

            def hop(v):
                v = np.ascontiguousarray(v, dtype=np.float64)
                hv = self.myci.contract_2e(
                    self.h2e, selected_ci._as_SCIvector(v, (sa, sb)),
                    self.norb, self.nelec, link).reshape(-1)
                return np.ascontiguousarray(hv, dtype=np.float64)
            return hop

        if dim <= 1000:
            # 分支 ①: 始终 CPU numpy eigh (不读 backend; GPU 小维度 25× 启动开销无优势)
            _build_cpu_hop()
            H = np.zeros((dim, dim))
            for col in range(dim):
                e = np.zeros(dim)
                e[col] = 1.0
                H[:, col] = hop(e)
            ev, cv = np.linalg.eigh(H)
            E, c1d = float(ev[0]), cv[:, 0]
        elif self.backend == "gpu":
            # 分支 ②-GPU: round_005 三模式路由 (hybrid/cupyx/cpu_fallback)。
            # round_003/004 实证 cupyx ARPACK 收敛停滞 (#3: matvec 次数 8.8-33× scipy),
            # round_005 方向 A 把本征引擎从 cupyx.eigsh 换成 scipy.sparse.linalg.eigsh,
            # matvec 仍走 GPU sigma (5-15× per-mv) -> 继承 scipy ~700-811 N_matvec。
            # _gpu_sigma(v) 共用: hybrid 与 cupyx 共用同一 GPU matvec 核 (含 eri 缓存),
            # 仅返回类型分支 (hybrid .get() 回 numpy 给 scipy; cupyx 留 cupy 给 cupyx)。
            # GPU 成功路径不调 _build_cpu_hop (方式 B 一半收益, round_004 保留)。
            import cupy as cp
            from pyscf.fci import selected_ci as _sci
            from .selected_ci_gpu import sigma_selected_ci_gpu, _get_kernels
            kernels = _get_kernels()
            links = [_sci.des_des_linkstr(sa, self.norb, self.nelec[0], True),
                     _sci.des_des_linkstr(sb, self.norb, self.nelec[1], True),
                     _sci.cre_des_linkstr(sa, self.norb, self.nelec[0], True),
                     _sci.cre_des_linkstr(sb, self.norb, self.nelec[1], True)]

            def _gpu_sigma(v):
                xv = cp.asarray(v).reshape(nA, nB)
                return sigma_selected_ci_gpu(
                    xv, sa, sb, self.norb, self.nelec, self.h1e, self.eri,
                    links=links, kernels=kernels,
                    eri1_aaaa=self._eri1_aaaa, eri1_bbaa=self._eri1_bbaa)

            try:
                if self.gpu_eigsh_mode == "hybrid":
                    # 方向 A: scipy ARPACK 黑盒驱动 GPU matvec (绕开 cupyx 收敛停滞)。
                    # matvec(v): v numpy (scipy 给) -> cp.asarray H2D -> sigma GPU ->
                    # .get() D2H 回 numpy 给 scipy。引擎 == CPU else 分支的同一个
                    # scipy.sparse.linalg.eigsh -> N_matvec 同分布 (~700-811, theory §1.2)。
                    # round_010: v0 经 _project_v0 投影 (含 except 回退共用同一 v0)。
                    def matvec(v):
                        self.last_n_mv += 1
                        return np.asarray(_gpu_sigma(v).get()).ravel()
                    op = LinearOperator((dim, dim), matvec=matvec, dtype=np.float64)
                    ev, cv = eigsh(op, k=1, which="SA", tol=1e-10, maxiter=3000, **kw)
                    E, c1d = float(ev[0]), np.asarray(cv).ravel()
                elif self.gpu_eigsh_mode == "cupyx":
                    # round_004 现状 (诊断/调参/方向 B 基线) + 方向 C maxiter 护栏。
                    # maxiter=3000 (~3.7× scipy 的 811): 正常 cupyx 收敛能过; 病理 stall
                    # (7169 matvec @ dim 5e5) 触发 ArpackNoConvergence -> except 回退 CPU。
                    # 复现 round_004 原始 cupyx wall (无护栏) 时临时改大 maxiter。
                    from cupyx.scipy.sparse.linalg import eigsh as cp_eigsh
                    from cupyx.scipy.sparse.linalg import LinearOperator as cp_LO
                    def matvec(v):
                        self.last_n_mv += 1
                        return _gpu_sigma(v).reshape(-1)
                    A = cp_LO((dim, dim), matvec=matvec, dtype=np.float64)
                    ev, cv = cp_eigsh(A, k=1, which="SA", tol=1e-10, maxiter=3000, **kw)
                    E, c1d = float(ev[0]), np.asarray(cv).ravel()
                else:  # "cpu_fallback" (逃生舱: scipy eigsh + contract_2e, GPU 不参与)
                    _build_cpu_hop()
                    op = LinearOperator((dim, dim), matvec=_counted(hop), dtype=np.float64)
                    ev, cv = eigsh(op, k=1, which="SA", tol=1e-10, maxiter=3000, **kw)
                    E, c1d = float(ev[0]), np.asarray(cv).ravel()
            except Exception:
                # OOM / cupyx 不收敛 (maxiter 触发 ArpackNoConvergence) / scipy 不收敛
                # -> 懒构 hop + scipy eigsh 回退 (round_004 护栏, 三模式共用)。
                # round_010: 同一 v0 照样有效 (同矩阵同子空间), 不浪费。
                _build_cpu_hop()
                op = LinearOperator((dim, dim), matvec=_counted(hop), dtype=np.float64)
                ev, cv = eigsh(op, k=1, which="SA", maxiter=3000, **kw)
                E, c1d = float(ev[0]), np.asarray(cv).ravel()
        else:
            # 分支 ②-CPU: scipy eigsh (默认, 与改造前逐字一致, L1 零回归)
            _build_cpu_hop()
            op = LinearOperator((dim, dim), matvec=_counted(hop), dtype=np.float64)
            ev, cv = eigsh(op, k=1, which="SA", maxiter=3000, **kw)
            E, c1d = float(ev[0]), np.asarray(cv).ravel()
        c2d = c1d.reshape(nA, nB)
        # round_010: diag 成功后缓存 (gate 在 warm_start 上; 含 prune 收缩后的新态)
        if self.warm_start:
            self._warm = (sa, sb, np.array(c2d, copy=True))
        return E, c2d, sa, sb

    def pt2_matrix_elements(self, str_a, str_b, cand, c2d, sa, sb):
        """扩展空间 (str_a∪cand_α, str_b∪cand_β) 上算各候选的 <a|H|Psi> 与对角元。

        返回 dict {(ca, cb): (hpsi, Ea)}。
        """
        idx_b = {int(s): i for i, s in enumerate(sb)}
        set_a = set(str_a)
        set_b = set(str_b)
        for ca, cb in cand:
            set_a.add(ca)
            set_b.add(cb)
        sA = np.asarray(sorted(set_a), dtype=np.int64)
        sB = np.asarray(sorted(set_b), dtype=np.int64)
        nB = len(sB)
        idx_a = {int(s): i for i, s in enumerate(sA)}
        idx_b2 = {int(s): i for i, s in enumerate(sB)}
        dim2 = len(sA) * nB

        psi = np.zeros(dim2)
        for ia in range(len(sa)):
            for ib in range(len(sb)):
                psi[idx_a[int(sa[ia])] * nB + idx_b2[int(sb[ib])]] = c2d[ia, ib]
        link2 = selected_ci._all_linkstr_index((sA, sB), self.norb, self.nelec)
        hdiag = selected_ci.make_hdiag(self.h1e, self.eri, (sA, sB),
                                       self.norb, self.nelec)
        Hpsi = self.myci.contract_2e(
            self.h2e, selected_ci._as_SCIvector(psi, (sA, sB)),
            self.norb, self.nelec, link2).reshape(-1)

        out = {}
        for ca, cb in cand:
            k = idx_a[ca] * nB + idx_b2[cb]
            out[(ca, cb)] = (Hpsi[k], hdiag[k])
        return out


def solve_cipsi(
    one_body_tensor: np.ndarray,
    two_body_tensor: np.ndarray,
    norb: int,
    nelec: Tuple[int, int],
    *,
    seed_bitstring_matrix: np.ndarray,
    max_strings: Optional[int] = None,
    dom_thresh: float = 1e-3,
    pt2_floor: float = 1e-7,
    max_iter: int = 40,
    ecore: float = 0.0,
    verbose: bool = False,
    backend: str = "cpu",
) -> float:
    """CIPSI 迭代精化: 从种子 det 集合出发补全到近 FCI 精度。

    Parameters
    ----------
    one_body_tensor : ndarray, shape (norb, norb)
        单电子积分 (MO 基, 闭壳层单矩阵)。
    two_body_tensor : ndarray, shape (norb, norb, norb, norb)
        两电子积分 (chemist 记号)。
    norb : int
        空间轨道数。
    nelec : tuple(int, int)
        ``(n_alpha, n_beta)``。
    seed_bitstring_matrix : ndarray, shape (S, 2*norb)
        种子 det 集合 (位串矩阵, 如 ``ucj_assisted_configurations`` 输出或
        ``np.vstack([exc, ucj])``)。
    max_strings : int | None
        字符串集合上限 (对角化维度 ≈ n_str_a × n_str_b)。``None`` = 默认
        补全到全空间 ``C(norb, nelec[0])``。
    dom_thresh : float
        主导 det 的 |c| 阈值 (低于此的 det 不参与生成集扩展)。
    pt2_floor : float
        |PT2| 低于此的候选 det 不再加入。
    max_iter : int
        迭代轮数上限。
    ecore : float
        Core 能量偏移 (核排斥 + frozen-core), 计入返回值。
    verbose : bool
        打印每轮空间大小 / 能量 / PT2 信息。

    Returns
    -------
    energy : float
        基态能量 (含 ``ecore``)。

    Notes
    -----
    - 矩阵元全部走 PySCF ``contract_2e``, 避免手写 Slater-Condon 的相位/符号坑。
    - 子空间表示沿用库的字符串乘积: 闭壳层 α/β 合并, 开壳层独立。
    - 空间扩展按 PT2 排序; 由于种子 (UCJ 辅助) 已覆盖全空间大部分字符串,
      通常 1-2 轮即补全到全空间 = FCI。
    """
    from .fermion import bitstring_matrix_to_ci_strs

    if one_body_tensor.ndim == 3:
        if not np.allclose(one_body_tensor[0], one_body_tensor[1]):
            raise ValueError(
                "solve_cipsi 不支持自旋分辨 h1e (h_alpha != h_beta); "
                "请传单个 (norb, norb) 闭壳层 h1e。"
            )
        h1e = np.asarray(one_body_tensor[0])
    else:
        h1e = np.asarray(one_body_tensor)
    eri = np.asarray(two_body_tensor)

    # 种子 -> 字符串集合 (闭壳层合并, 开壳层独立)
    na, nb = nelec
    if na == nb:
        ci_a, ci_b = bitstring_matrix_to_ci_strs(seed_bitstring_matrix)
        str_a = sorted(set(int(x) for x in ci_a))
        str_b = str_a
    else:
        ci_a, ci_b = bitstring_matrix_to_ci_strs(seed_bitstring_matrix, open_shell=True)
        str_a = sorted(set(int(x) for x in ci_a))
        str_b = sorted(set(int(x) for x in ci_b))

    full_size = int(cistring.num_strings(norb, na))
    if max_strings is None:
        max_strings = full_size

    sub = _Subspace(h1e, eri, norb, nelec, backend=backend)
    for it in range(max_iter):
        if len(str_a) >= max_strings or len(str_b) >= max_strings:
            break
        E, c2d, sa, sb = sub.diag(str_a, str_b)
        idx_a = {int(s): i for i, s in enumerate(sa)}
        idx_b = {int(s): i for i, s in enumerate(sb)}

        # 主导 dets
        nA, nB = c2d.shape
        flat = np.abs(c2d).ravel()
        order = np.argsort(flat)[::-1]
        dom = []
        for k in order:
            if flat[k] > dom_thresh:
                ia, ib = divmod(int(k), nB)
                dom.append((int(sa[ia]), int(sb[ib])))
            else:
                break
        if not dom:
            break

        # 候选连接 (新 det: 笛卡尔积空间里 ca 或 cb 不在当前集合)
        cand = set()
        for a, b in dom:
            for ca, cb in _excited_dets(a, b, norb):
                if ca not in idx_a or cb not in idx_b:
                    cand.add((ca, cb))
        if not cand:
            break

        # PT2 筛选
        me = sub.pt2_matrix_elements(str_a, str_b, cand, c2d, sa, sb)
        pt2 = {det_: hpsi * hpsi / (E - Ea) for det_, (hpsi, Ea) in me.items()
               if abs(E - Ea) > 1e-12}
        ranked = sorted(pt2.items(), key=lambda kv: -abs(kv[1]))

        add = []
        pt2_sum = 0.0
        for det_, v in ranked:
            if abs(v) < pt2_floor:
                break
            if len(str_a) + len(add) >= max_strings:
                break
            add.append(det_)
            pt2_sum += v
        if not add:
            break
        for ca, cb in add:
            str_a.append(ca)
            if cb not in str_b:
                str_b.append(cb)
        str_a = sorted(set(str_a))
        str_b = sorted(set(str_b))
        if verbose:
            dim_now = len(str_a) * len(str_b)
            print(f"[CIPSI] it{it}: strings={len(str_a)}x{len(str_b)} "
                  f"diag_dim={dim_now} E={E + ecore:.6f} pt2_top={pt2_sum:.2e}")

    E, c2d, sa, sb = sub.diag(str_a, str_b)
    return float(E) + ecore


# --------------------------------------------------------------------------- #
#  真正的 HCI (heat-bath CI): |<j|H|i>| >= eps_hb 选态 (Holmes 2016 JCTC)
# --------------------------------------------------------------------------- #
def solve_hci(
    one_body_tensor: np.ndarray,
    two_body_tensor: np.ndarray,
    norb: int,
    nelec: Tuple[int, int],
    *,
    seed_bitstring_matrix: Optional[np.ndarray] = None,
    eps_hb: float = 1e-3,
    dom_thresh: float = 1e-3,
    pt2_floor: float = 1e-7,
    max_iter: int = 40,
    ecore: float = 0.0,
    verbose: bool = False,
    return_details: bool = False,
    backend: str = "cpu",
):
    """SHCI (heat-bath CI + PT2 修正, Holmes 2016 / Sharma 2017).

    **与 solve_cipsi 的区别 (heat-bath 选态 vs PT2 全排序)**:
      - :func:`solve_cipsi` (CIPSI): 候选加入用完整 Epstein-Nesbet PT2 得分
        ``⟨a|H|Ψ⟩²/(E−E_a)`` 排序选 top —— 每轮对全波函数求 H|Ψ⟩。
      - 本函数 (SHCI): **两阶段** —— ① heat-bath 选态 ``|⟨j|H|i⟩| ≥ eps_hb``
        构建变分空间 V (只用单参考 det 对矩阵元, 不求完整 ⟨a|H|Ψ⟩); ② 对角化 V
        得 ``E_V``, 对 V 外候选算 **PT2 能量修正** ``E_PT2 = Σ_a |⟨a|H|Ψ⟩|²/(E−E_a)``。
        返回标准 SHCI 报告的总能量 ``E_total = E_V + E_PT2``。

    **参数** (SHCI 双阈值): ``eps_hb`` = ε₁ (变分空间选态); ``pt2_floor`` = ε₂
    (PT2 修正精度参考, 用于判断变分空间是否足够; 本实现不做 semistochastic,
    ε₂ 仅标注)。

    **实现** (朴素 heat-bath, 同 pyscf/naive-hci 思路): 对每个主导 det |i⟩,
    枚举单/双激发候选 |j⟩, 用单位向量 ``e_i`` 经 PySCF ``contract_2e`` 一次算
    ``⟨j|H|i⟩`` (复用 :class:`_Subspace.pt2_matrix_elements`, 传 ``c2d=e_i``)。

    Parameters
    ----------
    one_body_tensor : ndarray (norb, norb)
        单电子积分 (闭壳层单矩阵)。
    two_body_tensor : ndarray (norb, norb, norb, norb)
        双电子积分 (chemist 记号)。
    norb : int
        空间轨道数。
    nelec : tuple(int, int)
        ``(n_alpha, n_beta)``。
    seed_bitstring_matrix : ndarray (S, 2*norb) | None
        种子 det 集合 (位串)。``None`` = 从 HF 出发 (标准 HCI)。
    eps_hb : float
        heat-bath 选态阈值 (ε₁): ``|⟨j|H|i⟩| ≥ eps_hb`` 的候选 det 加入变分空间。
        越小变分空间越大, E_PT2 越小, 越接近 FCI。
    dom_thresh : float
        主导 det 的 |c| 阈值 (低于此不参与生成集扩展)。
    pt2_floor : float
        PT2 修正阈值 (ε₂ 参考): 仅 verbose 标注变分空间是否足够, 不强制收敛。
    max_iter : int
        迭代轮数上限。
    ecore : float
        Core 能量偏移, 计入返回值。
    verbose : bool
        打印每轮变分空间/能量/PT2。
    return_details : bool
        ``True`` 返回 ``(E_total, E_PT2, dim)`` 元组 (含 ecore 的 E_total,
        不含 ecore 的 E_PT2, 变分空间维度) —— 供诊断/绘图。

    Returns
    -------
    float | tuple
        ``return_details=False``: SHCI 总能量 ``E_V + E_PT2`` (含 ``ecore``)。
        ``return_details=True``: ``(E_total, E_PT2, dim)``。
    """
    from .fermion import bitstring_matrix_to_ci_strs

    if one_body_tensor.ndim == 3:
        if not np.allclose(one_body_tensor[0], one_body_tensor[1]):
            raise ValueError(
                "solve_hci 不支持自旋分辨 h1e; 请传闭壳层 (norb, norb)。"
            )
        h1e = np.asarray(one_body_tensor[0])
    else:
        h1e = np.asarray(one_body_tensor)
    eri = np.asarray(two_body_tensor)
    na, nb = nelec
    open_shell = na != nb

    # 种子 -> 字符串集合 (默认 HF)
    if seed_bitstring_matrix is None:
        hf_a = (1 << na) - 1
        hf_b = (1 << nb) - 1
        str_a = [hf_a]
        str_b = [hf_b] if open_shell else [hf_a]
    else:
        ci_a, ci_b = bitstring_matrix_to_ci_strs(
            seed_bitstring_matrix, open_shell=open_shell)
        str_a = sorted(set(int(x) for x in ci_a))
        str_b = sorted(set(int(x) for x in ci_b))
        if not open_shell:
            str_b = str_a

    sub = _Subspace(h1e, eri, norb, nelec, backend=backend)

    def _dominant(c2d, sa, sb):
        nA, nB = c2d.shape
        flat = np.abs(c2d).ravel()
        order = np.argsort(flat)[::-1]
        dom = []
        for k in order:
            if flat[k] > dom_thresh:
                ia, ib = divmod(int(k), nB)
                dom.append((int(sa[ia]), int(sb[ib])))
            else:
                break
        return dom

    # ---- 阶段 1: heat-bath 选态构建变分空间 V (|⟨j|H|i⟩| ≥ eps_hb, 到无新增) ----
    for it in range(max_iter):
        E, c2d, sa, sb = sub.diag(str_a, str_b)
        idx_a = {int(s): i for i, s in enumerate(sa)}
        idx_b = {int(s): i for i, s in enumerate(sb)}
        dom = _dominant(c2d, sa, sb)
        if not dom:
            break

        hb_new = set()
        sa_list, sb_list = list(sa), list(sb)
        for a, b in dom:
            cand = _excited_dets(a, b, norb)
            cand = {(ca, cb) for (ca, cb) in cand
                    if ca not in idx_a or cb not in idx_b}
            if not cand:
                continue
            # 单位向量 e_i: 只在主导 det (a,b) 处为 1 -> H e_i 的第 j 分量 = <j|H|i>
            e_i = np.zeros((len(sa), len(sb)))
            e_i[sa_list.index(a), sb_list.index(b)] = 1.0
            me = sub.pt2_matrix_elements(str_a, str_b, cand, e_i, sa, sb)
            for (ca, cb), (hji, _) in me.items():
                if abs(hji) >= eps_hb:
                    hb_new.add((ca, cb))
        if not hb_new:
            break
        for ca, cb in hb_new:
            str_a.append(ca)
            if cb not in str_b:
                str_b.append(cb)
        str_a = sorted(set(str_a))
        str_b = str_a if not open_shell else sorted(set(str_b))
        if verbose:
            print(f"[HCI:hb] it{it+1}/{max_iter}: dim={len(str_a)*len(str_b)} "
                  f"E_V={E + ecore:.6f} new={len(hb_new)}")

    # ---- 阶段 2: 对角化 + PT2 能量修正 (E_PT2 = Σ |⟨a|H|Ψ⟩|²/(E−E_a)) ----
    E, c2d, sa, sb = sub.diag(str_a, str_b)
    idx_a = {int(s): i for i, s in enumerate(sa)}
    idx_b = {int(s): i for i, s in enumerate(sb)}
    dom = _dominant(c2d, sa, sb)

    cand_all = set()
    for a, b in dom:
        for ca, cb in _excited_dets(a, b, norb):
            if ca not in idx_a or cb not in idx_b:
                cand_all.add((ca, cb))
    if cand_all:
        me = sub.pt2_matrix_elements(str_a, str_b, cand_all, c2d, sa, sb)
        pt2 = {d: h * h / (E - Ea) for d, (h, Ea) in me.items()
               if abs(E - Ea) > 1e-12}
        e_pt2 = float(sum(pt2.values()))
    else:
        e_pt2 = 0.0
    dim = len(str_a) * len(str_b)
    e_total = float(E + e_pt2) + ecore

    if verbose:
        print(f"[HCI] dim={dim} E_V={E + ecore:.8f} E_PT2={e_pt2:.2e} "
              f"E_total={e_total:.8f} "
              f"{'PT2 OK' if abs(e_pt2) < pt2_floor else 'PT2 large (reduce eps_hb)'}")
    if return_details:
        return e_total, float(e_pt2), dim
    return e_total


# --------------------------------------------------------------------------- #
#  自适应 SQD (方向①②组合): 自洽换基表示层 + 受限 PT2 选态选择层
# --------------------------------------------------------------------------- #
def solve_sqd_adaptive(
    one_body_tensor: np.ndarray,
    two_body_tensor: np.ndarray,
    norb: int,
    nelec: Tuple[int, int],
    *,
    bitstring_matrix: np.ndarray,
    probabilities: Optional[np.ndarray] = None,
    avg_occupancies: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    max_strings: Optional[int] = None,
    n_active_per_round: int = 50,
    max_pt2_iters: int = 3,
    dom_thresh: float = 1e-3,
    pt2_floor: float = 1e-7,
    max_rounds: int = 10,
    energy_tol: float = 1e-9,
    ecore: float = 0.0,
    rand_seed: Optional[int] = 0,
    verbose: bool = False,
    rounds_out: Optional[list] = None,
    backend: str = "cpu",
) -> float:
    """自适应 SQD: 自洽换基表示层 (方向①) + 受限 PT2 选态选择层 (方向②) 叠加。

    **统一视角** (REVIEW 方向③): 表示层 (自然轨道换基使展开系数集中) + 生成层
    (多样初猜) + 选择层 (PT2 确定性补足)。本函数组合**表示层与选择层**:

    每轮:
      1. 配置恢复 (当前基偏置平均占据) → 当前基 det 集合
      2. 受限 PT2 精化 (当轮/当前基): 主导 det 枚举单双激发 → PT2 top-K 注入
         (子空间受限, 不补全全空间)
      3. 子空间对角化 → E
      4. 解态 1-RDM → 自然轨道换基 → 更新 h1e/eri/平均占据 (下一轮采样更聚焦)
      5. 能量稳定则收敛

    **与单独方法的关系**:
      - ``solve_sqd_active`` 只有选择层 (基固定); 本函数换基使下一轮采样 det 更有效
      - ``solve_sqd_natural_orbitals`` 只有表示层 (无 PT2); 本函数当轮 PT2 补足
        采样缺口 → 更准的 1-RDM → 更准的换基 (正反馈)

    Parameters
    ----------
    同 :func:`solve_sqd_active`, 外加:
    max_pt2_iters : int
        每轮内受限 PT2 精化的迭代次数 (采样后确定性补足的程度)。
    energy_tol : float
        能量收敛阈值 (连续两轮变化小于它即停止换基)。

    Returns
    -------
    energy : float
        基态总能量 (含 ``ecore``)。
    """
    from .basis import rotate_to_natural_orbitals

    if one_body_tensor.ndim == 3:
        if not np.allclose(one_body_tensor[0], one_body_tensor[1]):
            raise ValueError(
                "solve_sqd_adaptive 不支持自旋分辨 h1e; 请传闭壳层 (norb, norb)。"
            )
        h1e = np.asarray(one_body_tensor[0])
    else:
        h1e = np.asarray(one_body_tensor)
    eri = np.asarray(two_body_tensor)
    na, nb = nelec
    open_shell = na != nb

    bsm = np.asarray(bitstring_matrix, dtype=bool)
    if bsm.ndim != 2 or bsm.shape[1] != 2 * norb:
        raise ValueError(
            f"bitstring_matrix must have shape (S, 2*norb={2*norb}), got {bsm.shape}."
        )
    probs = (np.full(bsm.shape[0], 1.0 / bsm.shape[0]) if probabilities is None
             else np.asarray(probabilities, dtype=np.float64))

    if avg_occupancies is not None:
        occ_a, occ_b = avg_occupancies
    else:
        occ_a = np.zeros(norb, dtype=np.float64)
        occ_a[:na] = 1.0
        occ_b = np.zeros(norb, dtype=np.float64)
        occ_b[:nb] = 1.0

    full_size = int(cistring.num_strings(norb, na))
    if max_strings is None:
        max_strings = full_size

    sub = _Subspace(h1e, eri, norb, nelec, backend=backend)
    e_prev = np.inf
    best_E = np.inf                       # 各轮 ③ 能量的 min (每轮自洽: B_r sub + B_r dets)
    n_rounds_done = 0

    for r in range(max_rounds):
        # ① 配置恢复 (当前基偏置平均占据) → 当前基 det
        rec, _ = recover_configurations(
            bsm, probs, (occ_a, occ_b), na, nb, rand_seed=rand_seed
        )
        ci_a, ci_b = bitstring_matrix_to_ci_strs(rec, open_shell=open_shell)
        str_a = sorted(set(int(x) for x in ci_a))
        str_b = str_a if not open_shell else sorted(set(int(x) for x in ci_b))

        # ② 受限 PT2 精化 (当轮, 当前基)
        for _ in range(max_pt2_iters):
            E, c2d, sa, sb = sub.diag(str_a, str_b)
            idx_a = {int(s): i for i, s in enumerate(sa)}
            idx_b = {int(s): i for i, s in enumerate(sb)}
            nA, nB = c2d.shape
            flat = np.abs(c2d).ravel()
            order = np.argsort(flat)[::-1]
            dom = []
            for k in order:
                if flat[k] > dom_thresh:
                    ia, ib = divmod(int(k), nB)
                    dom.append((int(sa[ia]), int(sb[ib])))
                else:
                    break
            cand = set()
            if dom:
                for a, b in dom:
                    for ca, cb in _excited_dets(a, b, norb):
                        if ca not in idx_a or cb not in idx_b:
                            cand.add((ca, cb))
            if not cand:
                break
            me = sub.pt2_matrix_elements(str_a, str_b, cand, c2d, sa, sb)
            pt2 = {d: h * h / (E - Ea) for d, (h, Ea) in me.items()
                   if abs(E - Ea) > 1e-12}
            ranked = sorted(pt2.items(), key=lambda kv: -abs(kv[1]))
            add = []
            for d, v in ranked:
                if abs(v) < pt2_floor:
                    break
                if len(str_a) + len(add) >= max_strings:
                    break
                add.append(d)
            if len(add) > n_active_per_round:
                add = add[:n_active_per_round]
            if not add:
                break
            for ca, cb in add:
                str_a.append(ca)
                if cb not in str_b:
                    str_b.append(cb)
            str_a = sorted(set(str_a))
            str_b = str_a if not open_shell else sorted(set(str_b))

        # ③ 最终对角化 (当前基)
        E, c2d, sa, sb = sub.diag(str_a, str_b)
        dim_now = len(sa) * len(sb)
        best_E = min(best_E, E)           # 每轮 E 自洽 (B_r sub + B_r dets), 取 min 作变分上界

        # ④ 表示层: 解态 1-RDM → 自然轨道换基
        st = SCIState(amplitudes=c2d, ci_strs_a=np.asarray(sa),
                      ci_strs_b=np.asarray(sb), norb=norb, nelec=nelec)
        dm1 = st.rdm(rank=1, spin_summed=True)
        h1e, eri, U_step, occ_nat = rotate_to_natural_orbitals(h1e, eri, dm1)
        sub = _Subspace(h1e, eri, norb, nelec, backend=backend)  # 重建 (新基)
        occ_a = np.clip(occ_nat / 2.0, 0.0, 1.0)
        occ_b = occ_a.copy()

        if verbose:
            print(f"[adaptive r{r+1}/{max_rounds}] E={E + ecore:.6f} "
                  f"dim={dim_now} |c2|max={float(np.abs(c2d).max() ** 2):.4f}")

        n_rounds_done = r + 1
        if rounds_out is not None:
            rounds_out.append(float(E))           # 各轮 ③ 自洽能量 (B_r sub + B_r dets)
        if r > 0 and abs(E - e_prev) < energy_tol:
            break
        e_prev = E

    # 返回各轮 ③ 能量的 min (每轮自洽: B_r sub + B_r dets)。**不**在循环后再做
    # sub.diag —— 末轮 ④ 已把 sub 重建到 B_{last+1}, 而 str_a 仍是 B_last 的,
    # 那会是混合基对角化 (与 solve_sqd_natural_orbitals F2 同类 bug), 能量不自洽
    # (实测返回值比真正最优轮差 ~2.6e-7, 正是 REVIEW 报的"adaptive 略逊 active"根因)。
    if verbose:
        print(f"[adaptive] 收敛 @ round {n_rounds_done}: best_E={best_E + ecore:.8f} "
              f"(各轮 ③ 自洽能量取 min, 变分上界)")
    return float(best_E) + ecore


# --------------------------------------------------------------------------- #
#  主动采样 SQD (方向②): 受限 PT2 选态 + 采样聚焦 双闭环
# --------------------------------------------------------------------------- #
def solve_sqd_active(
    one_body_tensor: np.ndarray,
    two_body_tensor: np.ndarray,
    norb: int,
    nelec: Tuple[int, int],
    *,
    bitstring_matrix: np.ndarray,
    probabilities: Optional[np.ndarray] = None,
    avg_occupancies: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    max_strings: Optional[int] = None,
    n_active_per_round: int = 50,
    dom_thresh: float = 1e-3,
    pt2_floor: float = 1e-7,
    max_rounds: int = 10,
    ecore: float = 0.0,
    rand_seed: Optional[int] = 0,
    verbose: bool = False,
    # ---- B1 预算闭环 (增量采样 + 能量收敛停采) ----
    shots_budget: Optional[int] = None,
    shots_step: int = 0,
    energy_tol: Optional[float] = None,
    usage: Optional[list] = None,
    # ---- 能量-方差外推轨迹 (方向 D) ----
    trajectory: Optional[list] = None,
    # ---- 自蒸馏 (方向②): 取出最终本征矢供重采样 ----
    state_out: Optional[list] = None,
    # ---- C1 尾部发现采样 (round_001) ----
    tail_suppression: bool = False,
    tail_max_draw_factor: int = 10,
    tail_n_target_per_round: int = 0,
    # ---- C1 预算随 shots 缩放 (round_002) ----
    tail_shots_ref: int = 0,
    # ---- 方向 B: PT2 排序剪枝 (round_007); 默认全关零回归 ----
    prune_keep: float = 1.0,
    # ---- 方向 C: 三激发定向注入 (round_008); 默认全关零回归 ----
    triple_injection: bool = False,
    n_triples_per_round: int = 0,
    # ---- round_010: warm-start v0 (默认关零回归) ----
    warm_start: bool = False,
    backend: str = "cpu",
) -> float:
    """主动采样 SQD: 采样/配置恢复 ↔ 受限 PT2 选态 双闭环 (AS-SQD 思想, 方向②)。

    **动机**: 纯采样 SQD 的子空间只含"采到"的 det, 低采样/噪声下覆盖不全
    (C₂ 曾 3/8 失败)。AS-SQD (Miura, arXiv:2603.13536) 用 Epstein-Nesbet
    PT2 得分从外部候选**确定性补足**采样缺口 —— 无需额外量子测量, 且噪声
    bitstring 的 PT2 得分近零 (抗噪)。

    **与 solve_cipsi 的区别**: :func:`solve_cipsi` 是**纯经典 det 空间精化**
    (静态种子, 补全到全空间, 不碰采样); 本函数是**采样与选态双闭环** ——
    每轮先用**偏置的平均占据**做配置恢复 (采样聚焦), 再用受限 PT2 注入
    高价值 det (子空间不补全全空间), 两者交替直到收敛。

    Parameters
    ----------
    one_body_tensor : ndarray, shape (norb, norb)
        单电子积分 (闭壳层单矩阵)。
    two_body_tensor : ndarray, shape (norb, norb, norb, norb)
        双电子积分 (chemist 记号)。
    norb : int
        空间轨道数。
    nelec : tuple(int, int)
        ``(n_alpha, n_beta)``。
    bitstring_matrix : ndarray, shape (S, 2*norb)
        采样位串 (电路 shot 或经典随机种子)。配置恢复每轮按当前平均占据修正。
    probabilities : ndarray, shape (S,), optional
        对应概率; 省略时均匀。
    avg_occupancies : tuple(ndarray, ndarray), optional
        初始平均占据 (采样偏置)。省略时退化为 HF。
    max_strings : int | None
        字符串集合上限 (对角化维度 ≈ n_str_a × n_str_b)。``None`` = 默认
        全空间 ``C(norb, nelec[0])`` (受限时给较小值)。
    n_active_per_round : int
        每轮 PT2 选态注入的 top 候选 det 数上限 (受限核心参数)。
    dom_thresh : float
        主导 det 的 |c| 阈值 (低于此不参与生成集扩展)。
    pt2_floor : float
        |PT2| 低于此的候选 det 不再加入。
    max_rounds : int
        采样↔选态轮数上限。
    ecore : float
        Core 能量偏移, 计入返回值。
    rand_seed : int | None
        配置恢复 tie-breaking 种子。
    verbose : bool
        打印每轮空间/能量/PT2 信息。
    shots_budget : int | None
        B1 预算: 总采样预算。``bitstring_matrix`` 行数不足时预生成随机位串补足成
        采样池 (经典模拟)。``None`` = 用给定 ``bitstring_matrix`` 全量 (原行为)。
    shots_step : int
        B1 增量步长: 每轮用池的前 ``n_cur`` 行, ``n_cur`` 逐轮递增 (``>0`` 启用
        增量采样; ``0`` = 一次性全量, 原行为)。
    energy_tol : float | None
        B1 停采阈值: 连续两轮能量变化小于它即停止 (能量已收敛, 省 shots)。
        ``None`` = 不停采 (原行为)。
    usage : list | None
        B1 输出参数: 调用方传空 list, 结束后 ``usage[0]`` 为**实际使用的 shots 数**
        (预算闭环的量化指标; 不传则只返回能量)。
    trajectory : list | None
        方向 D 输出参数: 调用方传空 list, 每轮追加 ``dict`` 记录 ``{round, E,
        sigma2, e_pt2, dim, shots}``。``sigma2 = Σ_a |⟨a|H|Ψ⟩|²`` (子空间外
        PT2 分子平方和, 即对生成集的**精确方差**); ``e_pt2`` 为 Epstein-Nesbet
        PT2 全和; ``dim`` 为对角化维度。供 :func:`solve_sqd_ev` 做能量-方差
        外推 (E(σ²)→0)。不传则无额外开销。
    tail_suppression : bool
        **C1 尾部发现采样 (round_001)**: ``True`` 时每轮 ① 前调
        :func:`tc_sqd.tail_sampling.discover_tail_pool`, 用过抽 (``tail_max_draw_factor``×)
        + 抑制已见 det 收集新贡献者替代本轮固定池。**默认 ``False``, 零行为变化**
        (与 AS-SQD/PT2 正交可叠加; distill 边界: 不读 ``c2d``, 只读
        ``(seen_a, seen_b)``)。
    tail_max_draw_factor : int
        C1 自举过抽倍数 (原始抽样预算 = ``tail_max_draw_factor × n_target``; 默认 10)。
    tail_n_target_per_round : int
        C1 每轮目标新 det 数; ``0`` = 用 ``n_active_per_round``。
    tail_shots_ref : int
        **C1 预算随 shots 缩放 (round_002, C1-v2)**: ``>0`` 时启用, 每轮目标新 det 数
        按当前 shots 游标 ``n_cur`` 与 ``tail_shots_ref`` 之比线性缩放::

            n_tgt = clip(ceil(n_active_per_round * n_cur / tail_shots_ref),
                         n_active_per_round, 3 * n_active_per_round)

        使得 C1 的发现努力随用户给的 shots 量级对齐 (修复 round_001 实证的 "bootstrap
        预算与 shots 解耦" 局限)。``@100 shots`` 恰给 ``n_tgt=30`` (默认 ``=100`` →
        与 round_001 C1-v1 **逐位一致**, 零回归); ``@500 shots`` 给 ``n_tgt=90`` (cap=3
        封顶, 用上多出的 shots)。``=0`` 关闭缩放, 走 round_001 路径
        (``n_tgt = tail_n_target_per_round or n_active_per_round``)。
    prune_keep : float
        **方向 B 子空间去稀释 (round_007)**: 收敛后按最终本征矢的**字符串边际权重**
        (``Σ|c|²`` 行/列和) 排序, 每自旋保留 top ``prune_keep`` 比例的字符串
        (``keep = ceil(prune_keep × n_strings)``), 剪掉低权重尾再重对角化 +
        重算 PT2 (被剪 det 移出子空间 → 进入 ``E_PT2`` 和式, 关联被二阶回补)。
        闭壳层 (``na==nb``) 用合并权重 (行和+列和) 剪**同一**集合, 保证
        ``str_a == str_b`` 不变式。``1.0`` = 不剪枝 (默认, 零回归); ``0.6`` =
        每自旋保留 60% top 字符串 (dim ≈ 0.36×)。区间 ``(0, 1]``, 越界
        raise ValueError。
    triple_injection : bool
        **方向 C 三激发定向注入 (round_008)**: ``True`` 时在收敛循环结束后、
        最终 diag 前, 对**全部已选字符串**迭代生成单激发连接 (occ→virt,
        激发阶 +1), 按字符串级 EN-PT2 得分 (``Σ_b |⟨(s',b)|H|Ψ⟩|²/(E−E_a)``)
        排序注入 top-N, 直到无新字符串 (单激发图连通 → 12,12 补全到全空间)。
        补上现有 ④ (S/D-from-dominant) 不可达的高激发阶字符串
        (round_007 诊断的 ~16 缺失字符串)。默认 ``False``, 零行为变化。
    n_triples_per_round : int
        每迭代注入的新字符串数上限; ``0`` = 无 cap (注入所有 ``|pt2|>pt2_floor``
        的新字符串, 到 fixpoint); ``>0`` = 每迭代取 top-N (大空间定向注入)。
        仅在 ``triple_injection=True`` 时读取。
    warm_start : bool
        **round_010 对角化迭代减少**: ``True`` 时缓存上次成功 diag 的解态,
        下次 diag 经 ``_project_v0`` 投影作 eigsh ``v0`` (子空间单调增长,
        旧字符串 ⊆ 新字符串)。只减少 ARPACK 迭代次数, 收敛值不变
        (E diff ≤ 1e-10)。缓存生命周期 = ``_Subspace`` 实例生命周期。
        默认 ``False`` 零回归。

    Returns
    -------
    energy : float
        基态能量 (含 ``ecore``)。
    """
    if one_body_tensor.ndim == 3:
        if not np.allclose(one_body_tensor[0], one_body_tensor[1]):
            raise ValueError(
                "solve_sqd_active 不支持自旋分辨 h1e; 请传闭壳层 (norb, norb)。"
            )
        h1e = np.asarray(one_body_tensor[0])
    else:
        h1e = np.asarray(one_body_tensor)
    eri = np.asarray(two_body_tensor)
    na, nb = nelec
    open_shell = na != nb
    # 方向 B (round_007): prune_keep 越界提前报错 (1.0 = 不剪枝零回归, 不报错)
    if not 0.0 < prune_keep <= 1.0:
        raise ValueError(
            f"prune_keep 须在 (0, 1] 区间 (1.0 = 不剪枝零回归), got {prune_keep!r}."
        )

    bsm = np.asarray(bitstring_matrix, dtype=bool)
    if bsm.ndim != 2 or bsm.shape[1] != 2 * norb:
        raise ValueError(
            f"bitstring_matrix must have shape (S, 2*norb={2*norb}), got {bsm.shape}."
        )
    probs = (np.full(bsm.shape[0], 1.0 / bsm.shape[0]) if probabilities is None
             else np.asarray(probabilities, dtype=np.float64))

    # 初始平均占据 (采样偏置)
    if avg_occupancies is not None:
        occ_a, occ_b = avg_occupancies
    else:
        occ_a = np.zeros(norb, dtype=np.float64)
        occ_a[:na] = 1.0
        occ_b = np.zeros(norb, dtype=np.float64)
        occ_b[:nb] = 1.0

    full_size = int(cistring.num_strings(norb, na))
    if max_strings is None:
        max_strings = full_size

    sub = _Subspace(h1e, eri, norb, nelec, backend=backend,
                    warm_start=warm_start)
    str_a: list = []
    str_b: list = []
    e_prev = np.inf

    # B1 预算闭环: 采样池 (预算 > 当前行数时补足随机位串) + 增量游标
    n_pool = bsm.shape[0]
    if shots_budget is not None and shots_budget > n_pool:
        rng = np.random.default_rng(rand_seed)
        extra = rng.random((shots_budget - n_pool, 2 * norb)) > 0.5
        bsm = np.vstack([bsm, extra])
        probs = np.concatenate(
            [probs, np.full(shots_budget - n_pool, 1.0 / n_pool)]
        )
        n_pool = shots_budget
    n_cur = n_pool if shots_step <= 0 else min(shots_step, n_pool)

    for r in range(max_rounds):
        # ① 采样聚焦: 配置恢复 (偏置平均占据) 生成当前基 det, 并入子空间。
        #    B1 增量采样: 每轮用池的前 n_cur 行 (shots 逐轮递增)。
        bsm_r = bsm[:n_cur] if shots_step > 0 else bsm
        probs_r = probs[:n_cur] if shots_step > 0 else probs
        # C1 尾部发现采样 hook (round_001 + round_002 预算缩放):
        #    默认全关 (tail_suppression=False) → block 完全跳过, 零行为变化。
        #    启用时 (tail_suppression=True): 用 discover_tail_pool 过抽 + 抑制已见
        #    det, 收集 "贡献新 α/β 字符串" 的位串替代本轮固定池。无新贡献 (恢复映像
        #    饱和) 则回退原 bsm_r/probs_r, 避免 ② 对角化饿死。distill 边界: 不读 c2d。
        #
        #    round_002 (C1-v2): n_tgt 随当前 shots 游标 n_cur 与 tail_shots_ref 之比
        #    线性缩放 (带 clip), 修复 round_001 "bootstrap 预算与 shots 解耦" 局限。
        #    tail_shots_ref>0 时覆盖 tail_n_target_per_round; =0 时走 round_001 路径。
        #    @100 shots + tail_shots_ref=100 → n_tgt=30 (= round_001) 逐位零回归。
        if tail_suppression:
            if tail_shots_ref > 0:              # C1-v2: shots 缩放 (round_002)
                n_tgt_raw = int(np.ceil(
                    n_active_per_round * n_cur / tail_shots_ref))
                n_tgt = max(n_active_per_round,
                            min(n_tgt_raw, 3 * n_active_per_round))
            else:                               # round_001 路径 (缩放关)
                n_tgt = tail_n_target_per_round or n_active_per_round
            # 每轮用 round 偏移种子推进 RNG → 每轮抽新随机位串 (固定种子会每轮
            # 恢复到同一批 det, 第 2 轮起全被抑制 → 退化)。rand_seed=None 时透传 None。
            round_seed = None if rand_seed is None else rand_seed + r + 1
            _bsm_new, _probs_new, _n_drawn = discover_tail_pool(
                (occ_a, occ_b), na, nb, norb,
                seen_a=set(int(s) for s in str_a),
                seen_b=set(int(s) for s in str_b),
                n_target_new=n_tgt, base_distribution="bootstrap",
                max_draw_factor=tail_max_draw_factor, rand_seed=round_seed,
            )
            if _bsm_new.shape[0] > 0:            # 有新贡献 → 用 C1 池替代本轮池
                bsm_r, probs_r = _bsm_new, _probs_new
            # else: 无新贡献 (恢复映像饱和) → 回退原 bsm_r/probs_r
            # _n_drawn 接住 (round_001 用 _ 丢弃): 诊断 = 本轮原始抽样数, 供
            # R5 P2 饱和度分析 (n_drawn/n_tgt → 恢复映像饱和度) 与 round_002 P1
            # 过抽自适应 (theory.md §1.3)。P0 不消费, 仅保留变量供后续扩展。
        rec, _ = recover_configurations(
            bsm_r, probs_r, (occ_a, occ_b), na, nb, rand_seed=rand_seed
        )
        ci_a, ci_b = bitstring_matrix_to_ci_strs(rec, open_shell=open_shell)
        n_before = len(str_a) + len(str_b)
        str_a = sorted(set(str_a) | set(int(x) for x in ci_a))
        str_b = sorted(set(str_a) if not open_shell else (set(str_b) | set(int(x) for x in ci_b)))
        n_sampled_new = len(str_a) + len(str_b) - n_before
        # 采样覆盖不受 max_strings 限制 (真实采样的 det 都应进子空间);
        # max_strings 只约束 PT2 扩展 (下方 ④), 与 solve_cipsi 语义一致。

        # ② 子空间对角化
        E, c2d, sa, sb = sub.diag(str_a, str_b)
        idx_a = {int(s): i for i, s in enumerate(sa)}
        idx_b = {int(s): i for i, s in enumerate(sb)}
        # 方向 D: 每轮默认 (无候选时方差/PT2 = 0)
        sigma2 = 0.0
        e_pt2_sum = 0.0

        # ③ 主导 dets
        nA, nB = c2d.shape
        flat = np.abs(c2d).ravel()
        order = np.argsort(flat)[::-1]
        dom = []
        for k in order:
            if flat[k] > dom_thresh:
                ia, ib = divmod(int(k), nB)
                dom.append((int(sa[ia]), int(sb[ib])))
            else:
                break

        # ④ 候选连接 → PT2 受限选态 (不补全全空间)
        cand = set()
        if dom:
            for a, b in dom:
                for ca, cb in _excited_dets(a, b, norb):
                    if ca not in idx_a or cb not in idx_b:
                        cand.add((ca, cb))
        if cand:
            me = sub.pt2_matrix_elements(str_a, str_b, cand, c2d, sa, sb)
            # 方向 D: σ² = Σ|⟨a|H|Ψ⟩|² (PT2 分子平方和, 对生成集的精确方差)
            sigma2 = sum(h * h for h, _ in me.values())
            pt2 = {d: h * h / (E - Ea) for d, (h, Ea) in me.items()
                   if abs(E - Ea) > 1e-12}
            e_pt2_sum = float(sum(pt2.values()))
            ranked = sorted(pt2.items(), key=lambda kv: -abs(kv[1]))
            add = []
            for d, v in ranked:
                if abs(v) < pt2_floor:
                    break
                if len(str_a) + len(add) >= max_strings:
                    break
                add.append(d)
            if len(add) > n_active_per_round:
                add = add[:n_active_per_round]
            for ca, cb in add:
                str_a.append(ca)
                if cb not in str_b:
                    str_b.append(cb)
            str_a = sorted(set(str_a))
            if open_shell:
                str_b = sorted(set(str_b))
            else:
                str_b = str_a
            n_pt2_new = len(add)
        else:
            n_pt2_new = 0

        # ⑤ 更新平均占据 (采样偏置): 解态 1-RDM 对角
        st = SCIState(amplitudes=c2d, ci_strs_a=np.asarray(sa),
                      ci_strs_b=np.asarray(sb), norb=norb, nelec=nelec)
        dm1 = st.rdm(rank=1, spin_summed=True)
        occ_a = np.clip(np.diag(dm1) / 2.0, 0.0, 1.0)
        occ_b = occ_a.copy()

        # 方向 D: 记录轨迹点 (E, σ², E_PT2, dim, shots) 供能量-方差外推
        if trajectory is not None:
            trajectory.append({
                "round": r + 1, "E": float(E), "sigma2": sigma2,
                "e_pt2": e_pt2_sum, "dim": len(str_a) * len(str_b),
                "shots": int(n_cur),
            })

        if verbose:
            print(f"[active r{r+1}/{max_rounds}] E={E + ecore:.6f} "
                  f"strings={len(str_a)}x{len(str_b)} "
                  f"sampled_new={n_sampled_new} pt2_new={n_pt2_new}")

        # 收敛: 无 PT2 新 det 且采样无新增 (子空间不再扩展) → 稳定
        if n_pt2_new == 0 and n_sampled_new == 0:
            break
        # PT2 贡献可忽略且能量稳定
        if n_pt2_new == 0 and abs(E - e_prev) < 1e-10:
            break
        # B1 预算闭环: 能量收敛停采 (ΔE < energy_tol → 已收敛, 省 shots)
        if energy_tol is not None and r > 0 and abs(E - e_prev) < energy_tol:
            break
        e_prev = E
        # B1 增量采样: 扩大下一轮使用的 shots
        if shots_step > 0:
            n_cur = min(n_cur + shots_step, n_pool)

    # ---- 方向 C: 三激发定向注入 (round_008); 默认全关零回归 ----
    # 收敛循环结束时子空间已被 S/D-from-dominant ④ + 采样 + tail 填满低激发阶;
    # 此处对全部已选字符串迭代单激发, 逐层 +1 激发阶, 补上 ④ 不可达的高阶字符串
    # (round_007 诊断的 ~16 缺失字符串)。单激发图连通 ⇒ 迭代到 fixpoint = 全空间。
    # 打分复用 pt2_matrix_elements 一次 contract_2e 得全部候选 <a|H|Ψ>, 按新字符串
    # 聚合 EN-PT2, 注入 top-N。默认 triple_injection=False 整块跳过 = 零回归。
    # 注入在 prune 之前 (剪枝会剪掉高阶字符串的父串, theory §1.4)。
    if triple_injection:
        E, c2d, sa, sb = sub.diag(str_a, str_b)  # 种子 c2d (当前最好波函数)
        set_a = set(int(x) for x in sa)
        set_b = set(int(x) for x in sb)
        for _ in range(norb):                    # 迭代护栏 (≤ norb 次爬阶)
            new_a = set()
            new_b = set()
            for s in set_a:
                new_a |= _single_excited_strings(s, norb) - set_a
            if open_shell:                       # 闭壳层 α/β 同集合, 单次生成即可
                for s in set_b:
                    new_b |= _single_excited_strings(s, norb) - set_b
            if not new_a and not new_b:
                break
            # 候选 det = 新字符串 × 现有对侧字符串的笛卡尔积 (字符串级, 非 det 级)
            cand = set()
            for ca in new_a:
                cand.update((ca, cb) for cb in set_b)
            for cb in new_b:
                cand.update((ca, cb) for ca in set_a)
            if not cand:
                break
            me = sub.pt2_matrix_elements(str_a, str_b, cand, c2d, sa, sb)
            agg = {}                             # 按新字符串聚合 EN-PT2 得分
            for (ca, cb), (h, Ea) in me.items():
                if abs(E - Ea) <= 1e-12:
                    continue
                key = ca if ca in new_a else cb  # 开壳层: 新串必属单一自旋扇区
                agg[key] = agg.get(key, 0.0) + h * h / (E - Ea)
            ranked = sorted(agg.items(), key=lambda kv: -abs(kv[1]))
            add = []
            for s_new, v in ranked:
                if abs(v) < pt2_floor:
                    break
                if n_triples_per_round > 0 and len(add) >= n_triples_per_round:
                    break
                if len(str_a) + len(add) >= max_strings:
                    break
                add.append(s_new)
            if not add:
                break
            for s_new in add:                    # 注入到所属自旋扇区
                if s_new in new_a:
                    str_a.append(s_new)
                else:
                    str_b.append(s_new)
            str_a = sorted(set(str_a))
            str_b = str_a if not open_shell else sorted(set(str_b))
            E, c2d, sa, sb = sub.diag(str_a, str_b)  # 补全后重对角化
            set_a = set(int(x) for x in sa)
            set_b = set(int(x) for x in sb)
        # 补全后的子空间交给下方最终 diag + prune + 最终轨迹点照旧处理

    E, c2d, sa, sb = sub.diag(str_a, str_b)

    # ---- 方向 B: PT2 排序剪枝 (round_007); 默认全关零回归 ----
    # 收敛后的最终子空间上, 用本征矢的字符串边际权重 (Σ|c|² 行/列和) 排序,
    # 每自旋保留 top ceil(prune_keep × n) 个字符串, 剪掉低权重尾再重对角化。
    # 被剪 det 移出子空间 → 自动进入下方最终轨迹点的 PT2 和式 (二阶回补, §1.2)。
    # prune_keep=1.0 (默认) 时本块整体跳过 → 与改动前逐位一致 (P0' 零回归)。
    if prune_keep < 1.0:
        nA, nB = c2d.shape
        w_a = (c2d ** 2).sum(axis=1)            # α 行和 (边际权重)
        if open_shell:
            w_b = (c2d ** 2).sum(axis=0)        # β 列和 (α/β 独立剪)
            ka = max(1, int(math.ceil(prune_keep * nA)))
            kb = max(1, int(math.ceil(prune_keep * nB)))
            keep_a = set(np.argsort(w_a)[::-1][:ka])
            keep_b = set(np.argsort(w_b)[::-1][:kb])
            str_a = sorted(x for i, x in enumerate(sa) if i in keep_a)
            str_b = sorted(x for i, x in enumerate(sb) if i in keep_b)
        else:
            # 闭壳层: 合并权重 (行和 + 列和) 剪同一集合 → str_a == str_b 不变式
            w = w_a + (c2d ** 2).sum(axis=0)
            k = max(1, int(math.ceil(prune_keep * nA)))
            keep = set(np.argsort(w)[::-1][:k])
            str_a = str_b = sorted(x for i, x in enumerate(sa) if i in keep)
        E, c2d, sa, sb = sub.diag(str_a, str_b)  # 剪后重对角化 (子空间更小)

    # 方向 D: 最终对角化点也进轨迹 (剪枝后 = 最终保留子空间; 方差/PT2 在剪后
    # 子空间上重算 → 剪枝时 |E_PT2| 增大是预期正信号, 非回归)
    if trajectory is not None:
        idx_a = {int(s): i for i, s in enumerate(sa)}
        idx_b = {int(s): i for i, s in enumerate(sb)}
        nA, nB = c2d.shape
        flat = np.abs(c2d).ravel()
        order = np.argsort(flat)[::-1]
        dom = []
        for k in order:
            if flat[k] > dom_thresh:
                ia, ib = divmod(int(k), nB)
                dom.append((int(sa[ia]), int(sb[ib])))
            else:
                break
        cand_all = set()
        for a, b in dom:
            for ca, cb in _excited_dets(a, b, norb):
                if ca not in idx_a or cb not in idx_b:
                    cand_all.add((ca, cb))
        if cand_all:
            me = sub.pt2_matrix_elements(str_a, str_b, cand_all, c2d, sa, sb)
            sigma2 = sum(h * h for h, _ in me.values())
            e_pt2_sum = sum(h * h / (E - Ea) for h, Ea in me.values()
                            if abs(E - Ea) > 1e-12)
        else:
            sigma2, e_pt2_sum = 0.0, 0.0
        trajectory.append({
            "round": -1, "E": float(E), "sigma2": sigma2,
            "e_pt2": float(e_pt2_sum), "dim": len(str_a) * len(str_b),
            "shots": int(n_cur),
        })

    if state_out is not None:
        state_out.append((np.asarray(c2d), np.asarray(sa), np.asarray(sb)))
    if usage is not None:
        usage.append(int(n_cur))
    return float(E) + ecore


# --------------------------------------------------------------------------- #
#  方向 D: 能量-方差外推 (不增大维度降误差) + 本征矢重要性采样 (学习型采样先验)
# --------------------------------------------------------------------------- #
def solve_sqd_ev(
    one_body_tensor: np.ndarray,
    two_body_tensor: np.ndarray,
    norb: int,
    nelec: Tuple[int, int],
    *,
    bitstring_matrix: np.ndarray,
    probabilities: Optional[np.ndarray] = None,
    avg_occupancies: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    max_strings: Optional[int] = None,
    n_active_per_round: int = 50,
    dom_thresh: float = 1e-3,
    pt2_floor: float = 1e-7,
    max_rounds: int = 10,
    correction: str = "pt2",
    degree: int = 1,
    ecore: float = 0.0,
    rand_seed: Optional[int] = 0,
    verbose: bool = False,
    shots_budget: Optional[int] = None,
    shots_step: int = 0,
    energy_tol: Optional[float] = None,
    return_details: bool = False,
    # ---- C1 尾部发现采样 (round_001): 透传给 solve_sqd_active ----
    tail_suppression: bool = False,
    tail_max_draw_factor: int = 10,
    tail_n_target_per_round: int = 0,
    # ---- C1 预算随 shots 缩放 (round_002): 透传 ----
    tail_shots_ref: int = 0,
    # ---- 方向 B: PT2 排序剪枝 (round_007): 透传 ----
    prune_keep: float = 1.0,
    # ---- 方向 C: 三激发定向注入 (round_008): 透传 ----
    triple_injection: bool = False,
    n_triples_per_round: int = 0,
    # ---- round_010: warm-start v0 透传 (默认关零回归) ----
    warm_start: bool = False,
    backend: str = "cpu",
) -> float:
    """改进 SQD (方向 D/③): active 采样 + 基于方差的能量修正, 不增大维度降误差。

    **三种修正 (都用 PT2 分子 Σ|⟨a|H|Ψ⟩|² = 精确方差, 纯经典后处理)**:
      - ``correction="pt2"`` (**默认, 推荐**): ``E + E_PT2``, 其中
        ``E_PT2 = Σ_a |⟨a|H|Ψ⟩|²/(E−E_a)`` (Epstein-Nesbet)。SHCI/CIPSI 标准修正,
        **行为良好**——N₂/STO-3G 受限子空间直接 err 4.3e-4 → 6.2e-5; C₂ 直接
        err 7.9e-3 (超化学精度) → **5.0e-4** (达化学精度)。
      - ``correction="evpt2"`` (**方向③, 备选**): ``E_V`` vs ``E_PT2`` 两点外推
        (:func:`extrapolate_ev_pt2`, SHCI 社区 Holmes 2016/Sharma 2017 标准)。
        用轨迹各轮 ``(E_V, E_PT2)`` 拟合线性外推到 ``E_PT2→0``。x 轴是带能量分母
        加权的 ``E_PT2`` (物理上更接近漏掉的关联能), 经验**不过冲**——优于 σ² 线性。
        需轨迹 ≥2 个 ``E_PT2`` 非零点; 子空间饱和 (``E_PT2≈0``) 时退化为直接能量。
      - ``correction="ev"`` (**诊断用**): 用轨迹 ``(E, σ²)`` 线性外推到 σ²=0
        (:func:`extrapolate_energy_variance`)。**注意: 实测会过冲到 FCI 之下**
        (N₂ −5.8e-4, C₂ −1.7e-2), 不推荐作为默认——保留作方差标度诊断。

    **动机**: :func:`solve_sqd_active` 的最终子空间能量是 FCI 的变分上界
    (残余误差 ∝ 漏掉 det 的方差)。PT2/σ² 修正都用已算的候选矩阵元估计漏掉
    的关联, **不增大最终子空间维度**即降误差。饱和子空间 (全空间, σ²≈0)
    时修正自然趋零。

    Parameters
    ----------
    其余参数与 :func:`solve_sqd_active` 一致 (``ecore`` 在返回/诊断中计入;
    轨迹内部不含 ecore)。``prune_keep`` (方向 B, round_007) 透传至 active 的
    最终子空间剪枝 (默认 ``1.0`` = 不剪枝零回归; ``<1.0`` 剪低权重字符串后
    重对角化, 被剪 det 的关联进入 ``E_PT2`` 回补)。``triple_injection`` /
    ``n_triples_per_round`` (方向 C, round_008) 透传至 active 的末轮三激发
    定向注入 (默认关零回归)。
    correction : {"pt2", "evpt2", "ev"}
        修正方式 (见上)。``"pt2"`` = E+E_PT2 (推荐); ``"evpt2"`` = E_V vs E_PT2
        两点外推 (方向③, 不过冲); ``"ev"`` = σ² 线性外推 (诊断, 可能过冲)。
    degree : int
        ``correction="evpt2"`` / ``"ev"`` 时外推多项式次数 (默认 1 = 线性)。
    return_details : bool
        ``True`` 返回 ``(能量, details_dict)``; ``details_dict`` 含
        ``E_direct`` (active 直接能量)、``correction``、``E_PT2`` (pt2 模式)、
        ``e_inf``/``alpha``(evpt2) 或 ``slope``(ev)/``r2``/``fit_std``、``trajectory``
        (每轮 E/σ²/PT2/dim/shots)。

    Returns
    -------
    float | tuple
        修正后能量 (含 ``ecore``); ``return_details=True`` 时返回 ``(能量, dict)``。
    """
    if correction not in ("pt2", "ev", "evpt2"):
        raise ValueError(f"correction 须为 'pt2' / 'ev' / 'evpt2', got {correction!r}.")
    trajectory: list = []
    solve_sqd_active(
        one_body_tensor, two_body_tensor, norb, nelec,
        bitstring_matrix=bitstring_matrix, probabilities=probabilities,
        avg_occupancies=avg_occupancies, max_strings=max_strings,
        n_active_per_round=n_active_per_round, dom_thresh=dom_thresh,
        pt2_floor=pt2_floor, max_rounds=max_rounds,
        ecore=0.0,                       # 轨迹 E 不含 ecore, 修正后统一加
        rand_seed=rand_seed, verbose=verbose,
        shots_budget=shots_budget, shots_step=shots_step,
        energy_tol=energy_tol, trajectory=trajectory,
        tail_suppression=tail_suppression,
        tail_max_draw_factor=tail_max_draw_factor,
        tail_n_target_per_round=tail_n_target_per_round,
        tail_shots_ref=tail_shots_ref,
        prune_keep=prune_keep,
        triple_injection=triple_injection,
        n_triples_per_round=n_triples_per_round,
        warm_start=warm_start,
        backend=backend,
    )
    if len(trajectory) < 2:
        raise ValueError(f"轨迹点不足 (<2), 无法修正: got {len(trajectory)}.")
    last = trajectory[-1]
    E = float(last["E"])
    e_direct = E + ecore                 # active 直接能量 (最终子空间)
    dim = int(last["dim"])

    if correction == "pt2":
        # E + E_PT2 (Epstein-Nesbet, 行为良好)
        e_pt2 = float(last["e_pt2"])
        e_corr = E + e_pt2 + ecore
        details = {
            "E_direct": e_direct, "correction": "pt2", "E_PT2": e_pt2,
            "dim": dim, "trajectory": trajectory,
        }
        if verbose:
            print(f"[EV:pt2] dim={dim} E_direct={e_direct:.8f} "
                  f"E_PT2={e_pt2:.2e} E_corr={e_corr:.8f}")
    elif correction == "ev":
        # σ² 线性外推 (诊断; 实测会过冲到 FCI 之下)
        es = np.asarray([t["E"] for t in trajectory], dtype=np.float64)
        vs = np.asarray([t["sigma2"] for t in trajectory], dtype=np.float64)
        if np.max(vs) < 1e-14:
            # 子空间饱和: 无残余可外推, 退化为直接能量
            e_corr = e_direct
            e_inf, slope, r2, fit_std = e_direct, 0.0, 1.0, 0.0
        else:
            e_inf, slope, r2, fit_std = extrapolate_energy_variance(
                es, vs, degree=degree)
            e_corr = float(e_inf) + ecore
        details = {
            "E_direct": e_direct, "correction": "ev", "e_inf": float(e_inf),
            "slope": slope, "r2": r2, "fit_std": fit_std, "dim": dim,
            "trajectory": trajectory,
        }
        if verbose:
            print(f"[EV:ev] dim={dim} r²={r2:.4f} E_direct={e_direct:.8f} "
                  f"E_ev={e_corr:.8f} (非变分, 可能过冲)")
    else:
        # E_V vs E_PT2 两点外推 (SHCI 标准, 方向③; 经验不过冲, 优于 σ² 线性)。
        # 用轨迹各轮 (E_V 变分能量, E_PT2 Epstein-Nesbet) 外推到 E_PT2→0。
        # **稳健性护栏**: solve_sqd_active 的 within-run 轨迹常退化 (受限时 round 间
        # E_PT2 重复, 或子空间饱和后 E_PT2≈0) —— 互异点 <2 时拟合病态 (alpha 爆炸),
        # 此时退化为 pt2 单点修正 (evpt2 永不劣于 pt2)。需稳健两点外推请用两次不同
        # max_strings 跑 solve_sqd_active, 再喂 :func:`extrapolate_ev_pt2`。
        es = np.asarray([t["E"] for t in trajectory], dtype=np.float64)
        pts = np.asarray([t["e_pt2"] for t in trajectory], dtype=np.float64)
        n_distinct = len(np.unique(np.round(pts, decimals=14)))
        if n_distinct < 2 or np.max(np.abs(pts)) < 1e-14:
            # 轨迹退化: 退化为 pt2 单点修正 (E + E_PT2)
            e_pt2_val = float(last["e_pt2"])
            e_corr = E + e_pt2_val + ecore
            details = {
                "E_direct": e_direct, "correction": "evpt2", "fallback": "pt2",
                "E_PT2": e_pt2_val, "dim": dim, "trajectory": trajectory,
                "note": "轨迹 E_PT2 互异点 <2 (受限/饱和), 外推病态, 退化为 pt2",
            }
            if verbose:
                print(f"[EV:evpt2→pt2 fallback] dim={dim} E_direct={e_direct:.8f} "
                      f"E_PT2={e_pt2_val:.2e} E_corr={e_corr:.8f}")
        else:
            e_inf, alpha, r2, fit_std = extrapolate_ev_pt2(es, pts, degree=degree)
            e_corr = float(e_inf) + ecore
            details = {
                "E_direct": e_direct, "correction": "evpt2", "fallback": None,
                "e_inf": float(e_inf), "alpha": alpha, "r2": r2,
                "fit_std": fit_std, "dim": dim, "trajectory": trajectory,
            }
            if verbose:
                print(f"[EV:evpt2] dim={dim} r²={r2:.4f} E_direct={e_direct:.8f} "
                      f"E_evpt2={e_corr:.8f}")
    if return_details:
        return e_corr, details
    return e_corr


def solve_sqd_distill(
    one_body_tensor: np.ndarray,
    two_body_tensor: np.ndarray,
    norb: int,
    nelec: Tuple[int, int],
    *,
    bitstring_matrix: np.ndarray,
    probabilities: Optional[np.ndarray] = None,
    n_rounds: int = 3,
    n_samples: Optional[int] = None,
    temperature_schedule: Optional[Sequence[float]] = None,
    max_strings: Optional[int] = None,
    n_active_per_round: int = 50,
    ecore: float = 0.0,
    rand_seed: Optional[int] = 0,
    keep_pool: bool = True,
    verbose: bool = False,
    backend: str = "cpu",
) -> float:
    """自蒸馏 SQD 闭环 (方向②): solve → 按 |c|^(2/T) 重采 → recover → solve。

    **思路 (库 TODO "solve_sqd_distill 蒸馏闭环" 落地)**: 子空间对角化的本征矢
    ``|Ψ⟩ = Σ_i c_i |i⟩`` 是体系当前最好的"波函数模型"。每轮用它驱动一次**重要性
    重采样** (:func:`eigenvector_importance_sample`, 按 ``p_i ∝ |c_i|^(2/T)`` 采 det),
    再喂回 :func:`solve_sqd_active`。这是 **EM 式量子-经典反馈**: E 步用当前波函数
    采, M 步重对角化。同 shots 下子空间对**主导 det 流形**覆盖更密 → 变分下界更低;
    或同精度省 shots。抗噪: 噪声 det 的 ``|c|²`` 自然小, 重采时淘汰 (自清洗)。

    **温度退火** ``temperature_schedule`` (高→低): 高温 ``T>1`` (``|c|^(2/T)`` 更平)
    保持探索, 低温 ``T<1`` (更锐) 聚焦主导 det。默认 ``[1.5]*(n_rounds-2) + [0.5]``
    (长度 ``n_rounds-1``, 最后一轮不重采; 前几轮探索, 倒数第二轮锐化)。

    Parameters
    ----------
    one_body_tensor, two_body_tensor, norb, nelec, ecore
        分子积分 (闭壳层单 h1e) + 电子数 + core 偏移。
    bitstring_matrix : ndarray (S, 2*norb)
        初始采样位串 (电路 shot 或随机种子)。第 0 轮的采样池。
    probabilities : ndarray | None
        初始概率 (第 0 轮); 省略均匀。后续轮重采位串用均匀 (来自 |c|²)。
    n_rounds : int
        solve→重采 循环次数 (≥1)。``n_rounds=1`` 退化为单次 :func:`solve_sqd_active`。
    n_samples : int | None
        每轮重采的 det 数; ``None`` = 用 ``bitstring_matrix`` 行数。
    temperature_schedule : Sequence[float] | None
        长度 ``n_rounds-1`` 的温度列表 (最后一轮不重采); ``None`` = 默认退火。
    max_strings, n_active_per_round
        透传 :func:`solve_sqd_active`。
    rand_seed : int | None
        第 0 轮配置恢复种子; 后续轮自动 +1 (避免重复采样序列)。
    keep_pool : bool
        ``True`` (默认): 每轮采样池 = ``vstack(初始 bsm, 重采 bsm)`` (不丢失原始
        电路覆盖, 仅聚焦增强); ``False``: 池 = 重采 bsm (纯蒸馏聚焦, 替换)。
    verbose : bool
        打印每轮能量 / 稀疏度。

    Returns
    -------
    float
        所有轮中**最低**的 active 能量 (含 ``ecore``)。变分保证 best_E 单调不增于
        各轮, 但因每轮采样池变化, 取 min 最稳。

    Notes
    -----
    - 依赖 :func:`eigenvector_importance_sample` (已修 F1 α/β 半区布局), 开壳层
      (na≠nb) 安全。
    - 与 :func:`solve_sqd_active` 的关系: 后者是单次"采样↔PT2 选态"闭环; 本函数
      在其外再套一层"解态驱动的采样分布更新"。可叠加 (内部仍跑 active)。
    - NQS 衔接 (research): 把"按 |c|² 采"升级为神经网络参数化 ``p_θ(det)`` 泛化到
      未采 det, 是本闭环的深度学习版 (见 REVIEW Part 2 B1)。
    """
    if one_body_tensor.ndim == 3:
        if not np.allclose(one_body_tensor[0], one_body_tensor[1]):
            raise ValueError(
                "solve_sqd_distill 不支持自旋分辨 h1e; 请传闭壳层 (norb, norb)。"
            )
        h1e = np.asarray(one_body_tensor[0])
    else:
        h1e = np.asarray(one_body_tensor)
    eri = np.asarray(two_body_tensor)

    bsm0 = np.asarray(bitstring_matrix, dtype=bool)
    if bsm0.ndim != 2 or bsm0.shape[1] != 2 * norb:
        raise ValueError(
            f"bitstring_matrix must have shape (S, 2*norb={2*norb}), got {bsm0.shape}."
        )
    if n_rounds < 1:
        raise ValueError(f"n_rounds must be >= 1, got {n_rounds}.")
    if n_samples is None:
        n_samples = bsm0.shape[0]
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}.")
    if temperature_schedule is None:
        # 长度 n_rounds-1 (最后一轮不重采); 前 n_rounds-2 轮高温探索, 倒数第二轮 0.5 锐化
        temperature_schedule = ([1.5] * max(n_rounds - 2, 0) + [0.5]) if n_rounds >= 2 else []
    # 最后一轮不重采 → schedule 长度应为 n_rounds-1
    if len(temperature_schedule) != max(n_rounds - 1, 0):
        raise ValueError(
            f"temperature_schedule 长度须为 n_rounds-1={max(n_rounds-1,0)}, "
            f"got {len(temperature_schedule)}。"
        )

    pool = bsm0
    pool_probs = (probabilities if probabilities is not None
                  else np.full(bsm0.shape[0], 1.0 / bsm0.shape[0]))
    best_E = np.inf
    cur_seed = rand_seed

    for r in range(n_rounds):
        state_out: list = []
        E = solve_sqd_active(
            h1e, eri, norb, nelec,
            bitstring_matrix=pool, probabilities=pool_probs,
            max_strings=max_strings, n_active_per_round=n_active_per_round,
            ecore=ecore, rand_seed=cur_seed, state_out=state_out,
            backend=backend,
        )
        c2d, sa, sb = state_out[0]
        if E < best_E:
            best_E = E
        if verbose:
            pmax = float(np.abs(np.asarray(c2d)).max() ** 2)
            print(f"[distill r{r+1}/{n_rounds}] E={E:.8f} pool={pool.shape[0]} "
                  f"|c|max²={pmax:.4f}")
        if r == n_rounds - 1:
            break
        # 解态驱动重要性重采 (温度退火)
        T = temperature_schedule[r]
        new_bsm = eigenvector_importance_sample(
            c2d, sa, sb, norb, n_samples, rand_seed=cur_seed, temperature=T)
        if keep_pool:
            pool = np.vstack([bsm0, new_bsm])
            pool_probs = np.full(pool.shape[0], 1.0 / pool.shape[0])
        else:
            pool = new_bsm
            pool_probs = np.full(n_samples, 1.0 / n_samples)
        cur_seed = (cur_seed or 0) + 1

    return float(best_E)


def eigenvector_importance_sample(
    c2d: np.ndarray,
    sa: np.ndarray,
    sb: np.ndarray,
    norb: int,
    n_shots: int,
    *,
    rand_seed: Optional[int] = 0,
    temperature: float = 1.0,
) -> np.ndarray:
    """本征矢重要性采样 (方向 D, 学习型采样先验): 按振幅平方 ∝c² 采样 det 位串。

    **思路**: 子空间对角化解出的本征矢 ``|Ψ⟩ = Σ_i c_i |i⟩`` 是体系当前最好
    的"波函数模型" (数据驱动先验)。按其振幅平方分布 ``p_i ∝ |c_i|²`` 重新
    采样 ``n_shots`` 个 det 位串 —— 高权重 det 被更多采样, 低权重 det 少量
    覆盖 —— 相比均匀/随机采样, 同 shots 下配置恢复更聚焦高价值 det, 子空间
    质量更高 (同维度误差更低)。

    **与 AI 方法衔接**: 这是"学习型采样分布"的最简实现 (从解态学分布)。更强
    版本可用神经网络/NQS 参数化 ``p_i`` 泛化到未采样 det (见 REVIEW 方向 D
    展望), 本函数是确定性、可验证的基线。

    Parameters
    ----------
    c2d : ndarray, shape (nA, nB)
        子空间对角化本征矢 (α × β 字符串网格振幅)。
    sa, sb : ndarray, shape (nA,) / (nB,)
        对应 α/β 字符串 (整数表示)。
    norb : int
        空间轨道数 (位串宽度)。
    n_shots : int
        采样 det 数。
    rand_seed : int | None
        随机种子。
    temperature : float
        分布锐度: ``p_i ∝ |c_i|^(2/temperature)``。``1.0`` = 原始振幅平方;
        ``<1`` 更锐 (只采主导 det), ``>1`` 更平 (更像均匀)。

    Returns
    -------
    ndarray, shape (n_shots, 2*norb)
        采样位串矩阵, 遵循库统一布局 ``[β_{n-1}..β_0 | α_{n-1}..α_0]``
        (左 norb 列 β = ``det_b``, 右 norb 列 α = ``det_a``)。开壳层消费者
        直接喂 ``bitstring_matrix_to_ci_strs(open_shell=True)`` 可还原 (α, β)。
    """
    from .counts import int_to_bitarray

    c2d = np.asarray(c2d)
    sa = np.asarray(sa)
    sb = np.asarray(sb)
    probs = np.abs(c2d) ** (2.0 / temperature)
    probs = probs.ravel()
    denom = probs.sum()
    if denom <= 0:
        raise ValueError("本征矢振幅全零, 无法采样。")
    probs = probs / denom
    rng = np.random.default_rng(rand_seed)
    idx = rng.choice(probs.size, size=n_shots, replace=True, p=probs)
    ia, ib = np.divmod(idx, c2d.shape[1])
    det_a = sa[ia]
    det_b = sb[ib]
    # 库比特串布局 [β_{n-1}..β_0 | α_{n-1}..α_0] (左 β 右 α, 见 counts.py /
    # fermion._det_to_bitstring / integrated carryover)。det_a=α → 右半,
    # det_b=β → 左半; int_to_bitarray 半内顺序 [orb_{n-1}..orb_0] 与约定一致。
    bsm = np.zeros((n_shots, 2 * norb), dtype=bool)
    bsm[:, :norb] = int_to_bitarray(det_b, norb)
    bsm[:, norb:] = int_to_bitarray(det_a, norb)
    return bsm
