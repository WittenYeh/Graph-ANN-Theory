# CLAUDE.md

本仓库是关于 **Graph-based Approximate Nearest Neighbor (ANN) Search 理论**的学习笔记，主要跟随 Indyk & Xu [NeurIPS 2023] 与 Gollapudi et al. [ICML 2025] 等工作，对 DiskANN 系列图的构建、规模、查询时间与近似精度进行形式化分析。

## 目录结构

- `notes/`：Markdown 笔记（`note1` ~ `noteN`），按主题递进。
- `paper-pdfs/`：参考论文 PDF。
- `resources/noteN/`：第 `N` 篇笔记引用的所有图片素材（算法伪代码图、示意图等）。**每篇笔记的素材放在与之同名的子目录下。**

笔记中引用图片时，使用从 `notes/` 出发的相对路径，例如 `![...](../resources/note4/beamsearch-algorithm.png)`。

## 符号约定 (Notation Convention)

全仓库笔记**统一**采用以下记号。注意这与部分原始论文（尤其 Gollapudi et al.）的记号**有冲突**，引用原文定理时需做转换并显式注明。

| 符号 | 含义 | 备注 |
|---|---|---|
| $(\mathcal{M}, D)$ | 度量空间与距离函数 | |
| $P \subseteq \mathcal{M}$ | 数据点集，$n = \lvert P \rvert$ | |
| $q \in \mathcal{M}$ | 查询点 | |
| $p^*$ | $q$ 的精确最近邻 (exact NN)；即第 1 近邻 $b^*_1$ | ⚠️ Gollapudi et al. 记作 $a$ |
| $b^*_j$ | $q$ 的**第 $j$ 个真实最近邻** (true $j$-th NN) | ⚠️ Gollapudi et al. 记作 $a_j$ |
| $(1+\epsilon)$-ANN | $(1+\epsilon)$-近似最近邻 | |
| $N_{\mathrm{out}}(p)$ | 点 $p$ 的出邻居集合 | |
| $\lambda$ | **Doubling Dimension（倍增维数）** | ⚠️ Gollapudi et al. 用 $\Delta$ 表示此量 |
| $\Delta$ | **Aspect Ratio（纵横比）** $= D_{\max}/D_{\min}$ | ⚠️ Gollapudi et al. 用 $\delta$ 表示此量 |
| $\alpha$ | Prune / reachability 参数 | 常取 $\alpha = 1 + 4/\epsilon$ |
| $G_{\mathrm{DA}}$ | DiskANN 构建出的 Proximity Graph | |
| $L$ | beam width / 搜索列表大小 | beam search 中 $L \ge k$ |
| $p$ | greedy 终止的**局部最优点**（亦泛指搜索中的当前点） | beam=1 时分析的对象 |
| $p^{\circ}$ | greedy / beam search 每步取出的**当前扩展点**（离 $q$ 最近且未扩展） | NOTE 1 greedy 与 NOTE 4 BeamSearch 共用 |
| $\hat{b}_j$ | beam search 返回的**第 $j$ 个近似结果** (approx)，按到 $q$ 距离升序 | 局部最优束 $B_L=\{\hat{b}_1,\dots,\hat{b}_L\}$ |
| $\rho_{\mathrm{opt}}$ | 最坏情形**近似比**（= 局部最优分析中优化问题的最优值） | ⚠️ Gollapudi et al. 记作 $\alpha_{\mathrm{opt}}$ |

> **关键冲突提醒：** Gollapudi et al. [ICML 2025] 用 $\Delta$ 表示 Doubling Dimension、$\delta$ 表示 Aspect Ratio，与本仓库约定正好相反。撰写或修改笔记时一律使用本表约定（$\lambda$=Doubling Dimension，$\Delta$=Aspect Ratio），仅在直接引用原文定理时注明对应关系。
>
> **搜索阶段记号（NOTE 1/3/4）：** 本仓库统一用 $p$=局部最优点、$p^{\circ}$=当前扩展点、$p^*$=精确最近邻、$b^*_j$=第 $j$ 个真实最近邻、$\hat{b}_j$=beam search 第 $j$ 个近似结果、$\rho_{\mathrm{opt}}$=近似比。其中 $p^*/b^*_j/\rho_{\mathrm{opt}}$ 对应 Gollapudi et al. 原文的 $a/a_j/\alpha_{\mathrm{opt}}$，引用原文时需转换。
>
> **建图阶段的局部重载：** 在 **Prune 算法（伪代码图内部）** 中，$p$=被剪枝的中心点、$p^*$=当前选中的最近候选、$p'$=被测试的候选——这是原论文 Prune 的记号，仅限该算法图内部，与上面搜索阶段的 $p$/$p^*$ 含义不同（建图 vs 搜索，上下文分离，不混用）。

## 伪代码插图工作流 (Pseudocode → Image Workflow)

笔记正文中的**所有伪代码（算法）一律以图片形式插入**，不使用 Markdown 代码块（` ``` `）直接排版伪代码。生成图片遵循固定流程：

1. **用 LaTeX 写伪代码**：在 `resources/noteN/<algorithm-name>.tex` 中编写。使用 `standalone` 文档类 + `algorithm2e` 宏包，以得到紧凑、自动裁剪边距的输出。模板：

   ```latex
   \documentclass[border=8pt]{standalone}
   \usepackage[ruled,vlined,linesnumbered]{algorithm2e}
   \usepackage{amsmath,amssymb}
   \usepackage{varwidth}
   \begin{document}
   \begin{varwidth}{\linewidth}
   % 若需指定算法编号（如 Algorithm 2），在此处加：\setcounter{algocf}{1}
   \begin{algorithm}[H]
   \SetAlgoLined
   \DontPrintSemicolon
   \caption{$\textsc{AlgoName}(\dots)$}
   \KwIn{...}
   \KwOut{...}
   % ... 算法主体 ...
   \end{algorithm}
   \end{varwidth}
   \end{document}
   ```

2. **PDF 转图片**：在 `resources/noteN/` 目录下执行

   ```sh
   pdflatex -interaction=nonstopmode -halt-on-error <name>.tex
   pdftoppm -png -r 300 <name>.pdf <name>      # 生成 <name>-1.png
   mv <name>-1.png <name>.png
   rm -f *.aux *.log                            # 清理中间文件
   ```

   （本环境无 `convert`/`magick`，使用 `pdftoppm` 完成 PDF→PNG；分辨率 `-r 300` 保证清晰度。）

3. **插入正文**：在笔记中以 Markdown 图片语法引用：
   `![Algorithm N: AlgoName](../resources/noteN/<name>.png)`

4. **保留产物**：`<name>.tex`、`<name>.pdf`、`<name>.png` 三个文件都保存在 `resources/noteN/` 下（`.tex` 便于后续修改重生成，`.aux`/`.log` 等中间文件删除）。

## 笔记写作风格

- 中文为主，关键术语保留英文。
- 分级编号：`(N.x)` / `(N.x.y)`；定理/引理用**加粗**命名。
- 数学公式用 `$...$`（行内）与 `$$...$$`（行间）。
- 用 `>` 引用块承载「直觉 / 备注 / 提示」。
- 证明以 `$\square$` 收尾；文末附 `Reference`。
- 新笔记应显式与前序笔记建立衔接（回顾、对照结论）。
