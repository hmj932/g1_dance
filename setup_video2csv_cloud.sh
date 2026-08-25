#!/usr/bin/env bash
# setup_video2csv_cloud.sh
# 在一台全新 Linux GPU 盒（单卡 4090/4090D 24G，PyTorch2.3+cu121+py310 镜像即可，无需预装）
# 上把「视频 → CSV」桥跑起来。复用仓内 GVHMR + gvhmr_to_csv.py，不写新算法。
#
# Cursor 评审 5 处修订已纳入：
#   1) 本地只验到 CSV；csv_to_npz.py（需 isaaclab）不在本盒跑，npz/训练走 flux
#   2) 装 env 直接走盒上 egress（torch cu121 / pytorch3d / smplx-git / pycolmap）
#   3) smoke：patch 掉 demo.py 的 render_incam/render_global（6GB 才需；24G 其实也省时间）
#   4) pycolmap 随 requirements.txt 装（仅免 DPVO，SimpleVO 顶层仍 import pycolmap）
#   5) 训练验收 = Isaac npz(*_isaac.npz) + flux entry_train_and_play.py
#
# 用法（推荐）：
#   # 1) 只 rsync g1_dance 本体（121M，排除两个依赖，省得传 1.5G 的 GMR）：
#   rsync -av --exclude='.git' --exclude='GVHMR' --exclude='GMR' \
#     ~/humanoid_ws/G1/Beyondminic-Weilai-G1/ box:/root/Beyondminic-Weilai-G1/
#   # 2) 盒上一键（脚本自动从 .gitmodules 的 URL shallow clone 缺失的 GVHMR/GMR）：
#   G1_ROOT=/root/Beyondminic-Weilai-G1 bash setup_video2csv_cloud.sh
#   # 若 g1_dance 是私有仓，clone 它本身要你自己的 git 凭证；rsync 则不需要。
#   # 云盒若 github 不通，先 git config --global http.proxy <平台学术加速地址>。
#
# 可选环境变量：
#   G1_ROOT   仓根（默认 /root/Beyondminic-Weilai-G1）
#   CONDA     conda 可执行（默认 conda）
#   MIRROR    1=用国内镜像(HF_ENDPOINT=hf-mirror + 阿里云 pip / 清华 conda)（默认，适用国内盒）；
#             0=直连（适用海外盒或平台已内置学术加速）
#   STAGE     all|env|weights|smoke|status（默认 all）。失败后重跑 all 会自动跳过已完成步骤、从断点续；status 只打印进度不干活
#   VIDEO     D 步推理的视频（相对 GVHMR 目录，默认 docs/example_video/tennis.mp4）。换舞蹈视频：VIDEO=docs/example_video/你的.mp4
set -euo pipefail

G1_ROOT="${G1_ROOT:-/root/Beyondminic-Weilai-G1}"
CONDA="${CONDA:-conda}"
MIRROR="${MIRROR:-1}"
STAGE="${STAGE:-all}"
STAGING="${STAGING:-/home/Downloads}"   # 你上传权重/SMPL 的目录；盒子 hf 不通时从这里 cp 兜底
VIDEO="${VIDEO:-docs/example_video/tennis.mp4}"   # D 步推理的视频（相对 GVHMR 目录）
STEM="${STEM:-$(basename "$VIDEO" .mp4)}"          # 视频名 → .pt 目录名 + csv 名

# ---- 镜像开关（仅本脚本进程内生效，不写全局 pip config / ~/.condarc）----
if [ "$MIRROR" = "1" ]; then
  PIP_INDEX=(-i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com)
  export HF_ENDPOINT=https://hf-mirror.com
  CONDA_MAIN=(-c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main --override-channels)
  CONDA_FORGE=(-c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge --override-channels)
else
  PIP_INDEX=()
  CONDA_MAIN=()
  CONDA_FORGE=(-c conda-forge)
fi

log(){ printf '\n\033[1;34m▶ %s\033[0m\n' "$*"; }
die(){ printf '\033[1;31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

# ---- 完成度探测（断点续跑：重跑 STAGE=all 时跳过已完成、从第一个未完成处续）----
gvhmr_env_ok(){ "$CONDA" env list 2>/dev/null | awk '{print $1}' | grep -qx gvhmr \
  && "$CONDA" run -n gvhmr python -c "import hmr4d,pytorch3d" >/dev/null 2>&1; }
gmr_env_ok(){ "$CONDA" env list 2>/dev/null | awk '{print $1}' | grep -qx gmr \
  && "$CONDA" run -n gmr python -c "import general_motion_retargeting,mujoco,mink" >/dev/null 2>&1; }
weights_ok(){ local f; for f in gvhmr/gvhmr_siga24_release.ckpt hmr2/epoch=10-step=25000.ckpt \
             vitpose/vitpose-h-multi-coco.pth yolo/yolov8x.pt; do
    [ -f "$G1_ROOT/GVHMR/inputs/checkpoints/$f" ] || return 1; done; }
smpl_ok(){ [ -f "$G1_ROOT/GVHMR/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz" ] \
        && [ -f "$G1_ROOT/GMR/assets/body_models/smplx/SMPLX_NEUTRAL.npz" ] \
        && [ -f "$G1_ROOT/GMR/assets/body_models/smplx/SMPLX_NEUTRAL.pkl" ]; }
pt_ok(){ [ -f "$G1_ROOT/GVHMR/outputs/demo/$STEM/hmr4d_results.pt" ] \
       || [ -n "$(find "$G1_ROOT/GVHMR/outputs" -name hmr4d_results.pt -path "*$STEM*" 2>/dev/null | head -1 || true)" ]; }
csv_ok(){ [ -f "$G1_ROOT/whole_body_tracking/motions/csv/$STEM.csv" ]; }
# 用法：rep "标签" <判定命令>
rep(){ local lbl="$1"; shift; if "$@" >/dev/null 2>&1; then printf "  \033[32m✓\033[0m %s\n" "$lbl"; else printf "  \033[31m✗\033[0m %s\n" "$lbl"; fi; }
progress_report(){
  log "进度检查（断点续跑依据）"
  rep "F  依赖 GVHMR 仓"        test -d "$G1_ROOT/GVHMR/.git"
  rep "F  依赖 GMR 仓"          test -d "$G1_ROOT/GMR/.git"
  rep "A  env gvhmr(hmr4d/pytorch3d)" gvhmr_env_ok
  rep "A  env gmr(retarget/mujoco/mink)" gmr_env_ok
  rep "B  GVHMR 权重×4"          weights_ok
  rep "C  SMPL neutral(npz+pkl)" smpl_ok
  rep "D  smoke .pt($STEM)"     pt_ok
  rep "D  smoke csv($STEM)"     csv_ok
}

# ===== 0. 前置检查 =====
log "0. 前置检查"
command -v "$CONDA" >/dev/null 2>&1 || die "找不到 conda（CONDA=$CONDA）。盒上一般自带 miniconda。"
# G1_ROOT 自动探测：默认值对不上时，回退到当前目录（在仓内直接 bash 脚本即可）
if [ ! -f "$G1_ROOT/.gitmodules" ] && [ -f "$PWD/.gitmodules" ]; then
  G1_ROOT="$PWD"; log "  G1_ROOT 自动设为 $G1_ROOT（取自当前目录）"
fi
[ -f "$G1_ROOT/.gitmodules" ] || die "G1_ROOT=$G1_ROOT 不像 g1_dance 仓（缺 .gitmodules）。cd 进仓目录再跑，或 G1_ROOT=<path> bash $0"
nvidia-smi >/dev/null 2>&1 || die "nvidia-smi 不可用，确认盒上有 GPU"
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
log "  GPU 显存: ${VRAM_MB} MiB（≥12000 即舒适跑 GVHMR 推理）"
[ "${VRAM_MB:-0}" -ge 12000 ] || log "  ⚠ 显存 <12G，推理可能 OOM；建议换 24G 卡"

progress_report
if [ "$STAGE" = status ]; then log "STAGE=status，仅检查不干活，退出。"; exit 0; fi

mkdir -p "$G1_ROOT/whole_body_tracking/motions/csv"

# ===== F. 自动补依赖：缺失的 GVHMR/GMR 从 .gitmodules 的 URL shallow clone =====
# 直接全量 shallow clone（云盒网快+盘大，1.5G 的 GMR ~1-2 分钟；比 sparse 简单且不会漏路径）
log "F. 检查/拉取 GVHMR + GMR（URL 取自 $G1_ROOT/.gitmodules）"
( cd "$G1_ROOT"
  git config -f .gitmodules --get-regexp 'submodule\..*\.(path|url)' | \
    awk '{split($1,a,"."); s=a[2]; k=a[3]; if(k=="path")p[s]=$2; if(k=="url")u[s]=$2} END{for(s in p) printf "%s\t%s\n",p[s],u[s]}' | \
    while IFS=$'\t' read -r path url; do
      if [ -d "$path/.git" ]; then log "  skip $path（已是 git 仓）"; continue; fi
      if [ -d "$path" ]; then log "  清理残留 $path（非 git 仓，删后重 clone）"; rm -rf "$path"; fi
      log "  clone $path ← $url（--depth 1）"
      git clone --depth 1 "$url" "$path"
    done
)
[ -d "$G1_ROOT/GVHMR" ] || die "GVHMR 仍缺失（git 不通？配 http.proxy 后重跑）"
[ -d "$G1_ROOT/GMR" ]   || die "GMR 仍缺失（git 不通？配 http.proxy 后重跑）"

# 目标目录在 clone 之后建（clone 前把 GVHMR/GMR 建成非空会导致 git clone 失败）
mkdir -p "$G1_ROOT/GVHMR/inputs/checkpoints/gvhmr" \
         "$G1_ROOT/GVHMR/inputs/checkpoints/hmr2" \
         "$G1_ROOT/GVHMR/inputs/checkpoints/vitpose" \
         "$G1_ROOT/GVHMR/inputs/checkpoints/yolo" \
         "$G1_ROOT/GVHMR/inputs/checkpoints/body_models/smplx" \
         "$G1_ROOT/GVHMR/inputs/checkpoints/body_models/smpl" \
         "$G1_ROOT/GMR/assets/body_models/smplx"

# ===== A. conda env（新建 gvhmr / gmr，不碰 base）=====
if [[ "$STAGE" == all || "$STAGE" == env ]]; then
  log "A. 建新 conda env：gvhmr + gmr（py3.10）"

  if gvhmr_env_ok; then log "  skip gvhmr env（已装好 hmr4d/pytorch3d）"
  else
    "$CONDA" env list | awk '{print $1}' | grep -qx gvhmr || "$CONDA" create -y -n gvhmr python=3.10 "${CONDA_MAIN[@]}"
    log "  gvhmr: 装 requirements.txt（torch2.3+cu121 / pytorch3d / pycolmap / smplx）+ pip install -e ."
    # 国内盒：pytorch.org 常不通 + 阿里云 pytorch-wheels 非 PEP503 → torch/torchvision 直接用阿里云直链 wheel，删 extra-index
    if [ "$MIRROR" = "1" ]; then
      sed -i \
        -e '/^--extra-index-url /d' \
        -e 's#^torch==2.3.0+cu121$#torch @ https://mirrors.aliyun.com/pytorch-wheels/cu121/torch-2.3.0%2Bcu121-cp310-cp310-linux_x86_64.whl#' \
        -e 's#^torchvision==0.18.0+cu121$#torchvision @ https://mirrors.aliyun.com/pytorch-wheels/cu121/torchvision-0.18.0%2Bcu121-cp310-cp310-linux_x86_64.whl#' \
        "$G1_ROOT/GVHMR/requirements.txt"
    fi
    # numpy 1.23.5 老 pin 会逼 pip 对 opencv/matplotlib/scikit-image 回溯试一堆版本；钉到 numpy 兼容版本省掉回溯
    sed -i \
      -e 's#^opencv-python$#opencv-python==4.10.0.84#' \
      -e 's#^matplotlib$#matplotlib<3.10#' \
      -e 's#^scikit-image$#scikit-image<0.25#' \
      "$G1_ROOT/GVHMR/requirements.txt"
    # cython_bbox 的 sdist 缺 src/cython_bbox.c（打包破损，无 cp310 wheel），且 GVHMR/wis3d 都不 import 它 → 注释掉跳过
    sed -i 's|^cython_bbox$|# cython_bbox  # skipped: broken sdist (no .c), unused by GVHMR/wis3d|' "$G1_ROOT/GVHMR/requirements.txt"
    # chumpy 是老 sdist，setup.py 里 `import pip`，PEP517 隔离构建环境没 pip → 先备 numpy/setuptools，-r 关构建隔离
    "$CONDA" run -n gvhmr pip install "${PIP_INDEX[@]}" numpy==1.23.5 "setuptools>=68" wheel
    PIP_R=("${PIP_INDEX[@]}" --no-build-isolation)
    "$CONDA" run -n gvhmr --live-stream pip install "${PIP_R[@]}" -r "$G1_ROOT/GVHMR/requirements.txt"
    "$CONDA" run -n gvhmr --live-stream pip install "${PIP_INDEX[@]}" -e "$G1_ROOT/GVHMR"
    gvhmr_env_ok || die "gvhmr env 装完仍 import 失败（hmr4d/pytorch3d）"
  fi

  if gmr_env_ok; then log "  skip gmr env（已装好 retarget/mujoco/mink）"
  else
    "$CONDA" env list | awk '{print $1}' | grep -qx gmr || "$CONDA" create -y -n gmr python=3.10 "${CONDA_MAIN[@]}"
    log "  gmr: pip install -e GMR（含 smplx@git+github，需 github 可达）+ mujoco + mink"
    "$CONDA" run -n gmr --live-stream pip install "${PIP_INDEX[@]}" -e "$G1_ROOT/GMR"
    "$CONDA" run -n gmr --live-stream pip install "${PIP_INDEX[@]}" mujoco mink
    "$CONDA" install -y -n gmr libstdcxx-ng "${CONDA_FORGE[@]}"
    gmr_env_ok || die "gmr env 装完仍 import 失败（retarget/mujoco/mink）"
  fi
fi

# ===== B. GVHMR 权重（4 个；盒子 hf 不通，只检查+提醒，不试下载）=====
if [[ "$STAGE" == all || "$STAGE" == weights ]]; then
  log "B. GVHMR 推理权重 → $G1_ROOT/GVHMR/inputs/checkpoints"
  if weights_ok; then log "  skip（4 个权重已就位）"
  else
    cat <<EOF

  ── 缺权重。你直接复制到这些路径，放好重跑 ──
    $G1_ROOT/GVHMR/inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt   (163MB)
    $G1_ROOT/GVHMR/inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt       (2.7GB，注意文件名是 = 不是 %3D)
    $G1_ROOT/GVHMR/inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth    (2.5GB)
    $G1_ROOT/GVHMR/inputs/checkpoints/yolo/yolov8x.pt                      (137MB)
  ── 放好后重跑：bash $0 ──
EOF
    exit 0
  fi
fi

# ===== C. SMPL 落位检查（许可门控，必须你手动下）=====
# SMPLX neutral .npz 要放两处：GVHMR 推理 + GMR 的 gvhmr_to_csv(smplx.create)；.pkl 一处(GMR L119 检查)
SMPLX_NPZ="$G1_ROOT/GVHMR/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz"
SMPLX_NPZ_GMR="$G1_ROOT/GMR/assets/body_models/smplx/SMPLX_NEUTRAL.npz"
SMPLX_PKL="$G1_ROOT/GMR/assets/body_models/smplx/SMPLX_NEUTRAL.pkl"
SMPL_PKL="$G1_ROOT/GVHMR/inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl"
log "C. SMPL 检查（这步只能你手动注册下，AI 替不了）"
miss=0
[ -f "$SMPLX_NPZ" ] && log "  ✓ SMPLX_NEUTRAL.npz (GVHMR)" || { log "  ✗ SMPLX_NEUTRAL.npz（GVHMR 推理要）→ $SMPLX_NPZ"; miss=1; }
[ -f "$SMPLX_NPZ_GMR" ] && log "  ✓ SMPLX_NEUTRAL.npz (GMR)" || { log "  ✗ SMPLX_NEUTRAL.npz（gvhmr_to_csv 的 smplx.create 要）→ $SMPLX_NPZ_GMR（同一个 .npz 复制到 GMR 这边）"; miss=1; }
[ -f "$SMPLX_PKL" ] && log "  ✓ SMPLX_NEUTRAL.pkl (GMR)" || { log "  ✗ SMPLX_NEUTRAL.pkl（gvhmr_to_csv L119 检查）→ $SMPLX_PKL"; miss=1; }
[ -f "$SMPL_PKL" ]  && log "  ✓ SMPL_NEUTRAL.pkl（仅 render 要，已跳过 render → 可暂缓）" || log "  ⚠ SMPL_NEUTRAL.pkl 缺（render 已跳过，不影响出 .pt/csv）"
if [ "$miss" = "1" ]; then
  cat <<EOF

  ── 缺 SMPL。SMPLX neutral 的 .npz 要放两处、.pkl 一处 ──
    $SMPLX_NPZ          （GVHMR 推理）
    $SMPLX_NPZ_GMR      （gvhmr_to_csv 的 smplx.create，同一个 .npz 复制到 GMR 这边）
    $SMPLX_PKL          （gvhmr_to_csv L119 检查）
  SMPL neutral（仅 render，已跳过→可暂缓）：$SMPL_PKL
  ── 放好后重跑：bash $0 ──
EOF
  exit 0
fi

# ===== D. smoke：$VIDEO -s → .pt → .csv（跳 render）=====
if [[ "$STAGE" == all || "$STAGE" == smoke ]]; then
  log "D. smoke test：$VIDEO → .pt → .csv"
  PT="$G1_ROOT/GVHMR/outputs/demo/$STEM/hmr4d_results.pt"
  [ -f "$PT" ] || PT=$(find "$G1_ROOT/GVHMR/outputs" -name hmr4d_results.pt -path "*$STEM*" 2>/dev/null | head -1 || true)

  if [[ "$STAGE" == all ]] && csv_ok && [ -n "$PT" ] && [ -f "$PT" ]; then
    log "  skip（$STEM .pt + csv 都在，smoke 已完成；STAGE=smoke 可强制重跑）"
  else
    DEMO="$G1_ROOT/GVHMR/tools/demo/demo.py"
    # patch 掉 render + merge（render 跳过后 incam/global 视频不存在→merge 必挂；且 ffmpeg 可能没装）。幂等。
    sed -i 's/^    render_incam(cfg)$/    pass  # SKIP_RENDER render_incam(cfg)/' "$DEMO"
    sed -i 's/^    render_global(cfg)$/    pass  # SKIP_RENDER render_global(cfg)/' "$DEMO"
    sed -i 's/^        merge_videos_horizontal(.*/        pass  # SKIP_RENDER merge_videos_horizontal/' "$DEMO"
    grep -q 'SKIP_RENDER' "$DEMO" && log "  已 patch demo.py 跳过 render + merge（恢复：sed -i '/SKIP_RENDER/s/pass  # //' \"$DEMO\"）"
    log "  [1/2] GVHMR 推理（gvhmr env，-s 静止机位）"
    ( cd "$G1_ROOT/GVHMR" && \
      "$CONDA" run -n gvhmr --live-stream python tools/demo/demo.py \
        --video="$VIDEO" -s )
    [ -f "$PT" ] || PT="$G1_ROOT/GVHMR/outputs/demo/$STEM/hmr4d_results.pt"
    [ -f "$PT" ] || PT=$(find "$G1_ROOT/GVHMR/outputs" -name hmr4d_results.pt -path "*$STEM*" 2>/dev/null | head -1 || true)
    [ -n "$PT" ] && [ -f "$PT" ] || die "没生成 hmr4d_results.pt"
    log "  ✓ .pt: $PT ($(du -h "$PT" | cut -f1))"
    log "  [2/2] gvhmr_to_csv（gmr env，SMPL-X→G1 IK retarget）"
    # patch GMR bug: mink.solve_ik 签名 (conf,tasks,dt,solver,damping, safety_break=False, limits=None,...)
    # GMR 调用 ...damping, self.ik_limits) 把 ik_limits 当第6位置参→落 safety_break(列表 truthy→超限 raise)，limits 反而空。
    # 修：末参 self.ik_limits → limits=self.ik_limits（keyword），safety_break 留默认 False（clamp 不 raise）。
    GMR_MR="$G1_ROOT/GMR/general_motion_retargeting/motion_retarget.py"
    if ! grep -q 'limits=self.ik_limits' "$GMR_MR" 2>/dev/null; then
      sed -i 's/, safety_break=False//' "$GMR_MR"   # 撤掉之前误加的（若有）
      sed -i 's/self\.ik_limits$/limits=self.ik_limits/' "$GMR_MR"
      log "  已 patch GMR motion_retarget.py: ik_limits→limits=（原被传成 safety_break→超限 raise）"
    fi
    "$CONDA" run -n gmr --live-stream python "$G1_ROOT/whole_body_tracking/scripts/gvhmr_to_csv.py" \
      --gvhmr_pred_file "$PT" \
      --output_file "$G1_ROOT/whole_body_tracking/motions/csv/$STEM.csv"
  fi

  CSV="$G1_ROOT/whole_body_tracking/motions/csv/$STEM.csv"
  [ -f "$CSV" ] || die "没生成 $STEM.csv"
  log "  ✓ CSV: $CSV"
  "$CONDA" run -n gvhmr python - "$CSV" <<'PY'
import sys,numpy as np
m=np.loadtxt(sys.argv[1],delimiter=",")
print(f"    形状={m.shape}  列数={'✓36' if m.shape[1]==36 else '✗'+str(m.shape[1])}  行数(帧)={m.shape[0]}")
print(f"    root_xyz 范围 {m[:,0:3].min(0).round(2)}~{m[:,0:3].max(0).round(2)}；关节弧度范围 {m[:,7:].min():.2f}~{m[:,7:].max():.2f}")
PY

  # bonus: 渲 G1 动作视频（offscreen，mujoco；验证用，失败不影响 csv）
  G1MP4="$G1_ROOT/whole_body_tracking/motions/csv/${STEM}_g1.mp4"
  log "  [3/3] 渲 G1 动作视频（offscreen）→ $G1MP4"
  "$CONDA" run -n gmr --live-stream python "$G1_ROOT/render_g1_motion_offscreen.py" \
    --csv "$CSV" --out "$G1MP4" 2>&1 | tail -3 || log "  ⚠ 渲染失败（不影响 csv/npz）"
  [ -f "$G1MP4" ] && log "  ✓ G1 动作视频: $G1MP4" || true

  cat <<EOF

  ═══ 视频→CSV 桥跑通 ═══
  CSV 在：$CSV
  G1 动作视频（验证用，下载本地看）：$G1MP4
  （本地验收到此为止——csv_to_npz.py 需 isaaclab，不在本盒跑）

  下一步（在 flux / Isaac 侧）：
  1. 把 $STEM.csv 传到仓 whole_body_tracking/motions/csv/
  2. Isaac 路径跑 csv_to_npz.py → motions/${STEM}_isaac.npz
  3. git add+commit+push hmj932/g1_dance（并 git push gitlab main）
  4. flux: gm-run g1_dance/whole_body_tracking/scripts/entry_train_and_play.py \\
        ... --motion_file motions/${STEM}_isaac.npz
  批量转舞蹈视频：VIDEO=docs/example_video/<你的>.mp4 bash $0（-s 仅适合静止机位拍摄）。
EOF
fi
