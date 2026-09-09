# DARC 可复现交付包

DARC（Damage-Aware Retrieval-Augmented Captioning）使用冻结的 CLIP ViT-B/32 图像表示，从有标注的结构损伤图片中检索近邻，通过相似度加权投票输出多标签损伤类别，并转移最近的非空参考描述。

本目录只包含最终采用的 DARC 方法，不包含效果不佳的探索分支或其他对照模型。

## 1. 最重要的实验边界

DARC 没有梯度训练过程。“训练 DARC”实际是：

1. 用冻结的 CLIP 为有标注图片提取特征；
2. 将特征、标签和参考描述构建成检索索引；
3. 对未知图片检索 top-5 近邻并生成结果。

本包提供两个数据作用域，但算法和参数完全相同：

- `research`：只用 836 张 grouped-train 图片构建索引，用于复现 182 张隔离测试集上的正式指标。
- `competition`：模型选择和参数冻结后，用赛事方提供的全部 1200 张有标注图片构建索引，只用于预测另外提供的未知比赛图片。

当前发布已使用修订后的 1200 张图片和标注重建：32 张内容发生变化的图片重新提取了 CLIP embedding，其余 1168 张沿用位级一致的冻结 embedding；所有 train/validation/test 标签均从新版 manifest 重新生成。

**可以使用全部 1200 张图片构建最终比赛模型，前提是最终评分图片不是这 1200 张中的图片，并且所有算法参数已经根据 train/validation 固定。**

不能用 1200 图索引重新预测原 182 张测试图，再把该结果报告为泛化性能；那些测试图已经进入检索库，会形成直接数据泄漏。代码在 competition artifact 中写入了 `eligible_for_original_grouped_test_evaluation=false`，用于防止混淆。

## 2. 固定 DARC 配置

配置文件为 `config/darc.json`。最终算法固定为：

```text
Encoder: frozen CLIP ViT-B/32
Similarity: cosine similarity
Retrieval: top-k = 5
Category vote: non-negative similarity-weighted multilabel vote
Category threshold: 0.35 x maximum category vote
Description: nearest neighbor with a non-empty reference description
```

输出严格为：

```json
{
  "image_id": "example.jpg",
  "damage_categories": ["crack", "spalling"],
  "description": "Visible cracking and local material spalling are present."
}
```

检索近邻和分数写到独立的 `*.evidence.json`，不会污染比赛结果 JSON。

## 3. 目录结构

```text
DARC_reproducible/
├── README.md
├── requirements.txt
├── config/
│   ├── darc.json
│   └── server_sync_manifest.json
├── data/
│   └── dataset/                     # 用户放置赛事数据，不随公开包分发
├── src/                             # DARC 核心实现
├── scripts/                         # 数据、特征、artifact、预测和验证入口
├── tests/                           # 独立交付包测试
└── artifacts/
    ├── data/                        # portable manifest 和两类 split
    ├── features/research/           # 已验证的冻结 CLIP 特征
    ├── features/competition/        # 1200 图全量特征
    ├── runtime/research/            # 836 图研究索引
    ├── runtime/competition/         # 1200 图比赛索引
    ├── predictions/                 # 冻结测试预测及证据
    ├── metrics/                     # 独立评价结果
    └── release_verification.json
```

## 4. 环境

正式特征 provenance 记录的环境：

```text
Python 3.11.7
PyTorch 2.6.0+cu124
Transformers 4.46.3
NumPy 1.26.4
Pillow 10.2.0
CUDA 12.4
NVIDIA A100 80GB PCIe
```

创建环境：

```bash
cd DARC_reproducible
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorch CUDA wheel 必须与服务器驱动兼容。若默认 PyPI wheel 不适合服务器，应根据 PyTorch 官方索引安装与 CUDA 12.4 对应的 `torch==2.6.0`，再安装其余依赖。

CPU 可以运行推理和单元测试，但完整 CLIP 特征导出会更慢。要复现正式服务器环境，应同时保存 `pip freeze`、Python 版本和 `nvidia-smi` 输出。

## 5. 数据放置

由于赛事数据和第三方模型可能受许可限制，本包不假定它们可以公开再分发。获得赛事方允许后，将数据放为：

```text
data/dataset/
├── description.json
└── image/
    ├── 00002.jpg
    ├── ...
    └── <共 1200 张图片>
```

`description.json` 应包含 2400 条标注，即每张图片一条类别问答和一条描述问答。运行：

```bash
python scripts/prepare_data.py \
  --dataset data/dataset \
  --output artifacts/data \
  --freeze-splits-from artifacts/data/splits_grouped
```

预期输出：

```text
Prepared 1200 labeled images
Frozen grouped split: {'train': 836, 'val': 182, 'test': 182}
```

分组规则使用 seed 2026、文件名窗口 10，并把序列邻近图片和精确重复图片保持在同一组。修订标注会改变按类别分层的重新分配结果，因此本次发布通过 `--freeze-splits-from` 按已发布的 image-ID roster 保留 836/182/182 成员；生成结果仍必须通过 `config/darc.json` 中的三份 sample-ID 序列哈希。

## 6. CLIP 模型

推理要求本地 CLIP ViT-B/32 目录，例如：

```text
models/clip-vit-base-patch32/
├── config.json
├── preprocessor_config.json
├── tokenizer.json
├── tokenizer_config.json
├── vocab.json
├── merges.txt
└── pytorch_model.bin
```

每个文件的固定 SHA-256 位于：

- `config/darc.json`
- `config/server_sync_manifest.json`

不能只依赖模型目录名。`scripts/predict.py` 会逐文件检查哈希，不匹配时拒绝预测。

## 7. 快速核验已保存研究结果

本交付目录已经包含冻结特征、research artifact、测试预测和指标。首先执行：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tests -q
python scripts/verify_release.py
```

预期：

```text
18 passed
PASS configuration
PASS data_protocol
PASS frozen_research_features
PASS research_runtime
PASS competition_runtime
PASS research_predictions
PASS research_metrics
PASS release_manifest
```

`RELEASE_MANIFEST.json` 记录所有可分发文件的校验值：普通文件按精确字节和大小校验，`.npz` 按数组内容校验。因此本地重建 artifact 后校验仍然通过，即使压缩容器字节改变。只有在真正修改并重新发布交付包时，才需要运行 `python scripts/build_release_manifest.py` 更新清单。

如果本地 CLIP 模型已经同步，还应执行：

```bash
python scripts/verify_release.py \
  --model models/clip-vit-base-patch32
```

正式 grouped-test 结果为：

```text
Test images:                 182
Exact-match accuracy:       0.7142857143
Primary-category accuracy:  0.9725274725
Micro-F1:                   0.8825910931
Macro-F1 (fixed 11 class):  0.6616346534
Description token F1:       0.5973337308
Description token Jaccard:  0.4393743333
```

## 8. 从冻结特征复现正式结果

重建 836 图研究 artifact：

```bash
python scripts/build_artifact.py research
```

命令会输出外部 artifact fingerprint。当前已保存 artifact 的 fingerprint 为：

```text
92f8a5bf5c41040d214ae8b9b8ab82df29015303ce05217f44f88a8a63a1d1c2
```

fingerprint 由 `runtime_config.json` 的规范化内容哈希和检索索引的**数组内容**哈希组成，不依赖 `.npz` 容器字节。因此在不同机器上用相同输入重建 artifact 会得到相同 fingerprint，即使压缩后的 `.npz` 文件字节不同。这一点已在 macOS 与 Linux A100 服务器上交叉验证。

在预测阶段只传入不含标签的 `test.npz`：

```bash
python scripts/predict_features.py \
  --features artifacts/features/research/test.npz \
  --artifact artifacts/runtime/research \
  --fingerprint 92f8a5bf5c41040d214ae8b9b8ab82df29015303ce05217f44f88a8a63a1d1c2 \
  --output artifacts/predictions/darc_grouped_test.json
```

随后由独立命令加载测试标签：

```bash
python scripts/evaluate.py \
  --truth artifacts/data/splits_grouped/test.json \
  --predictions artifacts/predictions/darc_grouped_test.json \
  --output artifacts/metrics/darc_grouped_test.json
```

预测命令不接收测试标签参数；只有评价命令读取测试真值。

## 9. 从原始图片重新导出研究特征

若不使用服务器保存的 `.npz`，可从 1200 张原始图重新导出：

```bash
python scripts/export_features.py research \
  --dataset data/dataset \
  --data-artifacts artifacts/data \
  --model models/clip-vit-base-patch32 \
  --output artifacts/features/research \
  --device cuda \
  --batch-size 32
```

该命令分别生成：

```text
train.npz          # 836 张，含标签
val.npz            # 182 张，含标签
test.npz           # 182 张，不含标签
test_labels.npz    # 独立测试标签，仅供 evaluate
provenance.json    # 模型、环境、输入顺序和 SHA-256
```

完成后依次执行第 8 节的 artifact、预测和评价命令。

## 10. 用全部 1200 张图构建比赛模型

必须先完成研究阶段、固定 `top_k=5` 和阈值 `0.35`，然后才能建立全量索引。两个全量入口都强制要求 `--confirm-parameters-frozen`；缺少该参数时程序会退出，不会读取测试标签或生成全量索引。

### 方法 A：合并已经冻结的三份特征

这是当前交付包采用的方法，不需要重新运行 CLIP：

```bash
python scripts/combine_frozen_features.py --confirm-parameters-frozen
python scripts/build_artifact.py competition
```

当前 1200 图比赛 artifact fingerprint：

```text
2e9e20a1aefeb01143ddc6f367f0398a923db938a85329f0c4b39bc6724ba6c7
```

### 方法 B：从 1200 张原图直接重新编码

```bash
python scripts/export_features.py competition \
  --dataset data/dataset \
  --data-artifacts artifacts/data \
  --model models/clip-vit-base-patch32 \
  --output artifacts/features/competition \
  --device cuda \
  --batch-size 32 \
  --confirm-parameters-frozen

python scripts/build_artifact.py competition
```

两种方法应得到相同的图片顺序和 CLIP embedding 数值。artifact fingerprint 基于数组内容而非 `.npz` 容器字节，因此跨机器重建会得到一致的 fingerprint；压缩文件本身的字节可能因 zlib 实现不同而变化，属于预期现象。

## 11. 预测未知比赛图片

将赛事方另外提供的未知图片放到任意目录，例如 `data/unseen_test/`，然后执行：

```bash
python scripts/predict.py \
  --input data/unseen_test \
  --artifact artifacts/runtime/competition \
  --fingerprint 2e9e20a1aefeb01143ddc6f367f0398a923db938a85329f0c4b39bc6724ba6c7 \
  --model models/clip-vit-base-patch32 \
  --output outputs/result.json \
  --device cuda \
  --batch-size 32
```

输出：

```text
outputs/result.json           # 严格比赛格式
outputs/result.evidence.json  # 检索证据和来源，不提交给比赛系统
```

赛事方尚未提供未知测试图片，因此本次发布只包含已重建的 competition artifact，不包含 `outputs/result.json`。

若比赛要求的字段名或外层 JSON 结构不同，只应在最终序列化层转换；不要修改 DARC 检索和投票逻辑。

### 11.1 可选：要求描述来源与预测类别有交集

DARC 的类别来自 top-5 加权投票，描述直接取最近的非空参考描述，两者可能指向不同缺陷。域外图片上这种错配更明显。`scripts/predict.py` 和 `scripts/predict_features.py` 提供一个**默认关闭**的开关：

```bash
python scripts/predict.py \
  --input data/unseen_test \
  --artifact artifacts/runtime/competition \
  --fingerprint 2e9e20a1aefeb01143ddc6f367f0398a923db938a85329f0c4b39bc6724ba6c7 \
  --model models/clip-vit-base-patch32 \
  --output outputs/result.json \
  --device cuda \
  --require-category-overlap
```

启用后，描述改为取 top-5 中**第一个自身类别与预测类别有交集**的非空描述；若没有邻居满足，则退回冻结行为并在 evidence 里记 `description_overlap_fallback: true`。

边界说明：

- 该开关**只影响描述文本**，不改变 `damage_categories`，也不改变检索、投票或阈值。
- `DARC_PARAMETERS` 未变动，因此两个 artifact 和它们的 fingerprint 都不受影响，无需重建。
- 复现正式 grouped-test 结果时必须保持关闭；开启不属于冻结算法定义。
- 该选项没有在隔离测试集上验证过收益，只用于提升域外输入的可读性。
- 判据是"交集非空"，因此只能挡住完全跑偏的来源，挡不住主类别错配。在一批 15 张域外病害图片上实测：13 条描述不变、2 条被纠正、0 条退回；此前 7 条类别/描述不一致中的另外 5 条属于部分重叠，仍保持不变。

## 12. 从服务器导入冻结特征

服务器同步清单和准确 SHA-256 位于 `config/server_sync_manifest.json`。新版服务器 `.npz` 中的 embedding 已刷新，但 train/validation 标签数组仍来自旧标注；不能直接覆盖发布特征。先验证并同步三份原始特征到 staging，再用新版 split 重建标签：

```bash
python scripts/sync_verified_artifacts.py \
  --source-root /path/to/server-or-staging-root \
  --destination /path/to/refreshed/features/research \
  --dry-run

python scripts/sync_verified_artifacts.py \
  --source-root /path/to/server-or-staging-root \
  --destination /path/to/refreshed/features/research

python scripts/import_refreshed_features.py \
  --source-features /path/to/refreshed/features/research \
  --source-provenance /path/to/server/provenance.json \
  --taxonomy artifacts/data/label_taxonomy.json \
  --data-artifacts artifacts/data \
  --dataset data/dataset \
  --output artifacts/features/research \
  --changed-image-rows 32
```

同步脚本只接受清单中的 3 份刷新特征，并在复制前后验证 SHA-256。导入脚本随后丢弃历史 train/validation 标签，依据新版 manifest 重建 train、validation 和独立 test labels，同时保留 embedding float32 位值。当前代码已经完整收录，不需要再从服务器复制 DARC 源码。仍建议保存：

- 刷新后的 `{train,val,test}.npz`
- 原始 CLIP 导出 provenance
- 本地 CLIP ViT-B/32 模型目录
- `python -m pip freeze` 输出
- Python、CUDA、GPU 和驱动信息

可在服务器的本交付目录中生成统一环境快照：

```bash
python scripts/capture_environment.py \
  --model /mnt/datablob/qizhili/hf_models/clip-vit-base-patch32 \
  --output artifacts/environment.json
python -m pip freeze > artifacts/pip-freeze-server.txt
```

## 13. 可复现性和提交注意事项

- 研究分数必须来自 research artifact，不能来自 1200 图 competition artifact。
- 复现研究分数时不要启用 `--require-category-overlap`；它只影响描述文本，不属于冻结算法。
- competition artifact 只面向与 1200 张训练图分离的未知图片。
- 不要在参数选择、阈值选择或错误分析过程中查看 182 张测试图标签。
- 每次重新导出特征或构建 artifact 后都要保存 provenance 和 fingerprint。
- 提交前检查 `result.json` 中每个对象恰好只有 `image_id`、`damage_categories`、`description`。
- 若赛事规则禁止使用外部预训练模型，应先确认 CLIP 是否允许；算法可复现不等于赛事规则自动允许。
- 若准备公开发布该目录，应先确认 1200 图数据、派生标注、CLIP 权重和比赛材料的再分发许可。

## 14. 完整验收顺序

```bash
python scripts/prepare_data.py \
  --dataset data/dataset \
  --output artifacts/data \
  --freeze-splits-from artifacts/data/splits_grouped
python scripts/build_artifact.py research
python scripts/predict_features.py \
  --features artifacts/features/research/test.npz \
  --artifact artifacts/runtime/research \
  --fingerprint 92f8a5bf5c41040d214ae8b9b8ab82df29015303ce05217f44f88a8a63a1d1c2 \
  --output artifacts/predictions/darc_grouped_test.json
python scripts/evaluate.py \
  --predictions artifacts/predictions/darc_grouped_test.json
python scripts/combine_frozen_features.py --confirm-parameters-frozen
python scripts/build_artifact.py competition
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tests -q
python scripts/verify_release.py
```

只有最后两条均通过后，才把目录标记为可交付。模型文件同步完成后，再追加 `python scripts/verify_release.py --model <CLIP目录>` 完成模型权重验收。
