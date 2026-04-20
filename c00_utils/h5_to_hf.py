# -*- coding: utf-8 -*-
# @Time    : 2026/4/20 17:48
# @Author  : CFuShn
# @Comments: 
# @Software: PyCharm

"""
(p312) cy@xin-huanan-gpu1:~$  python /home/cy/nuist-lab/cfs-code-lab/c00_utils/h5_inspect.py /home/data/OmniFace_202602042244.h5
# File: /home/data/OmniFace_202602042244.h5
# Size: 74417512549 bytes
# Driver: sec2
# Libver: ('earliest', 'v114')

// (group)
├─ age (dataset) :: float32; shape=(1121349,); filters=fillvalue=0.0
├─ arched_eyebrows (dataset) :: int64; shape=(1121349,); filters=fillvalue=0
├─ arousal (dataset) :: float32; shape=(1121349,); filters=fillvalue=0.0
├─ attractive (dataset) :: int64; shape=(1121349,); filters=fillvalue=0
├─ bags_under_eyes (dataset) :: int64; shape=(1121349,); filters=fillvalue=0
├─ bald (dataset) :: int64; shape=(1121349,); filters=fillvalue=0
├─ bangs (dataset) :: int64; shape=(1121349,); filters=fillvalue=0
├─ big_lips (dataset) :: int64; shape=(1121349,); filters=fillvalue=0
├─ big_nose (dataset) :: int64; shape=(1121349,); filters=fillvalue=0
├─ black_hair (dataset) :: int64; shape=(1121349,); filters=fillvalue=0
├─ blond_hair (dataset) :: int64; shape=(1121349,); filters=fillvalue=0
├─ blurry (dataset) :: int64; shape=(1121349,); filters=fillvalue=0
├─ brown_hair (dataset) :: int64; shape=(1121349,); filters=fillvalue=0
├─ bushy_eyebrows (dataset) :: int64; shape=(1121349,); filters=fillvalue=0
├─ chubby (dataset) :: int64; shape=(1121349,); filters=fillvalue=0
├─ double_chin (dataset) :: int64; shape=(1121349,); filters=fillvalue=0
├─ expression (dataset) :: int64; shape=(1121349,); filters=fillvalue=0
├─ eyeglasses (dataset) :: int64; shape=(1121349,); filters=fillvalue=0
├─ five_o_clock_shadow (dataset) :: int64; shape=(1121349,); filters=fillvalue=0
├─ gaze_dir (dataset) :: float32; shape=(1121349, 2); filters=fillvalue=0.0
├─ goatee (dataset) :: int64; shape=(1121349,); filters=fillvalue=0
├─ gray_hair (dataset) :: int64; shape=(1121349,); filters=fillvalue=0
├─ head_pose (dataset) :: float32; shape=(1121349, 3); filters=fillvalue=0.0
├─ heavy_makeup (dataset) :: int64; shape=(1121349,); filters=fillvalue=0
├─ high_cheekbones (dataset) :: int64; shape=(1121349,); filters=fillvalue=0
├─ id (dataset) :: |S200; shape=(1121349,); filters=chunks=(1024,),fillvalue=b''
├─ images (dataset) :: object; shape=(1121349,); filters=compression=gzip,compression_opts=4,chunks=(1024,)
├─ is_sr (dataset) :: int8; shape=(1121349,); filters=chunks=(1024,),fillvalue=0
├─ male (dataset) :: int64; shape=(1121349,); filters=fillvalue=0
├─ mouth_slightly_open (dataset) :: int64; shape=(1121349,); filters=fillvalue=0
├─ mustache (dataset) :: int64; shape=(1121349,); filters=fillvalue=0
├─ narrow_eyes (dataset) :: int64; shape=(1121349,); filters=fillvalue=0
├─ no_beard (dataset) :: int64; shape=(1121349,); filters=fillvalue=0
├─ origin (dataset) :: |S100; shape=(1121349,); filters=chunks=(1024,),fillvalue=b''
├─ oval_face (dataset) :: int64; shape=(1121349,); filters=fillvalue=0
├─ pale_skin (dataset) :: int64; shape=(1121349,); filters=fillvalue=0
├─ pointy_nose (dataset) :: int64; shape=(1121349,); filters=fillvalue=0
├─ prompt (dataset) :: |S300; shape=(1121349,); filters=fillvalue=b''
├─ race (dataset) :: int64; shape=(1121349,); filters=fillvalue=0
├─ receding_hairline (dataset) :: int64; shape=(1121349,); filters=fillvalue=0
├─ rosy_cheeks (dataset) :: int64; shape=(1121349,); filters=fillvalue=0
├─ sideburns (dataset) :: int64; shape=(1121349,); filters=fillvalue=0
├─ smiling (dataset) :: int64; shape=(1121349,); filters=fillvalue=0
├─ straight_hair (dataset) :: int64; shape=(1121349,); filters=fillvalue=0
├─ train_indices (dataset) :: int32; shape=(1059173,); filters=fillvalue=0
├─ val_indices (dataset) :: int32; shape=(62176,); filters=fillvalue=0
├─ valence (dataset) :: float32; shape=(1121349,); filters=fillvalue=0.0
├─ wavy_hair (dataset) :: int64; shape=(1121349,); filters=fillvalue=0
├─ wearing_earrings (dataset) :: int64; shape=(1121349,); filters=fillvalue=0
├─ wearing_hat (dataset) :: int64; shape=(1121349,); filters=fillvalue=0
├─ wearing_lipstick (dataset) :: int64; shape=(1121349,); filters=fillvalue=0
├─ wearing_necklace (dataset) :: int64; shape=(1121349,); filters=fillvalue=0
├─ wearing_necktie (dataset) :: int64; shape=(1121349,); filters=fillvalue=0
└─ young (dataset) :: int64; shape=(1121349,); filters=fillvalue=0
(p312) cy@xin-huanan-gpu1:~$ python /home/cy/nuist-lab/cfs-code-lab/c00_utils/h5_inspect.py /home/data/OmniShape1k_18000a_128x128_20251204.h5
# File: /home/data/OmniShape1k_18000a_128x128_20251204.h5
# Size: 133615010492 bytes
# Driver: sec2
# Libver: ('earliest', 'v114')

// (group)
├─ 2d_coverage (dataset) :: float32; shape=(18000000,); filters=fillvalue=0.0
├─ 2d_hw_ratio (dataset) :: float32; shape=(18000000,); filters=fillvalue=0.0
├─ 2d_rgb_complexity (dataset) :: float32; shape=(18000000,); filters=fillvalue=0.0
├─ 2d_silhouette_complexity (dataset) :: float32; shape=(18000000,); filters=fillvalue=0.0
├─ anisotropy (dataset) :: float32; shape=(18000000,); filters=fillvalue=0.0
├─ class (dataset) :: int32; shape=(18000000,); filters=fillvalue=0
├─ class100 (dataset) :: int32; shape=(18000000,); filters=fillvalue=0
├─ hull_volume (dataset) :: float32; shape=(18000000,); filters=fillvalue=0.0
├─ images (dataset) :: uint8; shape=(18000000, 3, 128, 128); filters=compression=gzip,compression_opts=4,chunks=(1365, 3, 128, 128),fillvalue=0
├─ mat_complexity (dataset) :: float32; shape=(18000000,); filters=fillvalue=0.0
├─ mat_count (dataset) :: int32; shape=(18000000,); filters=fillvalue=0
├─ mat_slots (dataset) :: int32; shape=(18000000,); filters=fillvalue=0
├─ meta/ (group)
│  ├─ class100_id (dataset) :: |S50; shape=(100,); filters=fillvalue=b''
│  ├─ class100_name (dataset) :: object; shape=(100,); filters=fillvalue=b''
│  ├─ class_id (dataset) :: |S50; shape=(8,); filters=fillvalue=b''
│  ├─ class_name (dataset) :: object; shape=(8,); filters=fillvalue=b''
│  ├─ model_anisotropy (dataset) :: float32; shape=(1000,); filters=fillvalue=0.0
│  ├─ model_hull_volume (dataset) :: float32; shape=(1000,); filters=fillvalue=0.0
│  ├─ model_id (dataset) :: |S50; shape=(1000,); filters=fillvalue=b''
│  ├─ model_mat_complexity (dataset) :: float32; shape=(1000,); filters=fillvalue=0.0
│  ├─ model_mat_count (dataset) :: int32; shape=(1000,); filters=fillvalue=0
│  ├─ model_mat_slots (dataset) :: int32; shape=(1000,); filters=fillvalue=0
│  ├─ model_name (dataset) :: object; shape=(1000,); filters=fillvalue=b''
│  ├─ model_surface_area_ratio (dataset) :: float32; shape=(1000,); filters=fillvalue=0.0
│  ├─ model_vert_count (dataset) :: int32; shape=(1000,); filters=fillvalue=0
│  ├─ model_volume (dataset) :: float32; shape=(1000,); filters=fillvalue=0.0
│  ├─ model_volume_ratio (dataset) :: float32; shape=(1000,); filters=fillvalue=0.0
│  └─ model_xyz_size (dataset) :: float32; shape=(1000, 3); filters=fillvalue=0.0
├─ model_id (dataset) :: |S32; shape=(18000000,); filters=fillvalue=b''
├─ surface_area_ratio (dataset) :: float32; shape=(18000000,); filters=fillvalue=0.0
├─ vert_count (dataset) :: int32; shape=(18000000,); filters=fillvalue=0
├─ view_label (dataset) :: float32; shape=(18000000, 3); filters=fillvalue=0.0
├─ volume (dataset) :: float32; shape=(18000000,); filters=fillvalue=0.0
├─ volume_ratio (dataset) :: float32; shape=(18000000,); filters=fillvalue=0.0
└─ xyz_size (dataset) :: float32; shape=(18000000, 3); filters=fillvalue=0.0

spec:
================================================================================
目标:
  将 OmniFace 和 OmniShape 两个 H5 数据集转换为 HuggingFace datasets 格式
  (多个 parquet 分片文件), 达到类似 imagenet-1k 的存储与使用效果.

源数据概况:
================================================================================

1) OmniFace_202602042244.h5  (~69.3 GB)
   - N = 1,121,349
   - images: dtype=object, 每个元素是 JPEG 字节流 (gzip 压缩存储在 h5 中)
   - 已有 train/val 划分:
     * train_indices: int32, shape=(1059173,)
     * val_indices:   int32, shape=(62176,)
   - 标量标签字段 (N,): age, arousal, valence, expression, race, male, ...
   - 向量标签字段: gaze_dir(N,2), head_pose(N,3)
   - 字符串字段: id(N, |S200), origin(N, |S100), prompt(N, |S300)
   - 二值字段: is_sr(N, int8)

2) OmniShape1k_18000a_128x128_20251204.h5  (~124.4 GB)
   - N = 18,000,000
   - images: dtype=uint8, shape=(N, 3, 128, 128), gzip 压缩
   - 无预定义 train/val 划分 (需要自行划分)
   - 标量标签字段 (N,): class(int32), class100(int32), anisotropy, volume, ...
   - 向量标签字段: view_label(N,3), xyz_size(N,3)
   - 字符串字段: model_id(N, |S32)
   - meta/ 子组: 1000 个模型的元信息 (model_name, model_id, class_name, ...)

目标格式:
================================================================================
  output_dir/
  ├── OmniFace/
  │   ├── data/
  │   │   ├── train-00000-of-{M}.parquet
  │   │   ├── train-00001-of-{M}.parquet
  │   │   ├── ...
  │   │   ├── val-00000-of-{K}.parquet
  │   │   ├── ...
  │   │   └── val-{K-1:05d}-of-{K:05d}.parquet
  │   └── dataset_infos.json
  └── OmniShape/
      ├── data/
      │   ├── train-00000-of-{M}.parquet
      │   ├── ...
      │   ├── val-00000-of-{K}.parquet
      │   ├── ...
      │   └── val-{K-1:05d}-of-{K:05d}.parquet
      └── dataset_infos.json

关键设计决策:
================================================================================

1. 图片存储方式:
   - OmniFace: images 已经是 JPEG 字节流 → 直接存入 HF Image() feature
     (HF 内部会保留原始编码, 不会二次编码, 体积几乎不变)
   - OmniShape: images 是 uint8 原始数组 (N,3,128,128) → 存入 HF Image() feature
     (HF 内部会编码为 PNG, 128x128 小图 PNG 压缩率尚可, 但体积会比 gzip uint8 大)
     → 备选方案: 先手动编码为 JPEG 再传入, 控制体积
     → 最终决策: 使用 PIL 编码为 JPEG (quality=95) 再传入, 减小 parquet 体积

2. 分片大小:
   - HF 默认 ~500MB/分片, 我们沿用这个标准
   - OmniFace: ~69.3GB / 500MB ≈ 139 个 train 分片 + ~9 个 val 分片
   - OmniShape: 估算 parquet 体积后计算分片数 (取决于图片编码方式)

3. train/val 划分:
   - OmniFace: 直接使用 h5 中已有的 train_indices / val_indices
   - OmniShape: 按 model_id 不重叠划分 (同模型的所有视角必须在同一 split)
     → 先收集所有唯一 model_id, 随机打乱, 前 95% 归 train, 后 5% 归 val

4. 内存控制 (核心难点):
   - OmniShape 有 1800 万张 128x128 图, 全部加载到内存需要 ~821 GB → 不可行
   - 必须分批从 h5 读取, 每批构建一个 HF Dataset, 然后追加写入 parquet 分片
   - 策略: 按 h5 的 chunk 大小分批读取 (chunk=1365 张/批), 每批 ~0.7 MB 原始数据
   - OmniFace 的 JPEG 字节流也是按 chunk=1024 存储的, 同样分批读取

5. 字段映射:
   - bytes 类型字段 (|S200, |S100, |S32) → 解码为 str
   - object 类型字段 (model_name 等) → 直接作为 str
   - int8 → int32 (HF 对小整数类型的支持有限, 统一升格)
   - 所有数值字段保持原始 dtype (float32, int32, int64)

6. meta/ 子组处理 (仅 OmniShape):
   - meta/ 下的数据是模型级元信息 (1000 个模型), 不是样本级数据
   - 方案: 不放入主 parquet, 而是单独保存为 meta.parquet 或 meta.json
   - 训练时如果需要模型元信息, 可以通过 model_id 字段 join 查询


plan:
================================================================================

Phase 0: 环境准备
  - 确认依赖: h5py, datasets, pyarrow, Pillow, numpy
  - 确认输出磁盘空间充足 (至少 2x 源数据大小, 用于中间产物和最终 parquet)

Phase 1: 通用工具函数
  1.1 bytes_decode(bytes_val) -> str
      - 处理 |S200, |S100, |S32 等 numpy bytes → python str
  1.2 build_features_schema(fields_config: dict) -> Features
      - 根据字段配置自动构建 HF Features 对象
      - 处理 Image(), Value("string"), Value("int32"), Value("float32"),
        Sequence(Value("float32"), length=2/3) 等类型
  1.3 write_shards(dataset_builder, output_dir, split_name, max_shard_size_mb=500)
      - 通用分片写入函数
      - 接收一个生成器/迭代器, 每次产出一批 dict 数据
      - 累积到 max_shard_size 后写入一个 parquet 文件
      - 自动编号: {split}-{i:05d}-of-{total:05d}.parquet
  1.4 save_dataset_infos(output_dir, dataset_name, features, splits_info)
      - 生成 dataset_infos.json

Phase 2: OmniFace 转换
  2.1 定义字段映射
      - image:  JPEG 字节流 → Image() (直接传入 bytes, HF 自动处理)
      - id:     |S200 → Value("string")
      - origin: |S100 → Value("string")
      - prompt: |S300 → Value("string")
      - is_sr:  int8 → Value("int32")
      - 其余 int64 字段 → Value("int64")
      - float32 标量 → Value("float32")
      - gaze_dir (N,2) → Sequence(Value("float32"), length=2)
      - head_pose (N,3) → Sequence(Value("float32"), length=3)
  2.2 读取 train_indices / val_indices
  2.3 对每个 split:
      a. 按 h5 chunk (1024) 分批读取
      b. 每批构建 dict: {field_name: list_of_values, ...}
      c. image 字段: 直接传入 JPEG bytes (HF Image() 接受 bytes)
      d. 累积到 ~500MB 后调用 Dataset.from_dict() + to_parquet() 写入分片
      e. 打印进度: 已处理 X / 总数, 已写入 Y 个分片
  2.4 保存 dataset_infos.json

Phase 3: OmniShape 转换
  3.1 定义字段映射
      - image:  uint8 (3,128,128) → Image()
        → 先 np.transpose(chw→hwc), 再 PIL.Image.fromarray, 再编码为 JPEG bytes
        → 传入 HF Image() feature
      - model_id: |S32 → Value("string")
      - class: int32 → Value("int32")
      - class100: int32 → Value("int32")
      - float32 标量 → Value("float32")
      - int32 标量 → Value("int32")
      - view_label (N,3) → Sequence(Value("float32"), length=3)
      - xyz_size (N,3) → Sequence(Value("float32"), length=3)
  3.2 train/val 划分 (按 model_id 不重叠)
      a. 读取所有 model_id, 获取唯一值列表
      b. random.shuffle(unique_model_ids)
      c. 前 95% 的 model_id 对应的样本 → train
      d. 后 5% 的 model_id 对应的样本 → val
      e. 保存划分索引到 split_indices.json (可复现)
  3.3 对每个 split:
      a. 按 h5 chunk (1365) 分批读取
      b. 每批: images chunk → transpose → PIL → JPEG bytes
      c. 构建 dict, 累积到 ~500MB 后写入分片
      d. 打印进度
  3.4 保存 dataset_infos.json
  3.5 单独保存 meta/ 子组为 meta.parquet
      - 包含 1000 个模型的元信息
      - 字段: model_id, model_name, class_id, class_name, class100_id, class100_name,
        model_anisotropy, model_hull_volume, model_mat_complexity, ...

Phase 4: 验证
  4.1 用 load_dataset("parquet", data_files=...) 读取转换后的 parquet
  4.2 检查:
      - 样本数量是否一致
      - image 能否正常显示 (PIL Image)
      - 各字段 dtype 是否正确
      - train/val 数量是否符合预期
  4.3 随机抽取几张图片可视化, 与 h5_preview 结果对比

注意事项:
================================================================================
1. OmniShape 1800 万张图转换耗时预估:
   - 读取 h5 chunk + transpose + JPEG 编码: ~0.5ms/张
   - 总计: 18000000 * 0.5ms ≈ 2.5 小时 (单线程)
   - 可考虑多进程加速 (但要注意 h5 文件的并发读取限制)
2. 磁盘空间:
   - OmniFace parquet: ~70-80 GB (JPEG bytes 直接存, 几乎不膨胀)
   - OmniShape parquet: 取决于 JPEG 编码质量
     * quality=95: ~50-70 GB (JPEG 压缩 128x128 效果好)
     * quality=100: ~80-100 GB
   - 中间 Arrow 缓存: 需要额外空间 (但分批写入可控制)
3. HF Image() feature 的 bytes 输入:
   - 传入 JPEG bytes 时, HF 不会重新编码, 直接存储原始 bytes
   - 传入 PIL Image 时, HF 会编码为 PNG (体积更大)
   - 所以两个数据集都应该传入 bytes 而非 PIL Image
4. 大数据集的 Dataset.from_dict() 内存:
   - 每批 1024-1365 张图, 内存占用可控 (<1GB)
   - 写完一个分片后释放, 进入下一批

"""