# SGLang 多模态模型推理链路与优化思路

## 0. VLM统一范式

```text
Image / Video
   ↓
Preprocess
resize / dynamic resolution / tiling / frame sampling
   ↓
Vision Encoder
ViT / SigLIP / CLIP / InternViT / native ViT
   ↓
Visual Token Compression
patch merge / pooling / resampler / token pruning
   ↓
Connector
Linear / MLP projector / Q-Former / cross-attention adapter
   ↓
LLM Decoder
Dense Transformer 或 MoE Transformer
   ↓
Text / bbox / OCR / grounding / tool action
```

## 1. 概括

纯语言模型推理的主线是：

```text
text -> tokenizer -> input_ids -> token embedding -> LLM prefill/decode
```

多模态模型推理的主线是：

```text
text + image/video/audio
  -> multimodal processor
  -> input_ids + media features + grid/offset/mrope metadata
  -> vision/audio encoder
  -> scatter media embeddings into token embeddings
  -> LLM prefill/decode
```

也就是说，多模态模型不是把图片直接塞进 LLM，而是在 prefill 阶段先把图片、视频、音频编码成 embedding，再替换 prompt 中对应的多模态占位 token。

## 2. 和纯语言模型的核心差异

### 2.1 输入不再只有文本

语言模型请求通常只有：

```text
text / input_ids
```

多模态请求还会带：

```text
image_data / video_data / audio_data
modalities
```

OpenAI 兼容接口会从 message content 中抽取 `image_url`、`video_url`、`audio_url` 等媒体输入，然后构造成内部 `GenerateReqInput`。

### 2.2 tokenizer 之后还有 multimodal processor

纯文本链路：

```text
text -> tokenizer -> input_ids
```

多模态链路：

```text
text + media
  -> tokenizer + multimodal processor
  -> input_ids
  -> padded_input_ids
  -> mm_items
  -> image_grid_thw / video_grid_thw
  -> mrope_positions
  -> offsets
```

processor 负责：

- 加载图片、视频、音频。
- 调用 HF processor 或模型自定义 processor 做预处理。
- 生成 `pixel_values`、`pixel_values_videos`、`audio_features` 等模型输入。
- 计算图片或视频对应的 grid 信息，例如 `image_grid_thw`、`video_grid_thw`。
- 把一个 `<image>` 或 `<video>` 展开成大量占位 token。
- 记录每个媒体 item 在 prompt 中对应的 token 区间 `offsets`。
- 对 Qwen-VL 这类模型计算或传递 `mrope_positions`。

### 2.3 prompt 会被大幅展开

一个图片 token 不是一个真正的视觉 token。以 Qwen2.5-VL 为例，一张图片会被 processor 展开成一段视觉占位 token：

```text
"describe <image>"
  -> "describe" + [image_placeholder] * N
```

`N` 取决于图片尺寸、patch size、spatial merge size 等配置。

例如 512x512 图片常见会被 resize 到 504x504：

```text
patch_size = 14
grid_h = 504 / 14 = 36
grid_w = 504 / 14 = 36

ViT raw patch tokens = 36 * 36 = 1296
LLM visual tokens after spatial merge = 18 * 18 = 324
```

因此多图请求会显著增加 prefill 长度和视觉编码成本。

### 2.4 prefix cache key 必须感知媒体内容

纯文本模型的 prefix cache key 基本由 token ids 决定。

多模态场景中，下面两个请求文本相同，但图片不同，不能共享同一段 KV cache：

```text
"这张图是什么？" + image A
"这张图是什么？" + image B
```

SGLang 会给每个多模态 item 计算内容 hash，并派生一个超出词表范围的 `pad_value`。这个 `pad_value` 会替换 prompt 中对应的多模态 token，参与 radix/prefix cache 匹配。

比如：

```sh
image A pad_value = 200001234

padded_input_ids =
[..., 200001234, 200001234, 200001234, ..., 200001234, ...]
```



真正进入 embedding 层前，`pad_value` 会被 clamp 回词表范围，然后对应位置的 text embedding 会被视觉或音频 embedding 覆盖。

### 2.5 性能分布差异

多模态额外开销主要集中在 prefill：

- 媒体加载与预处理。
- vision/audio encoder。
- 多模态占位 token 展开。
- embedding scatter。
- 多维 RoPE / grid metadata 构造。

decode 阶段通常不重复跑 vision encoder，因为多模态 embedding 已经进入 LLM 的 KV cache。之后的 decode 更接近普通语言模型。

## 3. SGLang 多模态调用链路

整体调用链可以概括为：

```text
OpenAI/native request
  -> GenerateReqInput(image_data/video_data/audio_data)
  -> TokenizerManager._tokenize_one_request
  -> mm_processor.process_mm_data_async
  -> MultimodalProcessorOutput
  -> Scheduler.handle_generate_request
  -> MultimodalInputs.from_processor_output
  -> ScheduleBatch / ForwardBatch
  -> model.forward
  -> general_mm_embed_routine
  -> get_image_feature/get_video_feature/get_audio_feature
  -> scatter media embeddings into text embeddings
  -> language model forward
```

函数路径：

OpenAI message parsing:

- [process_content_for_template_format](../../python/sglang/srt/parser/jinja_template_utils.py#L123)
- [OpenAIServingChat](../../python/sglang/srt/entrypoints/openai/serving_chat.py#L159)
- [GenerateReqInput](../../python/sglang/srt/managers/io_struct.py#L138)

Tokenization and multimodal processor:

- [TokenizerManager._tokenize_one_request](../../python/sglang/srt/managers/tokenizer_manager.py#L782)
- [get_mm_processor](../../python/sglang/srt/managers/multimodal_processor.py#L44)
- [BaseMultimodalProcessor](../../python/sglang/srt/multimodal/processors/base_processor.py#L180)
- [BaseMultimodalProcessor.process_mm_data_async](../../python/sglang/srt/multimodal/processors/base_processor.py#L487)
- [QwenVLImageProcessor](../../python/sglang/srt/multimodal/processors/qwen_vl.py#L260)
- [QwenVL process_mm_data_async](../../python/sglang/srt/multimodal/processors/qwen_vl.py#L672)

Scheduler and batch:

- [Scheduler._try_apply_padded_mm_input_ids](../../python/sglang/srt/managers/scheduler.py#L1929)
- [Scheduler._maybe_compute_mrope_positions](../../python/sglang/srt/managers/scheduler.py#L1951)
- [Scheduler.handle_generate_request](../../python/sglang/srt/managers/scheduler.py#L1979)
- [MultimodalDataItem](../../python/sglang/srt/managers/schedule_batch.py#L244)
- [MultimodalInputs](../../python/sglang/srt/managers/schedule_batch.py#L465)
- [ForwardBatch.init_new](../../python/sglang/srt/model_executor/forward_batch_info.py#L606)

Model forward:

- [Qwen2_5_VLForConditionalGeneration](../../python/sglang/srt/models/qwen2_5_vl.py#L547)
- [Qwen2_5_VLForConditionalGeneration.get_image_feature](../../python/sglang/srt/models/qwen2_5_vl.py#L643)
- [Qwen2_5_VLForConditionalGeneration.get_video_feature](../../python/sglang/srt/models/qwen2_5_vl.py#L685)
- [Qwen2_5_VLForConditionalGeneration.forward](../../python/sglang/srt/models/qwen2_5_vl.py#L724)
- [general_mm_embed_routine](../../python/sglang/srt/managers/mm_utils.py#L1223)
- [embed_mm_inputs](../../python/sglang/srt/managers/mm_utils.py#L982)
- [Qwen2_5_VisionTransformer](../../python/sglang/srt/models/qwen2_5_vl.py#L262)
- [Qwen2_5_VisionTransformer.get_window_index](../../python/sglang/srt/models/qwen2_5_vl.py#L333)
- [Qwen2_5_VisionTransformer.rot_pos_emb](../../python/sglang/srt/models/qwen2_5_vl.py#L386)
- [Qwen2Attention](../../python/sglang/srt/models/qwen2.py#L108)
- [Qwen2Model](../../python/sglang/srt/models/qwen2.py#L267)

### 3.1 TokenizerManager 阶段

[`TokenizerManager._tokenize_one_request`](../../python/sglang/srt/managers/tokenizer_manager.py#L782) 先处理文本，然后如果请求包含媒体输入，会调用：

```text
mm_processor.process_mm_data_async(...)
```

processor 输出 `MultimodalProcessorOutput`，通常包含：

```text
input_ids
padded_input_ids：将<image>/<video>/<audio>展开后的input_ids
mm_items
mrope_positions
mrope_position_delta
token_type_ids：什么类型的 token
```

### 3.2 Scheduler 阶段

[`Scheduler.handle_generate_request`](../../python/sglang/srt/managers/scheduler.py#L1979) 会把 processor 输出转成 [`MultimodalInputs`](../../python/sglang/srt/managers/schedule_batch.py#L465)。

这一阶段会：

- 为每个媒体 item 设置 `pad_value`。
- 替换 prompt 中的多模态 token。
- 必要时补算 `mrope_positions`。
- 检查展开后的 prompt 长度。
- 把多模态信息挂到 `ForwardBatch.mm_inputs` 上。

### 3.3 Forward 阶段

多模态模型的 forward 通常调用统一 helper [`general_mm_embed_routine`](../../python/sglang/srt/managers/mm_utils.py#L1223)：

```text
general_mm_embed_routine(...)
```

它的核心步骤是：

```text
1. 判断当前是否需要处理多模态。
2. 按 modality 收集 mm_items。
3. IMAGE -> get_image_feature。
4. VIDEO -> get_video_feature。
5. AUDIO -> get_audio_feature。
6. 得到 media embeddings。
7. 用 pad_value 定位 prompt 中的媒体 token 区间。
8. 先生成普通 token embeddings。
9. 用 masked_scatter_ 把 media embeddings 覆盖到对应位置。
10. 调用语言模型主体。
```



## 4. 个人总结

+ Vision Encoder 相对 LLM 参数和计算较轻，但由于图像 patch 序列长，activation 和 TP 通信开销不小，TP 效率可能偏低。因此视觉侧更适合采用 request/image-level batching 或 data-parallel encoder，而不是简单沿用 LLM TP。
+ 对于小 batch ViT，大概率是 host-bound，CUDA graph 与跨请求 ViT batching 能改善利用率。在 Qwen2.5-VL-72B 上测试跨请求合并，满并发 8/128 分别带来 5%/9% 吞吐提升。





## 5. SGLang 多模态热点文件地图

这一节是代码阅读和性能排查时最常用的文件地图。可以先从“请求入口 -> processor -> scheduler -> model forward -> ViT/LLM”这五层找。

### 5.1 请求入口与 OpenAI 兼容层

[`OpenAIServingChat`](../../python/sglang/srt/entrypoints/openai/serving_chat.py#L159)

- OpenAI Chat Completions 入口。
- 把 OpenAI message 转成内部 generate 请求。
- 多模态内容最终会进入 `image_data`、`video_data`、`audio_data` 等字段。

[`process_content_for_template_format`](../../python/sglang/srt/parser/jinja_template_utils.py#L123)

- 处理 OpenAI message content 中的 `image_url`、`input_image`、`video_url` 等 part。
- 把模板可见内容规范化为模型 chat template 能消费的格式。

[`GenerateReqInput`](../../python/sglang/srt/managers/io_struct.py#L138)

- 定义请求与响应数据结构。
- `GenerateReqInput` 是 native generate 和 OpenAI 入口都会落到的内部请求对象。

### 5.2 Tokenizer 与 multimodal processor

[`TokenizerManager._tokenize_one_request`](../../python/sglang/srt/managers/tokenizer_manager.py#L782)

- `_tokenize_one_request` 是请求进入 tokenizer 和 multimodal processor 的关键入口。
- 纯文本请求在这里完成 tokenizer。
- 多模态请求会继续调用 `mm_processor.process_mm_data_async`。

[`get_mm_processor`](../../python/sglang/srt/managers/multimodal_processor.py#L44)

- 管理 processor 注册和选择。
- 根据 HF config architecture 找到对应 processor。
- 如果模型没有专用 processor，会走 transformers 兼容兜底路径。

[`BaseMultimodalProcessor`](../../python/sglang/srt/multimodal/processors/base_processor.py#L180)

- 多模态 processor 的基类。
- 处理媒体加载、hash、offset、feature 封装等通用逻辑。
- 性能上常见热点包括媒体 decode、resize、processor 调用和 token 展开。

[`QwenVLImageProcessor`](../../python/sglang/srt/multimodal/processors/qwen_vl.py#L260) / [`QwenVL process_mm_data_async`](../../python/sglang/srt/multimodal/processors/qwen_vl.py#L672)

- Qwen-VL / Qwen2-VL / Qwen2.5-VL 相关 processor。
- 负责生成 `pixel_values`、`image_grid_thw`、`video_grid_thw`。
- 负责 Qwen-VL 系列的视觉 token 展开和 M-RoPE 位置相关数据。

### 5.3 Scheduler、batch 和多模态数据结构

[`Scheduler.handle_generate_request`](../../python/sglang/srt/managers/scheduler.py#L1979)

- `handle_generate_request` 接收 tokenized request 并进入调度。
- `_try_apply_padded_mm_input_ids` 处理多模态 token padding。
- `_maybe_compute_mrope_positions` 在需要时补算 M-RoPE positions。

[`MultimodalDataItem`](../../python/sglang/srt/managers/schedule_batch.py#L244) / [`MultimodalInputs`](../../python/sglang/srt/managers/schedule_batch.py#L465)

- `MultimodalDataItem` 表示单个图片、视频或音频 item。
- `MultimodalInputs` 是请求级多模态容器。
- `pad_value`、`offsets`、`feature`、`image_grid_thw` 等都在这里串起来。

[`ForwardBatch.init_new`](../../python/sglang/srt/model_executor/forward_batch_info.py#L606)

- `ForwardBatch` 是模型 forward 看到的 batch 表示。
- `ForwardBatch.mm_inputs` 把 scheduler 阶段的多模态信息传到模型执行阶段。

[`MultimodalCache`](../../python/sglang/srt/mem_cache/multimodal_cache.py#L11) / [`MultiModalStaticCache`](../../python/sglang/srt/mem_cache/multimodal_cache.py#L76)

- 多模态 embedding cache。
- chunked prefill 和重复媒体输入优化时需要关注。

### 5.4 Forward 融合与 embedding scatter

[`general_mm_embed_routine`](../../python/sglang/srt/managers/mm_utils.py#L1223) / [`embed_mm_inputs`](../../python/sglang/srt/managers/mm_utils.py#L982)

- `general_mm_embed_routine` 是多模态模型 forward 的统一入口。
- `embed_mm_inputs` 负责调用模型的 `get_image_feature/get_video_feature/get_audio_feature`。
- `_get_chunked_prefill_embedding` 和 `_get_chunked_embedding_by_item` 处理 chunked prefill 下的多模态 embedding 复用。
- 性能上常见热点包括 media embedding cache miss、mask 构造、`masked_scatter_`。

### 5.5 Qwen2.5-VL 模型实现

[`Qwen2_5_VLForConditionalGeneration`](../../python/sglang/srt/models/qwen2_5_vl.py#L547) / [`Qwen2_5_VisionTransformer`](../../python/sglang/srt/models/qwen2_5_vl.py#L262)

- Qwen2.5-VL 的主实现文件。
- `Qwen2_5_VLForConditionalGeneration.forward` 接收 `mrope_positions` 并进入 `general_mm_embed_routine`。
- `get_image_feature/get_video_feature` 把跨请求图片或视频 concat 后喂给 `self.visual`。
- `Qwen2_5_VisionTransformer.forward` 是 ViT 主链路。
- `rot_pos_emb` 构造 ViT H/W spatial RoPE。
- `get_window_index` 构造 window attention 需要的索引和 `cu_window_seqlens`。

[`Qwen2Attention`](../../python/sglang/srt/models/qwen2.py#L108) / [`Qwen2Model`](../../python/sglang/srt/models/qwen2.py#L267)

- Qwen2/Qwen2.5 decoder 语言模型实现。
- `Qwen2Attention` 中 `qkv_proj -> split(q,k,v) -> rotary_emb -> attention` 是 LLM decoder RoPE 热点路径。

[`VisionAttention`](../../python/sglang/srt/layers/attention/vision.py#L875)

- Vision attention 实现。
- ViT full attention/window attention 的后端调用和 `cu_seqlens` 处理在这里会被使用。

### 5.6 RoPE 与 M-RoPE

[`RotaryEmbedding`](../../python/sglang/srt/layers/rotary_embedding/base.py#L75)

- 普通 1D RoPE cache 构造和应用。
- `cos_sin_cache` 的基本组织方式在这里。

[`MRotaryEmbedding`](../../python/sglang/srt/layers/rotary_embedding/mrope.py#L137)

- M-RoPE 实现。
- 处理 `[3, seq]` positions 和 `mrope_section`。
- Qwen2.5-VL decoder 的 `[16,24,24]` 分段逻辑在这里落地。

[`get_rope`](../../python/sglang/srt/layers/rotary_embedding/factory.py#L63)

- 根据 config 选择普通 RoPE、M-RoPE、YaRN 等 RoPE 变体。
- `rope_scaling` 中包含 `mrope_section` 时会创建 `MRotaryEmbedding`。

[`get_rope_index`](../../python/sglang/srt/layers/rotary_embedding/mrope_rope_index.py#L47)

- Qwen-VL、Qwen3-Omni、GLM4V、Ernie VL 等模型的 M-RoPE position id 构造逻辑。

### 5.7 NPU/后端相关热点

[`AscendAttnBackend`](../../python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py#L277)

- NPU attention 后端。
- 排查 NPU attention、paged attention、flash attention 类算子时常用。

[`init_npu_backend`](../../python/sglang/srt/hardware_backend/npu/utils.py#L95)

- NPU runtime 初始化和 torch_npu patch。
- 排查 NPU 环境和后端行为时常用。

[`_ProfilerTorch`](../../python/sglang/srt/utils/profile_utils.py#L282)

- profiler 工具封装。
- 做 NPU/CUDA profiling 时常会经过这里。
