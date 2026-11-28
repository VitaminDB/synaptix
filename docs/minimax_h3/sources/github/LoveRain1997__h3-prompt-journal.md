# LoveRain1997__h3-prompt-journal

Source: https://github.com/LoveRain1997/h3-prompt-journal
Fetched: 2026-11-28

Description: Case-study journal of MiniMax H3 prompt engineering experiments
Stars: 561  Updated: 2026-11-27T17:39:23Z



---

## FILE: README.md (https://github.com/LoveRain1997/h3-prompt-journal/blob/HEAD/README.md)

<div align="center">

<img src="docs/images/banner.jpg" width="720" alt="H3 Prompt Journal Banner">

# 🎬 H3 Prompt Journal

**MiniMax H3 视频/舞蹈提示词案例日记 —— 31 篇真实反推与实测，每篇附可直接粘贴的完整提示词。核心资产是[舞蹈动链 H3 库](./case-studies/2026-10-dance-motion-chain-library/)。**

[![Cases](https://img.shields.io/badge/case_studies-31-blue)](./case-studies)
[![Library](https://img.shields.io/badge/🎯_首选入口-舞蹈动链_H3_库_518+_链-orange)](./case-studies/2026-10-dance-motion-chain-library/)
[![Platform](https://img.shields.io/badge/model-MiniMax_H3-red)](https://github.com/MiniMax-AI/MiniMax-H3)
[![Results](https://img.shields.io/badge/🎬_成片演示-X_文章页-1DA1F2)](https://x.com/LoveUolanda/articles)

</div>

---

## 这是什么

H3 提示词工程的难点不在调参数，而在**教会模型一种关于"转场与连续性"的思维方式**。

大多数公开 H3 提示词只描述"每个镜头里该出现什么"，所以生成出来是支离破碎的片段；能跑通的提示词描述的是**镜头必须遵循的连续物理逻辑**——人物从不消失、动作跨段续接、切点吸附在音乐事件上。本仓库把每一次跑通的过程都记录下来：真实尝试 → 踩过的坑 → 突破口 → 最终可用的提示词。

> 🎬 所有案例均由这些提示词实际生成，成片演示统一收录在 [X 文章页 @LoveUolanda](https://x.com/LoveUolanda/articles)。

---

## 🎯 从这里开始：舞蹈动链 H3 库（本仓库第一优先级资产）

> **如果你只点一个链接，点这个 →** [`case-studies/2026-10-dance-motion-chain-library/`](./case-studies/2026-10-dance-motion-chain-library/) · Rev.1.12

<p align="center">
  <a href="./case-studies/2026-10-dance-motion-chain-library/">
    <img src="case-studies/2026-10-dance-motion-chain-library/cover.png" width="420" alt="舞蹈动链 H3 库封面">
  </a>
</p>

案例是**结果**，这座库是**原料**。31 篇案例里的每一次编舞，几乎都从这座库里取链、拼舞、再回写；它不是某一支视频的提示词，而是把动作拆成了**可以像积木一样首尾相接的标准件**——每条链一个 2–8 拍的完整编舞单元，带明确的**入态 / 出态**，英文描述可直接粘进 H3，拍位用**相对拍**因此能挂到任何 BPM 网格上。

| 库里有什么 | 规模 |
|---|---|
| **反推动链扩展库（K 章）** | **K1–K518 · 121 支实测片 + 2 图**：逐片附背景人物档案、onset 实测 BPM、AIGC 判定与变装机制归纳 |
| 古风 / 宅舞 / 抖舞动链 | **C1–C49 古风身韵 · A1–A23 宅舞 · B 章抖舞 · D 章多人配合 12 条纪律** |
| 成品可直接粘贴范本 | **H 章 13 支 I2VA 全片反推 · L 章 T2VA 卡点续写 · J 章 9 支写真式** |
| 摄影机语言标准 | **I 章 25 类物理视角定义 · E 章 10 条实测运镜纪律**（另有 [System A](./case-studies/2026-10-physical-camera-position-library/) / [System B](./case-studies/2026-10-camera-movement-discipline-library/) 独立整理版） |
| Hero 与技术峰值 | **G 章五大技术族 · Hero Lock 写法 · 实现度分级** |
| 零丢失报告档案 | **`reports/` 18 件**：模板 v2 + D001–D015 逐片全文 + 聊天信息回收 SUPPLEMENT |

**为什么它是第一优先级**

1. **可复用性最高**：案例只能看，链能直接抄。`- H3:` 行就是可粘贴的英文动作描述，替换主语/道具即可跨项目复用。
2. **防漂移的硬规则都在「0 使用规则」**：三层结构、动作×运镜 **1:1 同步六原则**、旋转预算 ≤2、动作链**十要素**、生成前自检十条、**时间窗动作链完整性**（先定窗口容量再排动作）——这些是本仓库所有稳定输出的共同底座。
3. **已验证**：案例 [025 古风六段](./case-studies/2026-10-baifeng-mengchen-classical-dance-six-segment/)、[028 原地古风十七段](./case-studies/2026-11-yueyoukui-in-place-guofeng-dance-mv-17-segment/)、[029 单人舞台剧二十一段](./case-studies/2026-11-index-music-dance-theatre-21-segment/)、[031 范夫人古风九段](./case-studies/2026-11-fan-furen-indoor-guofeng-solo-dance-9-segment/) 全部为**直编本库**产出。

**三步上手**：① 读「0 使用规则」里的**十要素**与**时间窗容量表** → ② 按舞种选链（古风走 C 章、宅舞走 A 章）→ ③ 用文末**拼舞模板**按"建立→发展→扩展→高光→呼吸→收束"填 bar 表，再把每条链的 H3 行贴进分段提示词。

> 📌 **入库机制变更（Rev.1.12）**：K 章 K1–K518 已**封账**，存量链照常取用；新反推内容不再拆链，改为按整片档案制收入 `reports/`（全时间轴 + 人物/片型档案 + onset BPM + AIGC 判定 + 全量提示词）。

---

## 每篇案例的结构

```
case-studies/YYYY-MM-short-name/
├── README.md      <- 背景、问题、突破口、可复用的经验
└── prompt.md      <- 完整最终提示词，可直接粘贴进 H3
```

## 创作思路：音乐先行，动作跟随

本仓库一半以上的创作都是**从一首现有的歌出发**：先拿到音乐 → 切出 8–15 秒的分段 → 用 onset 检测建立节拍/卡点网格 → 让每一段编舞、转场、换装、运镜全部吸附在这条时间轴上。**音乐时间轴是 H3 稳定性的根基**——一条精确的时间码，同时驱动人物动作峰值与摄影机响应峰值（1:1 同步），H3 才不会漂。

音频切割工具：**[LoveRain1997/audio-lyric-splitter](https://github.com/LoveRain1997/audio-lyric-splitter)** —— 按 SRT 歌词把整首歌切成 H3 可用的分段音频。

动作从哪来：**全部走[舞蹈动链 H3 库](./case-studies/2026-10-dance-motion-chain-library/)** —— 按舞种取标准链（古风 C 章 / 宅舞 A 章 / 抖舞 B 章），用入态-出态对接、用十要素写全动力链，再按 bar 表拼成一支舞。

**没有现成音乐怎么办**：必须先让 AI 写出一段**详细的 BGM 提示词**（曲风/速度/乐器/段落/重音位置，越具体越好），把它集成进视频提示词里，并**同时写明"动作由这段 BGM 驱动"**——BGM 里的重音就是动作的峰值，BGM 的乐句就是分段边界。没有时间轴（无论来自真实音频还是文字化 BGM 网格），H3 的动作与运镜都会失去锚点而变得不稳定。

## 快速上手

以实际提示词需要的参数为准——每个案例的 `prompt.md` 头部写明了该案例所需的输入（锚定图数量与分工、音频、目标视频等）与使用方式，按各案例说明准备素材即可；没有音频的案例会集成 AI 生成的 BGM 提示词并让动作跟随它。

核心纪律速览：**音乐时间轴驱动一切 · 每段自包含 · 图 1 零外貌 · 全局规则拼进每一段（禁止规则段压底） · 卡点=onset · 分段间局部特写+动作续接 · 转场写成"动作钩子+切点原因+下段续动"**。

---

## 🧭 案例导航

> 先看核心资产 → **[舞蹈动链 H3 库](./case-studies/2026-10-dance-motion-chain-library/)**（518+ 条标准动作链，案例编舞的原料）；下面是按月份与主题展开的 31 篇案例。

### 2026-08 · 基础篇：运镜、结构与多主体

| 预览 | 案例 | 简介 |
|:---:|---|---|
| <img src="docs/images/004-furina-mv.jpg" width="160"> | [004 · Furina Windsurfing Fashion MV](./case-studies/2026-08-furina-windsurfing-fashion-mv/) | 分层参考架构 + 文本驱动高潮：角色主视觉/三视图装备/分段运镜指南多图分工。🎬 [成片](./case-studies/2026-08-furina-windsurfing-fashion-mv/result-video.mp4) · [运镜指南图](./case-studies/2026-08-furina-windsurfing-fashion-mv/picture-3-segment1-cinematography-guide.png) |
| <img src="docs/images/006-sticker-comedy.jpg" width="160"> | [006 · Sticker Character Kitchen Comedy](./case-studies/2026-08-sticker-character-kitchen-comedy/) | 贴纸角色混入真实厨房的混搭媒介喜剧。[角色贴纸图](./case-studies/2026-08-sticker-character-kitchen-comedy/picture-1-sticker-character.png) |
| <img src="docs/images/007-cinematic-reveal.jpg" width="160"> | [007 · Cinematic Character Reveal (Two-Part)](./case-studies/2026-08-cinematic-character-reveal-two-part/) | 两段式电影感角色揭示，BGM 跨段连续不断。🎬 [动作参考](./case-studies/2026-08-cinematic-character-reveal-two-part/reference-motion-sample.mp4) · [姿态库](./case-studies/2026-08-cinematic-character-reveal-two-part/picture-1-pose-library.png) |
| <img src="docs/images/008-finger-dance.jpg" width="160"> | [008 · First-Person Finger-Controlled Dance](./case-studies/2026-08-first-person-finger-controlled-dance/) | 单图输入实现第一人称"手指操控"舞蹈。🎬 [参考舞步](./case-studies/2026-08-first-person-finger-controlled-dance/reference-dance-demo.mp4) |
| — | [001 · Three-Person Occlusion-Linked Orbital Long Take](./case-studies/2026-08-three-person-orbital-long-take/) | 三人遮挡联动的环绕长镜头：谁被挡住、谁先出列，全部写进物理逻辑。🎬 成片见 [X 文章页](https://x.com/LoveUolanda/articles) |
| — | [002 · Asymmetric Speed-Ratio Duo Choreography](./case-studies/2026-08-dual-subject-speed-contrast/) | 双人不对称速度比编舞：一快一慢同框共舞的节奏控制。🎬 成片见 [X 文章页](https://x.com/LoveUolanda/articles) |
| — | [003 · Micro-Cam Anchor-Flow Flight](./case-studies/2026-08-single-subject-three-pose/) | 单人三姿态微运镜：锚点流动式贴身飞行机位。🎬 成片见 [X 文章页](https://x.com/LoveUolanda/articles) |
| — | [005 · Water Obstacle Variety Show](./case-studies/2026-08-water-obstacle-variety-show/) | 综艺式水上障碍闯关：节拍锚定的即兴表演。🎬 成片见 [X 文章页](https://x.com/LoveUolanda/articles) |

### 2026-09 · 进阶篇：卡点、换装与连续舞蹈

| 预览 | 案例 | 简介 |
|:---:|---|---|
| <img src="docs/images/009-prank-cosswap.jpg" width="160"> | [009 · Three-Person Prank, Freeze Gag + Cos Swap](./case-studies/2026-09-three-person-prank-freeze-gag-cos-swap/) | 反推实战：三人恶作剧定格笑点 + 仅换皮肤的 Cos 换装。🎬 [成片](./case-studies/2026-09-three-person-prank-freeze-gag-cos-swap/result-video.mp4) |
| <img src="docs/images/010-snap-lock.jpg" width="160"> | [010 · Snap-Lock Editorial Reveal](./case-studies/2026-09-snap-lock-editorial-reveal/) | 音频吸附式 SNAP-LOCK 时尚揭示：两态相机 + 全程零外貌文本。🎬 [参考视频](./case-studies/2026-09-snap-lock-editorial-reveal/reference-video.mp4) |
| — | [011 · Song-Style Classical Dance (Six Segments)](./case-studies/2026-09-song-style-classical-dance-six-segment/) | 六段歌曲式身韵水袖舞：手递手英雄姿势阶梯 + 音乐不重启。🎬 成片见 [X 文章页](https://x.com/LoveUolanda/articles) |
| — | [012 · Editorial Lamp-Rhythm Fashion A/B](./case-studies/2026-09-editorial-lamp-rhythm-fashion-ab/) | 灯光节奏时尚片 A/B：暗奢甩裙 vs 甜美折脚跳，音乐驱动灯光切换。🎬 成片见 [X 文章页](https://x.com/LoveUolanda/articles) |
| — | [013 · Action-Chain Sock Fan Comedy](./case-studies/2026-09-action-chain-sock-fan-comedy/) | 25 状态物理状态机：脱袜-闻袜-扔镜头-弹回-光脚扇子舞的喜剧动作链。🎬 成片见 [X 文章页](https://x.com/LoveUolanda/articles) |
| — | [014 · Purple-Veil Pavilion Dance (Seven Segments)](./case-studies/2026-09-purple-veil-dance-seven-segment/) | 七段紫纱亭舞：驱动轮换编舞 + 细节切转场链 + 紫底暖键打光 + 永久面纱锁。🎬 成片见 [X 文章页](https://x.com/LoveUolanda/articles) |

### 2026-10 · 大成篇：多段长舞与转场取证

| 预览 | 案例 | 简介 |
|:---:|---|---|
| <img src="docs/images/016-waterfront-solo.jpg" width="160"> | [016 · Waterfront Solo Dance (Two Segments)](./case-studies/2026-10-waterfront-solo-dance-two-segment/) | 水岸白裙独舞：低阈值切镜检测(≤0.08)逐帧取证单镜头真伪 + 小节吸附切分 + 链记法动作续接。🎬 成片见 [X 文章页](https://x.com/LoveUolanda/articles) |
| <img src="docs/images/017-sock-showcase.jpg" width="160"> | [017 · Sock Showcase, Right-Entry Three-Shot](./case-studies/2026-10-sock-showcase-right-entry-three-shot/) | 三镜袜子展示：每个服装状态=独立镜头，画右一大步跨进入画、首帧已穿新装，跳切痕迹可见是特征。逐帧转场取证实录。🎬 成片见 [X 文章页](https://x.com/LoveUolanda/articles) |
| <img src="docs/images/018-leg-lift-contact-cut.jpg" width="160"> | [018 · Leg-Lift Contact-Cut Stocking, Six Looks](./case-studies/2026-10-leg-lift-contact-cut-stocking-six-look/) | 抬腿触地即切袜·六形态：永久支撑腿 + 单一活动腿，"脚触地"这一物理事件=剪辑触发器（CONTACT=CUT）。AnimateDiff 渐变升级为利落硬切。🎬 成片见 [X 文章页](https://x.com/LoveUolanda/articles) |
| — | [015 · Xiashan Veiled Solo (Twelve Segments)](./case-studies/2026-10-xiashan-veiled-solo-twelve-segment/) | 十二段面纱独舞：双图分权（图1服装/图2脸+背+纱）+ 链记法编舞 + 切点原因转场 + 醉酒式运镜（只有相机醉，人物永不醉）。🎬 成片见 [X 文章页](https://x.com/LoveUolanda/articles) |
| <img src="docs/images/019-identity-only-half-body.jpg" width="160"> | [019 · Identity-Only Half-Body Dance (Six Segments)](./case-studies/2026-10-identity-only-half-body-dance-six-segment/) | Identity-only 双图分权：图1只锁脸、图2只锁全身外观+环境，"图1半身像感"降级为纯镜头语言，开场零姿势继承（六段都从舞谱第一拍起跳）。111 BPM 六段连续可爱舞蹈，Motion Cut 全部切在未完成动作中间，Hero Action 专属镜头。🎬 成片见 [X 文章页](https://x.com/LoveUolanda/articles) |
| — | [022 · Classical Wuxia Dance (Thirteen Segments)](./case-studies/2026-10-classical-wuxia-dance-13-segment/) | 十三段真人古典武侠舞：每段 self-contained（同图 + 该段裁剪音频），节拍网格（90.5 BPM · 拍 0.663s / 小节 2.65s / 乐句 5.30s），段内 local 时间从 0 起算 + 音乐连续 hand-off + 参考图/武器状态硬保护（图中无剑则双手空），第 13 段纯器乐尾奏。附 20 条导演级方法论。🎬 成片见 [X 文章页](https://x.com/LoveUolanda/articles) |
| — | [023 · Kanzaki Sunny-Beach Sword-Dance MV (Eighteen Segments)](./case-studies/2026-10-kanzaki-beach-sword-dance-18-segment/) | 十八段真人海滩剑舞音乐 MV（230.034s · 139.5 BPM · 9 组 18 段）：图 1 人物+现场双锁（海滩不换景）；道具外观/状态双规则（Seg01–02 拔刀、Seg03–18 持刀+空鞘、全曲一把刀身一个鞘）；每段一镜到底 `[Shot 1]`、组间靠后期匹配剪辑；NO DEGREE LAW 物理机位、每段一个主运镜、景别服从当拍动作；纯表演式无对白无口型。附 11 章真人 MV 导演方法论 + 两次自检。🎬 成片见 [X 文章页](https://x.com/LoveUolanda/articles) |
| <img src="case-studies/2026-10-summer-pool-beach-waterline-mv-4-segment/cover.png" width="160"> | [024 · Summer Pool/Beach/Waterline MV (Four Segments)](./case-studies/2026-10-summer-pool-beach-waterline-mv-4-segment/) | 四段真人夏日 MV（55s · 9:16 竖屏 · 音频窗口 00:00-10/10-25/25-40/40-55）：图 1 只锁人物身份、不复制姿势/背景/机位，四段都重传同一张图；倾斜构图=高位斜俯+画面滚转+海岸线对角（非简单歪相机）；Fixed Floor-Level POV 贴地垂直仰拍；第三段唯一硬切精确 00:03.000、00:08 为连续后退跟拍不切；第四段 split-level waterline 半水上半水下 + 泡沫前景逐步揭示、一镜到底。硬切与镜头内连续运动分开写。🎬 成片见 [X 文章页](https://x.com/LoveUolanda/articles) |
| <img src="case-studies/2026-10-baifeng-mengchen-classical-dance-six-segment/cover.png" width="160"> | [025 · BaiFeng MengChen Classical Dance (Six Segments, Dual-Generation Cut)](./case-studies/2026-10-baifeng-mengchen-classical-dance-six-segment/) | 六段真人古风舞蹈 MV（79.882s · 105.25 BPM）：图 1 锁人物+完整环境、不重写外貌；从《舞蹈动链 H3 库》调用 C16/C17/C24/C28 等标准件重新编舞（进入路径/受力/峰值/惯性/回收）；第三段奉上完整 12.455s 原文提示词一条（不拆），实际以该原文双生成源片、手动剪辑，剪辑点全曲 33.916（local 6.958，Shot3结束/Shot4开始）；段内局部特写引导 vs 段间动作匹配分开；Dolly Zoom 只放关键失控点；人物称呼统一 Song Yu。附通用模板+五步手动剪辑流程。🎬 成片见 [X 文章页](https://x.com/LoveUolanda/articles) |

### 2026-11 · 商业 MV 篇：多来源合并与制作方法论

| 预览 | 案例 | 简介 |
|:---:|---|---|
| — | [026 · Index Morning Trouble Commercial MV (Eighteen Segments)](./case-studies/2026-11-index-morning-trouble-commercial-mv-18-segment/) | 十八段真人商业 Idol MV（224.888s · 178 BPM · 16:9）：四张参考图分权（人物／运镜 A／运镜 B 路径／居住环境）；多来源合并 18 段（Rev.8.2–8.6 + 独立重写稿）；SEG06 从早餐桌静物重构为 178 BPM 舞蹈优先段落并加裙内同色系打底层；魔法书脑内化（删除现实实体书）；Spoken 结构化 `<Subject 1>` + `<d>[Japanese]</d>`；17 处转场动作匹配。🎬 成片见 [X 文章页](https://x.com/LoveUolanda/articles) |
| — | [027 · One Image + One Song Duet Lip-Sync MV (Seventeen Segments)](./case-studies/2026-11-one-image-one-song-duet-lipsync-mv/) | 十七段真人古装奇幻爱情 folk 对唱 MV（215.420s · 147.5 BPM · 4:3）：一张双人定妆图（只锁人物身份与服装）+ 整首歌，歌词感知切成 17 段、零切词；Full-Reference Mode 六字段；严格按 JSON `vocal` 标签绑定 S1/S2，68 句（F43/M24/D1）只选 24 个对口型窗口，单人唱口单人可见、唯一 Duet 双人同框；唱口镜头稳定、空手表演；第 03/07/11/16 段无可见唱口、Seg16 纯器乐。 |
| — | [028 · YueYouKui In-Place Guofeng Dance MV (Seventeen Segments)](./case-studies/2026-11-yueyoukui-in-place-guofeng-dance-mv-17-segment/) | 十七段写实古风竖屏舞蹈 MV（207.520s · 133.93 BPM · 9:16）：图 1 绝对锁定（人物+环境+光线零重写，全片空手）；**原地舞蹈硬约束**——不走动、不横穿场景、不逼近镜头，脚步只允许点步/盖步/并步/轴心转身；直接从《舞蹈动链 H3 库》调用 C1/C2/C3/C5/C7/C12/C16 古风身韵链并改造为无道具原地版；三 Hero 分落 SEG07 点步翻身探海线 / SEG13 掖腿转 / SEG16 高位延展（全片旋转合计 2 次）；歌词只做情绪不做插画、不对口型；每段 Shot 块动作与机位同段同时码，段尾出态即下段入口。 |
| — | [029 · INDEX Music-Dance Theatre Solo Stage Performance (Twenty-One Segments)](./case-studies/2026-11-index-music-dance-theatre-21-segment/) | 二十一段写实单人音乐舞台剧 MV（252.648s · 170.15 BPM · 9:16）：图 1 全程唯一人物母版（零外貌重写）、图 2 取上一段真实导出末帧锚定开场姿态/运动向量；真·单人舞台（无工作人员/伴舞/观众），全程持续悬浮舞蹈、不在地面行走或起跳落地；全片仅 SEG17 一次头朝下垂直下潜、SEG18 专门恢复竖直，完整竖轴旋转合计 2 次（SEG06/13）；动链库 Rev.1.12 直编并为无支撑悬浮改编（删地面承重链、抽上身路径、浮空反向配重）；Rev.8.3 用 20 个「身体局部落点→硬切→同局部近景接住→沿身体路径揭示→回到本段舞蹈」切口做段间接力；情绪非歌词插画，口型仅 SEG07/12/17(Come on!)/19；两段一组生成、SEG21 尾声单独出。附低清样片。 |
| — | [030 · Basketball Girl Cheer Dance MV (Four Independent Segments)](./case-studies/2026-11-basketball-girl-cheer-dance-mv-4-segment/) | 四段日系 K-pop 篮球应援舞 MV（50.04s · 150.50 BPM · 9:16）：Picture 1 全程全参考（金发短发不变）；歌词感知切 4 段零切词、每段自包含独立生成后期拼接，段间靠**局部特写开场匹配**（手掌/前臂/指尖）不靠模型续接；全程**虚空篮球 mime**——接球弧形空间/单手带动运球/五阶段投篮 follow-through，禁实体球特效球轨迹；全片折腕禁令；醉酒式物理运镜（贴鞋 25–35cm 起机、遮挡揭示、无角度标签）；附 ComfyUI 本地生成工作流（MiniMaxH3ReferenceToVideo + AddGuide 续接组）。🎬 [成片](./case-studies/2026-11-basketball-girl-cheer-dance-mv-4-segment/result-video.mp4) · 工作流上手文见 [docs](./docs/segmented-generation-guide.md) |
| — | [031 · Fan Furen Indoor Guofeng Solo Dance (Nine Segments, Hero Director's Cut)](./case-studies/2026-11-fan-furen-indoor-guofeng-solo-dance-9-segment/) | 九段室内古风独舞 MV（86.592s · 105 BPM · 9:16 · 动作链与 Hero 导演版 Rev.1.1）：图 1 **人物与整个室内空间一起锁死**（明令禁止反打空间/墙面背后/画外结构，不重打光不改色温）；道具清零、空手舞蹈、歌词只做情绪不做道具清单、无歌词口型；每条动作按「预载支撑→发起→动力传递→峰值→惯性→制动→入口」七段式写；**四 Hero 技术族互不替代**——SEG03 点步翻身接探海 / SEG05 紫金冠式腾空腿弧 / SEG07 跪姿螺旋扫腿起身 / SEG09 斜向大跳落地终章延展；反旋转填充（禁无意义 360°、禁摄影机绕人代转）；SEG02–09 全部**局部特写开场**——前段末 0.3–0.5s 落一个身体局部、后段首帧该局部近景、0.9–1.25s 真实后退横移升降揭示，只做视觉匹配不做动作续接，Hero 前必恢复全身。🎬 [成片 360p](./case-studies/2026-11-fan-furen-indoor-guofeng-solo-dance-9-segment/result-video.mp4) |
| <img src="case-studies/2026-11-action-motion-transfer-tutorial/picture-1-travelers-prayer-ribbons.png" width="160"> | [教程 · Action Motion Transfer（反推→切片→ComfyUI 出片）](./case-studies/2026-11-action-motion-transfer-tutorial/) | **双人动作迁移完整教程**：反推 skill 拆解白模舞参考 → 按乐句边界切 4 段（5.0/5.0/5.0/6.32s）→ ComfyUI 工作流传人物参考图与段片、设时长分辨率逐段生成 → 校验拼接。含参考分权规则、节点地图、上传/时间/分辨率逐项参数表、八项成片校验清单与翻车点；附可直接拖入的 `comfyui-workflow.json`、反推 skill 全文与本案例四段成品提示词。 |

### 摄影机语言工具箱 · 双系统标准 ＋ 舞蹈动链 H3 库（★ 第一顺位）

> 本节第一行就是本仓库的**核心资产**——[舞蹈动链 H3 库](./case-studies/2026-10-dance-motion-chain-library/)，先看它，再看摄影机双系统。

| 封面 | 标准库 | 简介 |
|:---:|---|---|
| <img src="case-studies/2026-10-dance-motion-chain-library/cover.png" width="160"> | ★★ [Dance Motion Chain Library · 舞蹈动链 H3 库（第一顺位）](./case-studies/2026-10-dance-motion-chain-library/) | **动作链标准件总库（Rev.1.14，K1-K518 封账）**：0 使用规则（三层结构/动作×运镜 1:1 同步六原则/动作链十要素/旋转预算/生成前自检十条/时间窗动作链完整性）/ A 宅舞 23 链 / B 抖舞 / C 古风 49 链 / D 多人纪律 / E 运镜纪律 / F 同曲异构 / G Hero 库 / H 13 支成品范本 / I 25 类视角库 / J 写真 9 范本 / **K 批量反推扩展库（121 支片+2 图、518 条链：BPM 全部 onset 实测+置信分层、AIGC 判定全帧证据化、变装机制五型归纳、器乐/新疆舞/京剧武戏等语汇首录；自 2026-11-17 起 K 章封账，新反推内容按整片档案制入报告库不再拆链存储，档案内动作按 `Dxxx-Ayy` 编号并登记动作索引表）** / **L 成品 T2VA 范本（Miniskirt 12 Jobs 两版 + 8s 卡点续写）** / **M 整片档案索引（batch7 起，档案内动作编号 `Dxxx-Ayy`）**（batch7 收官 57/57：首批30+D组10+E组7+F组10 全部入库，见报告库 BATCH7 单文件版与续产单文件版） / **报告库（双 skill 全量档案模板 v2 + D001–D015 逐片档案 + BATCH7 首批 D016–D066 30 片进行中 + 聊天信息回收 SUPPLEMENT）** / 拼舞模板——每条链带入态/出态与十要素描述，积木式拼舞；案例 025/028/029/031 即用本库直编。本仓库只读副本，正本在本地 git。 |
| <img src="case-studies/2026-10-physical-camera-position-library/cover.png" width="160"> | [System A · Physical Camera Position Library](./case-studies/2026-10-physical-camera-position-library/) | **机位视角系统（摄影机在哪里）**：25 类视角物理定义块全文（贴地仰视/虫眼/正上方90°/荷兰角/贴身广角/防简化锁/8要素结构）。核心原则：不要命名效果（`low angle`），要定义产生效果的现实条件——机位一旦是"场景里的一件实物"，就无法被 H3 平均掉。 |
| <img src="case-studies/2026-10-camera-movement-discipline-library/cover.png" width="160"> | [System B · Camera Movement Discipline Library](./case-studies/2026-10-camera-movement-discipline-library/) | **动态运镜系统（摄影机怎么动、为什么动）**：E 章十条实测运镜纪律 + 物理运镜十法则（POSITION/NO DEGREE/SUBJECT-TRIGGER…）+ 十三词物理词库 + 运镜块五件套 + 反目录。核心纪律："动作负责舞蹈，摄影机负责观看"——每段一个主运镜行为，镜头只为重大空间变化而动。 |
| <img src="case-studies/2026-10-advanced-camera-combinations/cover.png" width="160"> | [Advanced Camera Movement Combinations](./case-studies/2026-10-advanced-camera-combinations/) | **物理运镜组合库（NO DEGREE VERSION）**：14 条 SEGMENT RULES + 13 词词库 + 10 组即用运镜组合（前随→侧移→越肩→回正 / 推进→遮镜→暗侧移→揭示 / 贴地滑移→起身 / Hero Lock→微推→触发响应…）。彻底取消度数语言：`ARC 40°` 淘汰，改为 START POSITION → PHYSICAL PATH → SUBJECT TRIGGER → END POSITION → FINAL FRAMING。 |
| <img src="docs/images/020-motion-transfer.jpg" width="160"> | [020 · H3 Motion Transfer — Subject-First Reconstruction](./case-studies/2026-10-h3-motion-transfer/) | **动作迁移（图1人物×目标视频动作×音频时序）**：三参考严格分权（图1=人+世界，视频1首帧=镜头几何，视频1正文=动作机制，音频1=时序权威），SUBJECT-FIRST 不可逆初始化顺序 + 八项 FINAL VALIDATION 自检。⚠️ 硬约束：**单次迁移最长 8–9 秒**，长舞按乐句拆段。附案例成片与 ComfyUI 工作流要点。🎬 [成片](./case-studies/2026-10-h3-motion-transfer/result-video.mp4) |
| <img src="docs/images/021-formation-lock.jpg" width="160"> | [021 · Four-Subject Formation Choreography (Fourteen Segments)](./case-studies/2026-10-four-subject-formation-choreography/) | 十四段四人群舞：逐字锚定块（Universal Visual Anchor）×14 次粘贴 + 动作态接力（OUTGOING HOOK→INCOMING STATE）+ 舞蹈化队形转换（脚先转/限象限/切镜代替拉远）。第 10 段完整英文提示词 + 14 段终版整合表。🎬 [SEG 10 成片](./case-studies/2026-10-four-subject-formation-choreography/assets/demo_segment10.mp4) |
| — | [Dance MV Production Skill Rev.1.0](./case-studies/2026-11-dance-mv-production-skill/) | **舞蹈 MV 通用制作方法论**：制作总公式（音乐结构→MV 概念→视觉母题→舞种语法→动作链→摄影机→Shot 切分→段间钩子→提示词→检查修订）+ 十条最高优先级 + 动作描述三档密度；覆盖参考图锁定、12 类舞种语法、动作链十要素、物理机位、剪辑连续性、提示词工程与交付标准，16 章可反复执行的商业工作流。 |




---

## 拉片研究 · 《凡人修仙传》

[`lapian-research/`](./lapian-research/) 是对 B 站 46 期《凡人修仙传》动画拉片视频的逐帧拆解与再整理：把讲解内容变成可检索、可复用的镜头规则与案例库，用的仍是本仓库这套「看片 → 拆解 → 落成规则」的反推思路。

<p align="center">
  <img src="./lapian-research/screenshots/casebook-overview.png" width="640" alt="拉片案例书界面截图：133 个案例、11 类导演意图、9 类手法类型、128 张连续帧、18 场场景蓝图的统计条，导演意图分布带与手法类型筛选条，以及插入镜头／拉轨变焦／小动作对比三张案例卡">
</p>

<p align="center"><i>拉片案例书首页：133 案例 · 11 类导演意图 · 9 类手法类型 · 128 张连续帧 · 18 场场景蓝图；按导演意图与手法类型筛选，每张案例卡含 UP 论点 / 导演问题 / 做法 / IF-THEN 决策规则 / 效果 / H3 写法。</i></p>

| 内容 | 说明 |
|---|---|
| [拉片案例书](./lapian-research/05_拉片案例书.html) | 133 个案例的视听语法库，单文件 HTML，浏览器直接打开，按导演意图筛选、逐帧查看 |
| 分析报告 · 文案记录 | 42 份逐期分析报告、43 份原文案（`视频分析报告/`、`文案记录/`） |
| 规则 · 抽帧 | 558+ 条 IF-THEN 导演决策规则；364 张连续帧抽帧图，按 46 个场景归档 |
| 60 秒介绍片 | 先讲案例书怎么查（118 案例 / 114 帧 / 15 场蓝图），再讲焦距、运镜叠加、越轴三个手法 |

目录、分镜与版权说明见 [`lapian-research/README.md`](./lapian-research/README.md)——该目录只保留研究用的静态抽帧与页面截图，不收录原片片段。

拉片数据来源：B 站 UP 主 **王十二** — <https://space.bilibili.com/20762167>

---

## 反推工作流（本仓库案例的生产方式）

每个 2026-09 之后的案例都经过同一套双 skill 流水线：

1. **probe**：ffmpeg 提取时长/帧率/音频波形，onset 检测输出卡点网格；
2. **五遍逻辑通读 + 低阈值切镜检测**（scene 阈值 0.27 探不到的转场，≤0.08 现形）；
3. **全帧率逐帧核对**：确认每个切点真伪、每个服装状态边界；
4. **复述确认** → 产出 H3 提示词 → 回测 → 收入日记。

配套工具：[`video-to-h3-prompt`](https://github.com/LoveRain1997/video-to-h3-prompt)（可复用的 H3 反推 skill）· 上游规范：[`h3-prompt-writing`](https://github.com/MiniMax-AI/MiniMax-H3/tree/main/skills/h3-prompt-writing)

## License

MIT — see [LICENSE](./LICENSE).


---

## FILE: case-studies/2026-08-cinematic-character-reveal-two-part/README.md (https://github.com/LoveRain1997/h3-prompt-journal/blob/HEAD/case-studies/2026-08-cinematic-character-reveal-two-part/README.md)

# Case Study 007 — Two-Part Cinematic Character Reveal with Continuous BGM

## Inputs

- `<Picture 1>` — Same character's complete pose library, body, costume, hairstyle, accessories, church environment, all photographed poses
- `<Picture 2>` — The ONLY facial identity reference for the same character
- Plus supplementary pose references in the same scene
- Reference BGM track (instrumental) and reference motion sample
- Target duration: **30 seconds total** (Part 1 0–15s + Part 2 15–30s), 16:9, 24fps
- Format: realistic photographic cinematography with high-end motion-graphics editing
- Setting: a believable, restrained British-Puritan-style church — same location throughout

## Media files

| File | Role |
|---|---|
| [`picture-1-pose-library.png`](./picture-1-pose-library.png) | Picture 1 — complete pose library, costume, church environment |
| [`picture-2-face-identity.png`](./picture-2-face-identity.png) | Picture 2 — facial identity lock for the same character |
| [`picture-1-additional-poses.png`](./picture-1-additional-poses.png) | Supplementary poses (same character, same church) |
| [`picture-1-supplementary-poses.png`](./picture-1-supplementary-poses.png) | More reference poses for Part 2 selection |
| [`reference-motion-sample.mp4`](./reference-motion-sample.mp4) | Reference motion / pacing sample |
| [`reference-bgm-tokyo-drifter.wav`](./reference-bgm-tokyo-drifter.wav) | Reference instrumental BGM (no vocals) |

## The problem

A character reveal is not one video. It's two.

Any character trailer longer than ~20 seconds faces a structural break: the first half must *introduce*, the second half must *establish*. These are different jobs. Trying to do both with one continuous prompt produces a flat, unmodulated video — a 30-second opening that never pays off.

But splitting into two separate prompts creates a different disaster: at the seam, the music restarts, the church changes, the lighting shifts, the character "rediscovers" themselves, and the audience feels two unrelated clips glued together instead of one trailer.

The constraint is harder than it sounds because the model's prior is to **reset** between generations. Anything you can name — character, pose, costume, church, palette, BGM key, reverb tail, room ambience — will subtly restart unless you explicitly tell it to keep going.

## The breakthrough

Split the prompt, but write the seam as a contract.

The two parts are issued as two independent prompts. To make them read as one trailer, the contract has six clauses, written into **both** prompts in matching language:

| Contract clause | Part 1 commitment | Part 2 commitment |
|---|---|---|
| Same character / costume / church | "The character remains a realistic photographic subject" | "Use exactly the same character, identity, costume, church established in Part 1" |
| Final-frame continuity | "Final frame should feel like a deliberate pause before the next 15-second section" | "Begin exactly from the final visual state of Part 1" |
| BGM continuity | "Do not create a musical ending at 15 seconds" + "keep harmonic and rhythmic momentum alive" | "Do NOT restart the music" + "exact harmonic and rhythmic state continues" |
| Sound design continuity | "Do not end the sound design at 15 seconds" | "Do not create a new ambience when Part 2 begins" |
| Non-repetition of shots | (implicit — first half) | "Do not repeat Part 1's exact shot mechanisms" + an explicit avoidance list |
| Pose library discipline | "Treat these poses as the ONLY approved major poses" | "Use the remaining approved poses from Picture 1 that were not emphasized in Part 1" |

The non-repetition list in Part 2 is worth highlighting. It doesn't just say "don't repeat." It names the six shot mechanisms Part 1 used — eye-only opening, face reveal, simple architectural reveal, simple hand-to-face match, same negative-space composition, same foreground wipe, same initial pose presentation — and tells the model to avoid them. Without this list, Part 2's first cut will reflexively copy Part 1's first cut, because the model's prior is "this is the structure of a character reveal."

The technique stack:

| Technique | Role |
|---|---|
| **Two-Prompt / One-Video Architecture** | Independent prompts bound by an explicit seam contract |
| **Mirror Contract Clauses** | Each continuity rule appears in BOTH prompts, phrased from that half's point of view |
| **Pose-Library Discipline** | Picture 1 poses are the *only* approved poses; never invent |
| **Face-Lock Separation** | Picture 2 = face only, Picture 1 = everything else (no merge, no redesign) |
| **Beat-Indexed Shot Tables** | 13 cuts per half with explicit time ranges and visual actions |
| **Non-Repetition Avoidance List** | Part 2 names Part 1's exact shot mechanisms to avoid |
| **Continuous-Sound Design** | Room reverb, cloth movement, whoosh vocabulary carry across seam |
| **Sister-Half Pacing** | Part 1 = establishment tempo; Part 2 = escalation → peak → resolve |
| **Final-Frame Hand-off** | Part 1 ends on a pause-state pose; Part 2 starts on that exact state |
| **Pixel-Priority Doctrine** | Face > eyes > hands > environment; never inverse |

## The named language

> **Two-Part Cinematic Reveal with Continuous BGM**

Three things packed into one name. "Two-Part" flags that this is a two-prompt workflow, not one. "Cinematic Reveal" flags the genre — character introduction, not action. "Continuous BGM" flags the hardest constraint, the one the model's prior most wants to violate: the music must not restart.

## 48-hour takeaway

Continuity is a contract, not an assumption.

Anything that needs to survive a prompt boundary — character identity, room ambience, BGM key, visual state — must be named in both prompts. Saying it once in Part 1 and hoping Part 2 inherits it is a recipe for a restart. The model's default between sessions is to begin clean. To defeat that default, you write the inheritance into the second prompt's opening section as a *new* rule the model will follow.

Three more principles worth keeping:

**1. Pose libraries beat pose invention.** A character that has 9–10 photographed poses in Picture 1 can do 30 seconds of varied shots without the model inventing a single new pose. The trick is to never say "show new poses." Say: "treat these as the ONLY approved major poses." This makes the library a constraint, not a suggestion.

**2. Non-repetition lists beat non-repetition platitudes.** "Don't repeat Part 1" is too vague. Listing the exact six shot mechanisms Part 1 used gives the model concrete things to avoid. Vague rules get vague compliance.

**3. Final-frame design is hand-off design.** Part 1's last shot is not a closing shot — it's a *pause-state pose* that Part 2 opens on. The last 1.4 seconds of Part 1 and the first 0.9 seconds of Part 2 are designed together as one continuous breath. The seam disappears when both ends of it are written as handoff.

## The final prompt

See [`prompt.md`](./prompt.md) for both Part 1 and Part 2 in full (usage notes in Chinese + structured English prompts).

The prompt file is organized as:

1. **Usage notes** — two-Picture architecture, BGM contract, non-repetition doctrine
2. **Part 1 (0–15s)** — Eye hook → face reveal → 9 poses with editorial cut grammar → pause-state hand-off
3. **Part 2 (15–30s)** — Continuation from Part 1 final state → escalation montage → Hero reveal → final key visual

Each part has its own beat-indexed shot table (13 cuts each, 26 total), its own negative-list, and matching continuity contract clauses for character / church / BGM / sound design.

## Result

A 30-second two-part character trailer that reads as one film:

- **Part 1 (0–15s):** Eye extreme close-up → face reveal → nine photographed poses strung together with hard cuts, graphic matches, architectural wipes, negative-space compositions, and a shutter-transition final pose held as a pause-state hand-off
- **Part 2 (15–30s):** Continues from Part 1's final pose, escalates tempo through a rapid detail montage and a five-beat character peak, resolves with a slow Hero Reveal into a clean final Key Visual — face locked to Picture 2, character large and dominant, restrained church ambience, sustained instrumental chord
- **Continuous BGM and room ambience** across the seam, with no musical restart, no new church, no character redesign
- **No shot mechanism repeats** between halves — Part 1's eye-hook / face-reveal / architectural-wipe grammar is explicitly absent from Part 2, which uses a different visual vocabulary (callback reinterpretation, rapid detail montage, architectural cut, rhythmic pose montage, momentary silence, peak build, Hero reveal)

## Tags

`#H3` `#prompt-engineering` `#character-reveal` `#two-part-workflow` `#continuous-bgm` `#pose-library` `#face-lock` `#beat-indexed-cuts` `#motion-graphics` `#church-aesthetic` `#cinematic-trailer`


---

## FILE: case-studies/2026-08-cinematic-character-reveal-two-part/prompt.md (https://github.com/LoveRain1997/h3-prompt-journal/blob/HEAD/case-studies/2026-08-cinematic-character-reveal-two-part/prompt.md)

# 高级电影感角色揭示蒙太奇「两段式 30 秒连续剪辑」提示词

## 使用方法

上传 **2 张参考图**：

- **`<Picture 1>`** = 同一角色的全部姿态库、身体、服装、发型、配饰、场景
- **`<Picture 2>`** = 同一人物的人脸锁定

将 30 秒切成两段：

- **Part 1（0–15s）** — 建立角色、神秘出场、逐步展示姿态
- **Part 2（15–30s）** — 延续视觉语言，节奏加快，姿态蒙太奇进入高潮与 Hero 收尾

两段都使用同一对 Picture 1 / Picture 2。两段之间必须遵守：

- **不重复同一个 CUT 设计**
- **不重复同一姿态的主要展示方式**
- **不重复同一种背景构图**
- 第二段直接承接第一段最后的状态
- **BGM 和环境音效连续，第二段不能重新起一首歌**
- 仍保留最初那种高级平面剪辑思路
- 不把人物变成卡通 / 插画
- 不远景，最大限度把像素留给人物

下面是两段可以**直接独立生成**的提示词。

---

## Part 1 — 0:00–0:15

```
Create a premium cinematic character reveal montage in 16:9, 24fps, total duration 15 seconds.

REFERENCE:

<Picture 1> = the SAME character's complete pose library, body, costume, hairstyle, accessories, environment and all photographed poses.

<Picture 2> = the ONLY facial identity reference for the SAME character.

Picture 2 must only lock and refine the face of the person already shown in Picture 1.

Do not create a second person.
Do not merge identities.
Do not redesign the character.
Do not reconstruct the character from scratch.

Preserve the exact face, body proportions, hairstyle, costume, materials, accessories, pose structure and photographic appearance from the references.

Picture 1 contains multiple existing poses. Treat these poses as the ONLY approved major poses for this character.

Do not invent unrelated full-body poses.

The purpose of this first 15-second section is to introduce the character through sophisticated cinematography and graphic editing, using the existing photographed poses as key visual states.

The character remains a realistic photographic human/cosplay subject throughout.

No cartoon rendering.
No illustration.
No cel shading.
No anime rendering.
No artificial fantasy color treatment.

VISUAL LANGUAGE:

Borrow the sophisticated editing grammar of the original high-end motion-graphics character trailer:

hard cuts,
graphic matches,
editorial crops,
split compositions,
foreground wipes,
architectural wipes,
snap reframing,
controlled push-ins,
shutter transitions,
stroboscopic photographic echoes,
poster-like framing,
negative space,
visual callbacks,
and strong graphic composition.

These techniques are applied to the PHOTOGRAPHIC IMAGE, not to the character's body.

Do not turn the character into flat graphics.

EYE OPENING — REQUIRED:

CUT 01 MUST begin with an extreme close-up of the character's eye.

The eye occupies most of the frame.

Use Picture 2 for precise facial identity.

Use Picture 1 for the surrounding hairstyle, veil, headpiece, lighting and costume context.

Subtle eyelid movement and a tiny eye movement create the first sense of life.

No glowing eye.
No artificial eye color change.

15-SECOND STRUCTURE:

CUT 01 | 0.00–0.90s — EYE HOOK:
Extreme close-up of one eye. Very shallow depth of field. Minimal movement. A restrained church ambience and a very quiet instrumental opening.

CUT 02 | 0.90–1.80s — FACE REVEAL:
Reveal more of the same face. Use a precise reframing rather than a large camera move. A veil, hair edge or architectural foreground element briefly crosses the frame as a natural wipe.

CUT 03 | 1.80–3.00s — POSE 01:
Reveal the first important photographed pose from Picture 1. Start with hands or costume detail, then reveal the pose with a short controlled camera move. Keep the character large.

CUT 04 | 3.00–4.10s — POSE 02:
Use a graphic match from Pose 01 into Pose 02. Match hand shape, head angle, sleeve contour, veil shape or another real visual geometry. Do not morph the body.

CUT 05 | 4.10–5.20s — POSE 03:
Reveal another photographed pose through church architecture. Use a column, arch or dark foreground structure to mask the transition. The new pose appears as the architectural element passes.

CUT 06 | 5.20–6.20s — PORTRAIT DETAIL:
Extreme close-up of the face from Pose 03. Eyes, lashes, veil and headpiece remain sharp and realistic. Briefly shift focus between eye and foreground detail.

CUT 07 | 6.20–7.30s — POSE 04:
Reveal another existing pose with a completely different composition. Use a three-quarter medium close-up. Keep the person dominant in frame.

CUT 08 | 7.30–8.30s — POSE 05 FRAGMENT:
Do not show the entire pose immediately. Rapidly reveal it through hand → sleeve → face → upper body → complete pose. All fragments must belong to the same photographed pose.

CUT 09 | 8.30–9.50s — POSE 06:
Abruptly switch to a clean negative-space composition. Reveal another existing pose. Use minimal visual effects. The contrast comes from silence and composition.

CUT 10 | 9.50–10.70s — POSE 07:
A faster editorial transition. Use two or three short reframings of the same existing pose: close face → hand → medium composition. Hard cuts synchronized to the music.

CUT 11 | 10.70–12.10s — POSE 08:
Use an architectural graphic match. A real line from the church architecture visually aligns with the character's costume trim, sleeve, veil or body contour, then cuts into Pose 08.

CUT 12 | 12.10–13.60s — POSE 09:
The editing begins to slow. Reveal a stronger and more iconic existing pose. Use a medium-full composition only if necessary, keeping the character large in frame.

CUT 13 | 13.60–15.00s — FIRST SECTION END:
Hold the final selected pose of this section long enough to establish a clear visual state that can be continued naturally in Part 2.

Do not create a final ending card.

Do not finish with a distant shot.

Do not abruptly freeze the character.

The final frame should feel like a deliberate pause before the next 15-second section.

CAMERA PRIORITY:

Extreme close-up,
close-up,
bust,
medium close-up,
medium shot,
occasional medium-full shot.

No extreme wide shot.
No establishing shot.
No tiny character inside the church.
No long pullback.

The character must occupy most of the available resolution.

PIXEL PRIORITY:

Face first.
Eyes second.
Hands and costume details third.
Environment fourth.

The church supports the character and never dominates the frame.

BACKGROUND:

Keep the same believable church environment from Picture 1.

Do not invent a new fantasy environment.

Transform the same location through framing:

columns,
arches,
sttained glass,
dark foreground structures,
architectural reflections,
cropped windows,
symmetrical details,
shallow blurred backgrounds,
and negative space.

Every CUT should feel compositionally different without breaking spatial continuity.

REFERENCE GRAPHICS:

If Picture 1 contains letters, Chinese characters, symbols, decorative lines, patterns or other intentional graphic details, they may be used as authentic graphic-design elements.

Use existing visual assets before inventing new ones.

No random words.
No fake logos.
No random Chinese characters.
No meaningless UI.

REALISTIC COLOR:

Preserve the natural photographic color relationships of Picture 1.

Keep the restrained cool white, pale blue, silver and subtle gold atmosphere already present.

No neon.
No artificial cyan.
No purple fantasy glow.
No cartoon color.
No painted color treatment.

MUSIC — CONTINUOUS MASTER TRACK:

Use a completely instrumental cinematic soundtrack inspired by the sacred, mysterious and ecclesiastical atmosphere associated with Index / British Puritan church imagery.

Use restrained organ, delicate bell resonance, chamber strings, low strings, subtle piano or celesta, dark orchestral harmony and restrained electronic pulse.

Absolutely no human voice.
No singing.
No lyrics.
No narration.
No chanting.
No choir vocals.
No vocal samples.

This is the BEGINNING of a longer continuous soundtrack.

Do not create a musical ending at 15 seconds.

At 14–15 seconds, keep harmonic and rhythmic momentum alive so Part 2 can continue seamlessly from this exact musical state.

SOUND EFFECTS — CONTINUOUS:

Maintain continuous church ambience throughout:

subtle room reverb,
soft cloth movement,
veil movement,
very restrained footsteps when appropriate,
small metallic details,
soft cinematic whooshes,
editorial shutter sounds,
and low-frequency transition impacts.

Do not end the sound design at 15 seconds.

The final sound should naturally lead into Part 2.

NEGATIVE:

No character redesign.
No face replacement.
No body deformation.
No pose invention.
No costume changes.
No hairstyle changes.
No extra characters.
No distant shots.
No extreme wide shots.
No repeated framing.
No generic fantasy environment.
No neon colors.
No cartoon colors.
No illustration.
No cel shading.
No glowing eyes.
No excessive VFX.
No particles covering the character.
No random typography.
No fake logos.
No human voice.
No singing.
No lyrics.
No narration.
No chanting.
No choir.

FINAL QUALITY:

Premium AAA character reveal × high-end editorial photography × sophisticated Japanese motion-graphics editing × restrained British church atmosphere.

The first 15 seconds should feel like a carefully directed opening chapter, not a complete standalone ending.

The final frame and soundtrack must naturally continue into Part 2.
```

---

## Part 2 — 0:15–0:30

```
Create the SECOND continuous 15-second section of the same premium cinematic character reveal in 16:9, 24fps.

IMPORTANT CONTINUITY:

This is NOT a new video and NOT a new scene.

It is the direct continuation of Part 1.

Use exactly the same character, same identity, same costume, same facial structure, same hairstyle, same church location, same photographic color treatment and same overall visual identity established in Part 1.

<Picture 1> remains the complete pose library.

<Picture 2> remains the ONLY facial identity reference.

Picture 2 only locks and preserves the face of the SAME character.

Do not redesign the character.

Do not introduce a new character.

Do not restart the visual language.

Do not repeat the same shots from Part 1.

The second section must feel like the camera and edit have continued moving forward from the final state of Part 1.

SECOND-SECTION PURPOSE:

Part 1 introduced and established the character.

Part 2 should increase the editorial tempo, reveal the remaining visual information more aggressively, create a sophisticated character montage climax, then resolve into a strong final hero image.

The visual language remains realistic photographic cinematography with advanced motion-graphics editing.

No cartoon.
No illustration.
No cel shading.
No artificial fantasy colors.

NON-REPETITION RULE:

Do not repeat Part 1's exact shot mechanisms.

Avoid repeating:

eye-only opening,
face reveal,
simple architectural reveal,
simple hand-to-face match,
same negative-space composition,
same foreground wipe,
same initial pose presentation.

Part 2 should feel like the same film evolving into a more energetic second movement.

CHARACTER MOTION:

The body remains faithful to the existing photographed poses from Picture 1.

Do not invent major new poses.

Do not morph one pose into another.

Create motion primarily through camera reframing, crop changes, foreground occlusion, editorial repetition, visual matches, snap transitions and subtle natural movement.

Use the remaining approved poses from Picture 1 that were not emphasized in Part 1.

15-SECOND STRUCTURE:

CUT 01 | 0.00–0.90s — CONTINUATION:
Begin exactly from the final visual state of Part 1.

Continue the same pose or its immediate visual aftermath for a brief moment.

The music and room ambience continue without restarting.

A small camera movement immediately carries the viewer forward.

CUT 02 | 0.90–1.90s — POSE 10:
Hard cut into a new existing pose from Picture 1.

Use a completely different composition from Part 1.

Prefer a close or medium-close framing.

Reveal the pose through a bold crop rather than a full-body presentation.

CUT 03 | 1.90–2.80s — VISUAL CALLBACK:
Use a photographic match to a detail from Part 1, but reinterpret it differently.

For example:

same sleeve shape,
different camera direction.

same veil curve,
different crop.

same hand position,
different depth.

The audience recognizes the visual connection without seeing the same shot again.

CUT 04 | 2.80–3.90s — POSE 11:
Reveal another approved pose from Picture 1.

Use a dynamic three-quarter camera move.

The character remains large in frame.

Do not pull away.

CUT 05 | 3.90–4.80s — RAPID DETAIL MONTAGE:
Use several extremely short fragments from existing poses:

eye,
hand,
fabric,
gold trim,
headpiece,
face.

These fragments appear on musical beats.

Do not introduce new body movement.

The fragments should feel like a high-fashion editorial montage.

CUT 06 | 4.80–5.90s — POSE REVISIT / NEW SCALE:
Return briefly to one previously shown pose, but completely change the framing and scale.

For example:

previously medium,
now extreme close-up.

Previously full body,
now face and hand.

The pose itself remains unchanged.

This creates visual callback without repetition.

CUT 07 | 5.90–7.00s — ARCHITECTURAL CUT:
Use a strong church architectural shape as a visual divider.

The camera passes behind a column, arch or dark foreground plane.

Emerging from the obstruction reveals another existing pose.

The transition should feel like a real camera move, not a digital morph.

CUT 08 | 7.00–8.10s — RHYTHMIC POSE MONTAGE:
Increase the editing speed.

Use three different existing poses in rapid succession.

Each one receives a distinct crop:

close face,
medium upper body,
medium-full silhouette.

Every cut lands on the music.

Do not use the same background framing twice.

CUT 09 | 8.10–9.20s — MOMENTARY SILENCE:
Suddenly reduce the cutting speed.

Choose one of the most emotionally distinctive existing poses.

Use a quiet close or medium-close composition.

Let the music briefly breathe.

Tiny eye movement or fabric movement gives the image life.

CUT 10 | 9.20–10.40s — BUILD:
Begin accelerating again.

Use short camera pushes, diagonal reframing and fast foreground transitions through another sequence of existing poses.

The viewer should feel that the film is approaching its final character reveal.

Do not introduce new fantasy effects.

CUT 11 | 10.40–11.70s — CHARACTER MONTAGE PEAK:
Create a rapid five-beat photographic montage:

Beat 1 — strongest face close-up.

Beat 2 — distinctive hand / costume detail.

Beat 3 — strongest silhouette pose.

Beat 4 — second strongest face angle.

Beat 5 — another recognizable pose.

The montage should feel like premium character campaign photography in motion.

CUT 12 | 11.70–13.20s — HERO REVEAL:
Slow down sharply.

Choose the single strongest pose from the entire Picture 1 pose library.

Reveal it progressively:

cropped detail
→ face
→ upper body
→ medium-full composition.

Use the architecture to frame the character elegantly.

The character occupies most of the frame.

Allow the viewer time to appreciate the face, costume, silhouette and pose.

No explosion.

No excessive VFX.

No distant environment.

CUT 13 | 13.20–15.00s — FINAL KEY VISUAL:
Resolve into the definitive final character image.

Use the strongest existing pose from Picture 1.

Do not invent a new pose.

The character is large, sharp and dominant.

The face is locked to Picture 2.

Hair, veil and clothing have subtle residual movement.

The camera gently settles into its final composition.

Use restrained church ambience and a final sustained instrumental chord.

Do not zoom away.

Do not reveal a giant environment.

Do not introduce new characters.

Do not end with random graphics.

The final frame should feel like the official character key visual after a 30-second premium reveal.

MUSIC — CONTINUOUS FROM PART 1:

This is the SAME uninterrupted instrumental soundtrack from Part 1.

Do NOT restart the music.

Do NOT introduce a new song.

Do NOT change to a different musical style.

The exact harmonic and rhythmic state from the end of Part 1 continues naturally into this section.

The second section becomes gradually more rhythmic and emotionally intense from 0–11 seconds, reaches the strongest instrumental peak around CUT 11–12, then resolves elegantly during CUT 13.

Still absolutely no human voice.

No singing.
No lyrics.
No narration.
No chanting.
No choir vocals.
No vocal samples.

SOUND DESIGN — CONTINUOUS:

Church ambience continues seamlessly from Part 1.

Room resonance remains consistent.

Cloth movement, veil movement, subtle footsteps, cinematic whooshes and editorial transition sounds continue naturally.

The sound effects should feel like one uninterrupted recorded sound world.

Do not create a new ambience when Part 2 begins.

Do not use an artificial "new scene" sound.

Each cut may have subtle rhythmic impacts, shutter sounds and controlled whooshes, but they must remain part of the same acoustic environment.

FINAL CAMERA RULE:

Keep the camera close.

Prioritize:

extreme close-up,
close-up,
bust,
medium close-up,
medium,
occasional medium-full.

No extreme wide shot.

No distant final shot.

No tiny character in the church.

No long pullback.

No infinite zoom.

The character owns the available pixels.

FINAL IDENTITY:

Picture 1 = exact body / costume / pose library.

Picture 2 = exact face identity.

Same character from beginning to end.

No redesign.

No deformation.

No pose invention.

No identity drift.

FINAL ART DIRECTION:

Part 1 was the character introduction.

Part 2 is the visual escalation and final establishment.

They must feel like two halves of ONE professionally edited character trailer.

The second half should be more rhythmic and concentrated, not simply repeat the first half.

End with the character's strongest existing pose as a clean, elegant, realistic photographic key visual.

No distant ending.

No random graphics.

No artificial color treatment.

No human voice.

Exactly 15 seconds.
```


---

## FILE: case-studies/2026-08-dual-subject-speed-contrast/README.md (https://github.com/LoveRain1997/h3-prompt-journal/blob/HEAD/case-studies/2026-08-dual-subject-speed-contrast/README.md)

# Case Study 002 — Asymmetric Speed-Ratio Duo Choreography

## Inputs

- `<Picture 1>` — Two characters together: Kokomi (left) and Qiqi (right)
- Single reference image, both identities locked from it
- Target: butterfly-step dance to instrumental BGM (Paradise-style)
- Prompt language: **Chinese**

## The problem

H3 synchronizes dancers to the beat.

Give the model two people and music, and it defaults to mirror-sync: both bodies lock to the same tempo, the same downbeat, the same phrasing. The brief wanted the opposite — Kokomi fast and continuous, Qiqi slow and delayed, a "master leads, student follows with a lag" dynamic. But "fast" and "slow" are adjectives. The model rounded them to "same speed, slightly different."

## The breakthrough

Speed difference is not a feeling. It is a **ratio**.

Instead of describing tempo qualitatively, the prompt gives the model an explicit arithmetic anchor:

> When Kokomi has completed **3 actions**, Qiqi has completed **1**.

This converts an aesthetic tempo gap into a checkable count. The model cannot round "3:1" to "roughly equal." It has to hold the asymmetry or visibly fail the rule.

The second lever is **behavioral motivation**. Qiqi isn't just "slow" — she watches Kokomi, then imitates the action she just saw. This gives the lag a narrative cause rather than an arbitrary speed cap, which makes the desynchronization feel intentional instead of broken.

The technique stack:

| Technique | Role |
|---|---|
| **Count-Ratio Anchor** | 3:1 action count — the concrete rule the model must hold |
| **Watch-Then-Imitate Logic** | Gives the slow subject a reason for the lag |
| **Asymmetric Speed Choreography** | The overall temporal desynchronization pattern |
| **Close-Range Continuous Orbit** | Camera stays tight on both faces throughout |

## The named language

> **Asymmetric Speed-Ratio Duo Choreography**

Naming the pattern as a *ratio* rather than a *feeling* is the key. "One fast, one slow" is subjective. "3:1 count-locked" is a specification.

## 48-hour takeaway

Adjectives don't survive generation. "Fast" becomes "normal." "Slow" becomes "normal." The model's default is convergence toward the beat.

To hold a visible difference, convert the aesthetic intent into a **number the model can track**. A 3:1 action-count ratio is harder to ignore than "one person is faster." And to make the asymmetry feel deliberate rather than defective, give the slower subject a **reason** — she's watching, then copying. Lag with motivation reads as choreography. Lag without motivation reads as a bug.

## The final prompt

See [`prompt.md`](./prompt.md) for the full prompt (Chinese, ~170 lines).

The prompt is organized in this order:

1. **Reference lock** — Picture 1 is the sole reference, both identities frozen
2. **Dance type** — butterfly steps, light footwork, side-to-side
3. **Asymmetric speed rule** — Kokomi fast/continuous, Qiqi slow/one-at-a-time
4. **Count-ratio anchor** — the explicit 3:1 action-count specification
5. **Behavioral logic** — Qiqi watches, then imitates with a delay
6. **Negative constraints** — no sync, no catching up, no sudden acceleration
7. **Camera** — close continuous orbit, waist-to-head framing
8. **Audio** — instrumental BGM only, no vocals

## Result

A duo dance where Kokomi completes roughly 3 butterfly-step actions for every 1 that Qiqi completes, sustained from first second to last, with Qiqi visibly watching and lagging behind throughout.

## Tags

`#H3` `#prompt-engineering` `#choreography` `#asymmetric-speed` `#count-ratio` `#duo-dance` `#desynchronization` `#butterfly-step`


---

## FILE: case-studies/2026-08-dual-subject-speed-contrast/prompt.md (https://github.com/LoveRain1997/h3-prompt-journal/blob/HEAD/case-studies/2026-08-dual-subject-speed-contrast/prompt.md)

不要唱歌 模仿极乐净土的蝴蝶步和bgm

我输入的 Picture 1 是唯一参考图片。

图片里有两个人：

左边是心海。
右边是七七。

严格按照 Picture 1 保持两个人的脸、发型、服装和人物特征。


舞蹈：

两个人跳少女蝴蝶步。

但是两个人的动作速度完全不同。


心海：

心海跳得非常快。

心海连续不断地做动作。

心海一个动作结束后，马上开始下一个动作。

心海在很短的时间里连续完成很多个蝴蝶步动作。


七七：

七七必须明显慢很多。

七七一次只做一个动作。

七七做完一个动作以后，才慢慢开始下一个动作。

七七不能连续快速做动作。

七七不能追上心海。


最重要的画面效果：

当心海已经连续完成三个动作的时候，

七七只完成了一个动作。


例如：

心海：
第一个动作 → 第二个动作 → 第三个动作

七七：
第一个动作

然后：

心海已经开始第四个动作，

七七才开始第二个动作。


所以整个视频里：

心海一直快速连续跳很多动作。

七七只慢慢完成少量动作。

心海的动作数量明显比七七多。

七七永远跟不上心海。


不要让七七和心海同步。

不要让七七跟着音乐快速连续跳。

不要让七七追上心海。

不要让七七突然加速。

七七必须明显慢。

这种“一人快速连续跳，一人明显慢慢跟着学”的差异必须从第一秒一直保持到最后。


七七的状态：

七七看着心海。

心海快速做动作。

七七看完以后，才慢慢模仿刚才的动作。

当七七还在完成这个动作时，心海已经连续完成了好几个新动作。

七七始终落后很多。


舞蹈动作：

少女蝴蝶步。

轻快小步。

左右交替脚步。

交叉脚步。

侧向移动。

换重心。

不要康康舞大踢腿。


运镜：

镜头一直贴着两个人。

进行非常近距离的连续环绕。

摄影机从正面绕到侧面，再绕到另一侧。

不要远距离环绕。

不要大全景。

不要全身镜头。

主要拍：

腰部到头顶。

需要看到腿部动作时：

膝盖到头顶。

看到膝盖以后不要继续后退。

保持近距离继续环绕。

环绕过程中不断推近两个人的脸。

保持两个人的脸清楚。


AUDIO:

最终音频只有纯音乐 BGM。

只有无歌词的背景音乐。

人物没有任何声音。

不要生成任何人声。

不要生成歌曲。

不要生成歌词。

不要生成对白。

不要生成哼唱。

最终只有：

INSTRUMENTAL BGM ONLY.

---

## FILE: case-studies/2026-08-first-person-finger-controlled-dance/README.md (https://github.com/LoveRain1997/h3-prompt-journal/blob/HEAD/case-studies/2026-08-first-person-finger-controlled-dance/README.md)

# Case Study 008 — Single-Image First-Person Finger-Controlled Dance

> **用手指控制 AI 跳性感卡点舞｜H3 单图手势控舞教程**

## Inputs

- `<Picture 1>` — The ONLY visual reference. Locks dancer's appearance, face, hairstyle, costume, environment, lighting, and the original high-angle smartphone perspective
- Target: vertical 9:16 first-person smartphone recording, continuous single shot
- Prompt language: **English** (descriptive blocks + literal command clauses)
- Output: short-form viral dance video controlled by a foreground finger

## Media files

| File | Role |
|---|---|
| [`reference-dance-demo.mp4`](./reference-dance-demo.mp4) | Original AnimateDiff补帧 demo clip used as the visual reference source |

## What this prompt solves

A solo dancer + a controlling hand in the same frame, and the hand must visibly *cause* the dance.

Two problems collide here. **First**, the model's prior for "dance" is to freelance — once a dancer is on screen, the model improvises choreography it considers aesthetically pleasing, ignoring any external cue. **Second**, a hand in the foreground is almost always a problem: H3 will either grow it to dominate the frame, slide it across the center to cover the face, or shrink it to a barely visible background element. None of these work when the hand is supposed to be the puppeteer.

The third hidden problem: most prompt authors would write a list of dance moves (hip sway, arm wave, hair flip) and let the model sequence them. That fails because the model doesn't *see* a connection between the hand in the foreground and the dancer's body, so the two drift out of sync, or the hand disappears during the most important beat.

## The breakthrough

State the control relationship as a literal mapping, then forbid alternatives.

The prompt establishes an explicit, one-to-one control grammar at the top:

```
UP → WOMAN RISES.
DOWN → WOMAN LOWERS.
LEFT → WOMAN MOVES LEFT.
RIGHT → WOMAN MOVES RIGHT.
```

Each direction is then given a paragraph defining exactly how the body responds (torso extension for UP, hip drop for DOWN, full-body weight shift for LEFT/RIGHT) and — crucially — a negative clause: `LEFT and RIGHT must not become tiny hip twitches`. Without this, the model would interpret "left/right movement" as subtle hip sways. The prompt forces a *visible full-body shift*.

The same anti-drift logic applies to the hand itself:

- A connected forearm / wrist / palm / fingers chain (no floating phantom hand)
- Lower-right quadrant only, 10–18% of image area (not a giant, not a sliver)
- Hand never extends across the center, never covers the face or torso

The dance description is intentionally generic — `confident, sexy viral short-form influencer dance` — because the prompt's purpose is the *control contract*, not the choreography. The usage note at the top of the article tells the user: swap the dance style freely, swap the command sequence freely, but never break the finger-to-body mapping.

The technique stack:

| Technique | Role |
|---|---|
| **Literal Control Mapping** | UP/DOWN/LEFT/RIGHT as one-to-one body-direction commands |
| **Directional Anti-Twitch Clause** | Forbids subtle hip-sway interpretation of LEFT/RIGHT |
| **Foreground Element Quota** | 10–18% image area, lower-right quadrant only |
| **No-Drift Composition Locks** | Camera distance 1.0–1.2m, body 70–80% of frame, no zoom-out |
| **Sync-Within-Beat Rule** | No reaction delay, no one-beat lag, no anticipation |
| **Command Sequence Table** | Pre-defined 10-beat sequence (UP→DOWN→L→R→L→R→UP→DOWN→L→R) |
| **Modular Dance Section** | Dance style is a swappable description, not a hard-coded choreo |
| **Closing Triplet Pose** | L→R→UP with a held final pose as the resolution |

## The named language

> **Finger-Controlled First-Person Dance**

The name encodes the three things that make this prompt work: the hand is the *controller* (not a prop), the camera is *first-person* (not a third-party observer), and the dancer is *in the shot* (not a stock figure pasted in). Lose any of these and the prompt collapses into either an unconstrained dance video or a hand-in-foreground demo with no relationship between them.

## 48-hour takeaway

When a prompt has two visible actors, name their *relationship* explicitly.

Most prompt authors describe each actor's appearance and actions separately and hope the model connects them. It doesn't. A dancer description and a hand description, written in isolation, become two independent scenes that happen to share a frame. The connection only appears if the prompt *itself* contains the connection — and the simplest way to write a connection is a literal mapping: hand goes up, body goes up; hand goes left, body goes left.

Three reinforcing rules make this stick:

**1. Define each direction with a body response, not just a label.** "LEFT means the body moves left" is too abstract. Specify the body parts that move: weight, torso, hips — and how. The model will follow mechanics more reliably than abstract directions.

**2. Negate the cheap interpretation.** "LEFT/RIGHT must not become tiny hip twitches" prevents the model from satisfying the prompt with the easiest possible interpretation. Always include the wrong-but-allowed version in the negative list.

**3. Quota the foreground element.** A 10–18% area constraint for the hand prevents both the "giant hand" failure (model inflates the controller) and the "barely visible hand" failure (model shrinks it). Specifying the range is much more reliable than saying "natural size."

## The final prompt

See [`prompt.md`](./prompt.md) for the full usage notes + ready-to-paste prompt.

The prompt file is organized as:

1. **Usage notes** (Chinese) — single-Picture workflow, swappable dance style, unbreakable control rules
2. **integrated_multimodal_description** — Picture 1 reference scope, high-angle POV, hand placement quota, body distance lock
3. **Control grammar** — UP/DOWN/LEFT/RIGHT with body-response definitions
4. **Anti-twitch / anti-drift clauses** — for body and hand separately
5. **Dance style description** — generic viral-dance vocabulary, swappable
6. **Command sequence** — 10-beat hard-coded sequence
7. **Closing triplet** — L → R → UP final pose
8. **Soundscape + music** — indoor ambience + viral dance track

## Result

A continuous first-person smartphone recording, locked to the high-angle perspective of Picture 1:

- A hand in the lower-right foreground issuing clear UP/DOWN/LEFT/RIGHT gestures
- The dancer immediately following each gesture with full-body movement (rise, drop, side-shift, side-shift)
- No phantom hand, no floating fingers, no hand covering the face, no distant dancer
- Ten gesture-movement pairs on rhythm, ending in a held confident pose
- Music: viral dance track with punchy beats synchronized to the gesture timing

## Tags

`#H3` `#prompt-engineering` `#first-person-pov` `#finger-control` `#dance` `#short-form` `#single-image` `#control-mapping` `#anti-drift` `#viral-dance` `#chinese-tutorial`


---

## FILE: case-studies/2026-08-first-person-finger-controlled-dance/prompt.md (https://github.com/LoveRain1997/h3-prompt-journal/blob/HEAD/case-studies/2026-08-first-person-finger-controlled-dance/prompt.md)

# 用手指控制 AI 跳性感卡点舞｜H3 单图手势控舞教程

## 使用说明

**只输入图片 1**，然后直接导入下面整段提示词。

* 图片 1：唯一视觉参考，负责人物外貌、服装、环境、光照、构图和原图摄影视角。
* 不需要输入第二张动作图或手部参考图。
* 可以直接修改提示词里的 **DANCE STYLE** 和动作顺序，自定义舞蹈。
* 但不要破坏核心控制关系：**手指向上/下 → 人物上下变化；手指向左/右 → 人物左右晃动。**
* 想换成其他舞蹈，只修改舞蹈描述，不要删除手势控制和同步规则。

## 完整提示词

```text
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

integrated_multimodal_description: [Shot 1] A continuous live-action first-person smartphone recording based entirely on <Picture 1>. <Picture 1> is the only visual reference for the adult female dancer's appearance, face, hairstyle, hair color, costume, body proportions, environment, room layout, wall, floor, lighting, color and initial spatial composition. Preserve the original visual identity and environment of <Picture 1>.

The camera closely reproduces the distinctive perspective of <Picture 1>: a strong high-angle first-person smartphone POV looking substantially downward at the woman. The camera is clearly above her eye level. The top of her head, shoulders and upper body are visibly seen from above. Keep substantial ceiling visible in the upper part of the frame. Do not use an eye-level view, third-person view, orbiting camera or cuts.

A single adult arm belonging to the camera operator enters naturally from the lower-right foreground. Show a connected forearm, wrist, palm and fingers. The hand is a moderate-size foreground element, not a giant object. Keep the hand primarily within the lower-right quadrant, occupying approximately 10–18% of the image area. The hand must never extend across the center, cover the woman's face or cover her torso. The woman remains the primary visual subject and the hand is only the foreground controller.

Keep the woman physically close to the smartphone, approximately 1.0–1.2 meters away. Her full body occupies approximately 70–80% of the vertical frame. Do not pull the camera backward, do not place her deep in the room and do not progressively make her smaller.

The foreground index finger directly controls the woman's dance movement. The woman does not independently freestyle. The hand is the visible movement cue and the woman's body immediately follows it.

Maintain these exact control relationships throughout the entire video:

UP → WOMAN RISES.
DOWN → WOMAN LOWERS.
LEFT → WOMAN MOVES LEFT.
RIGHT → WOMAN MOVES RIGHT.

For UP, the index finger moves clearly upward and the woman's body immediately rises. Her torso extends upward, her posture becomes taller, her arms naturally rise and her center of gravity moves upward.

For DOWN, the index finger moves clearly downward and the woman's body immediately lowers. Her knees bend slightly, hips lower, center of gravity drops and torso lowers into a controlled dance position.

For LEFT, the index finger moves toward screen-left and the woman's body immediately shifts and sways toward screen-left. Her weight, torso and hips follow the same direction. The entire body movement must be clearly visible.

For RIGHT, the index finger moves toward screen-right and the woman's body immediately shifts and sways toward screen-right. Her weight, torso and hips follow the same direction. The entire body movement must be clearly visible.

LEFT and RIGHT must not become tiny hip twitches. The woman's body must visibly move from side to side. Keep the horizontal movement simple so the connection between finger direction and body direction remains unmistakable.

The hand gesture and corresponding body movement are tightly synchronized. The woman's movement begins as the corresponding finger gesture begins and develops within the same musical beat. No noticeable reaction delay, no one-beat delay, no anticipation and no independent movement.

The dance is a confident, sexy viral short-form influencer dance: energetic rhythmic hip movement, controlled waist isolation, subtle torso waves, attractive shoulder movement, pronounced but tasteful hip accents, confident posture, playful facial expression, direct eye contact and natural hair movement. The dance should feel like a polished viral social-media dance challenge. Make the movement sexy, stylish, rhythmic and visually captivating while keeping it non-explicit.

During LEFT and RIGHT movements, maintain the direct horizontal hand control while adding sensual hip accents. During UP and DOWN movements, maintain the direct vertical hand control while adding smooth waist and body movement. Never sacrifice hand-to-body control for complicated choreography.

Use a fast modern viral dance rhythm with strong clear beats.

Command sequence:
UP → DOWN → LEFT → RIGHT → LEFT → RIGHT → UP → DOWN → LEFT → RIGHT.

Each command produces one clearly visible corresponding body movement. Do not add unrelated freestyle movements, spins, walking or complicated footwork.

The index finger moves with strong, deliberate gestures. The wrist and forearm naturally follow the finger. Use short, clear upward, downward, leftward and rightward gestures. Do not use weak floating gestures. Keep the hand in the lower-right foreground and never allow it to dominate the center of the frame.

Maintain one continuous handheld smartphone recording with subtle natural camera instability. Preserve the strong high-angle perspective and close composition throughout the entire shot. Do not zoom out, pull backward, lower the camera or lose either the hand or the woman's full body.

At the end, use LEFT → RIGHT → UP. The woman performs a left body movement with a sensual hip accent, a right body movement with a sensual hip accent, then rises into a confident final dance pose. The hand remains in the lower-right foreground and does not move into the center. Hold the final pose briefly.

overall_soundscape: Natural indoor room ambience, subtle foot movement, fabric movement and quiet smartphone handling sounds, preserving the acoustic character of <Picture 1>.

non_diegetic_music: A modern viral short-form dance track with strong punchy beats, crisp percussion and clear rhythmic accents. Each major beat provides a timing point for the hand gesture and its corresponding dance movement.
```


---

## FILE: case-studies/2026-08-furina-windsurfing-fashion-mv/README.md (https://github.com/LoveRain1997/h3-prompt-journal/blob/HEAD/case-studies/2026-08-furina-windsurfing-fashion-mv/README.md)

# Case Study 004 — Layered Reference Architecture with Text-Driven Climax

## Inputs

- `<Picture 1>` — Character key visual / first frame (identity, face, hair, costume)
- `<Picture 2>` — Character turnaround + windsurfing equipment reference (board, sail, mast, boom)
- `<Picture 3>` — Segment-specific cinematography guide (used in segments 1 & 2 only)
- Target: **3 × 15-second segments** → 45-second vertical 9:16 fashion-sports music video
- Post-production: full video audio → Suno remix → final music-video sync
- Prompt language: **Chinese** (design doc) + **English** (H3 prompts)

## The problem

H3 treats all reference images as equal-weight visual targets.

Give it three pictures and it tries to reproduce all three simultaneously — the character, the equipment sheet, and the camera angle all become "things to match." The result is a compromised compromise: the character looks like a turnaround model, the camera framing fights the text instructions, and the equipment renders as a flat illustration rather than a 3D prop in motion.

Worse, when the project needs a **cinematic climax** — a windsurfing aerial jump at the musical peak — a cinematography reference image becomes a cage. The model tries to match the guide image's composition instead of executing the dynamic hero-shot timing the text describes.

## The breakthrough

Give each picture a **job title**, and when the job is done, **fire the picture**.

### Layer 1 — Role-segregated reference architecture

Instead of three equal-weight references, each picture gets a specific role:

| Picture | Role | Controls | Does NOT control |
|---|---|---|---|
| **Picture 1** | First Frame Visual Anchor | Identity, face, hair, costume, color, opening frame state | Camera, composition, equipment structure |
| **Picture 2** | Character + Equipment Reference | Body construction, costume structure, footwear, board, sail, mast, boom, rigging | Storyboard, camera, motion |
| **Picture 3** | Cinematography Reference | Camera position, framing, lens language, movement direction, subject scale, composition | Character appearance, equipment |

The prompt explicitly tells H3: *"Picture 2 is not a storyboard. Do not reproduce the turnaround-sheet layout."* This prevents the model from flattening a 3D scene into a 2D character sheet.

### Layer 2 — Text-driven climax (drop Picture 3 for the final segment)

The third segment — the hero moment where the windsurfer launches off a giant wave — uses **only Picture 1 + Picture 2**. No Picture 3.

This is not because of cross-segment visual memory (each 15s clip is independently generated). It's because a cinematography guide image would **over-constrain the camera** during the most complex action sequence:

```
giant wave → ride up wave face → takeoff → airborne → hero shot → landing → fashion ending
```

When the text needs to precisely time camera stops, music peaks, and spray explosions to specific seconds, a guide image fights the text. Removing it lets the text be the sole director.

### Layer 3 — Music-driven timeline structure

The 15-second timeline is not divided evenly. It's divided by **musical function**:

```
0–3s    Build-up
3s      Main Drop
3–10s   High-energy
6–7s    Mini-turnaround
9–10s   Absolute visual peak
10–12s  Release
12–15s  Next phrase
```

The rule: **music's accent decides the action; music's release decides the camera ease.** The 9–10s hero shot must align with the strongest musical accent — camera stops aggressive movement, stabilizes into a low-angle telephoto hero frame, and holds.

### Layer 4 — H3 + Suno split workflow

The final music is not generated per-segment. Instead:

```
H3 generates 3 segments → edit into full video → extract full audio → Suno remixes complete audio → final music placed back → final sync edit
```

This gives Suno the complete musical structure (build-up → drop → development → climax → ending) rather than three disconnected 15-second fragments.

The technique stack:

| Technique | Role |
|---|---|
| **Role-Segregated Reference** | Each picture has a specific job, preventing equal-weight compromise |
| **Text-Driven Climax** | Drop Picture 3 for the final segment; let text be the sole director |
| **Music-Driven Timeline** | BPM-anchored segmentation; accents decide actions |
| **H3 + Suno Split** | H3 owns visual; Suno owns final complete music |
| **Hero Shot Timing Lock** | Camera stops at musical peak; holds the iconic frame |

## The named language

> **Layered Reference Architecture with Text-Driven Climax**

"Layered" tells the model: these images are not equal — each has a role. "Text-Driven Climax" tells the model: when the action peaks, text is the only director. Together they solve the two failure modes — reference-image compromise and climax constraint — in one concept.

## 48-hour takeaway

A reference image is an employee, not a partner. Give it a job title, tell it what it does NOT control, and when its job is done, dismiss it.

The most important experiment in this project was **removing Picture 3 from the final segment**. The instinct is to add more references for more control. But during a climax — when camera, action, music, and spray all need to hit specific marks at specific seconds — a guide image becomes a constraint that fights the text. Less reference input gave more output control.

And for multi-segment music videos: **generate all video first, then remix the complete audio.** Feeding Suno the full 45-second audio structure produces a coherent musical arc; feeding it three isolated 15-second clips produces three unrelated tracks.

## Media files

This case includes the original reference images and generated result:

| File | Role |
|---|---|
| [`picture-1-character-key-visual.png`](./picture-1-character-key-visual.png) | Picture 1 — Character key visual / first frame anchor |
| [`picture-2-turnaround-equipment.png`](./picture-2-turnaround-equipment.png) | Picture 2 — Character turnaround + windsurfing equipment reference |
| [`picture-3-segment1-cinematography-guide.png`](./picture-3-segment1-cinematography-guide.png) | Picture 3 — Segment 1 cinematography guide (0–15s) |
| [`picture-3-segment2-cinematography-guide.png`](./picture-3-segment2-cinematography-guide.png) | Picture 3 — Segment 2 cinematography guide (15–30s) |
| [`result-video.mp4`](./result-video.mp4) | Generated result video |

Note: Segment 3 (the final hero moment) intentionally has **no Picture 3** — camera control is text-only.

## The final prompt

See [`prompt.md`](./prompt.md) for the full production document (~1465 lines, Chinese design doc + three English H3 prompts).

The document is organized in this order:

1. **Project structure** — 9:16 vertical, fashion-sports hybrid, visual energy curve
2. **Multi-image lesson** — role segregation for Pictures 1/2/3
3. **The segment-3 exception** — why Picture 3 is dropped for the climax
4. **Music design** — BPM 128, 4/4, timeline function map
5. **Segment 1 prompt** — high-speed windsurfing entry (Picture 1+2+3)
6. **Segment 2 prompt** — large wave zone approach (Picture 1+2+3)
7. **Segment 3 prompt** — final hero moment, text-only camera control (Picture 1+2, no Picture 3)
8. **Segment comparison table** — reference usage per segment
9. **Why text-only for segment 3** — cinematography language upgrade rationale
10. **Post-production workflow** — H3 → full video → extract audio → Suno → final sync
11. **Music-video sync map** — which visual beat aligns with which musical function
12. **Complete workflow diagram** — end-to-end production flowchart
13. **Four retained lessons** — reference separation, Picture 3 per-segment, music as director, H3+Suno split

## Result

A 45-second vertical fashion-sports music video produced in three independent 15-second H3 segments, with the final climax segment using text-only camera control and the complete soundtrack remixed by Suno from the full video's extracted audio.

The signature frame: Furina cosplayer airborne above a breaking wave, transparent blue sail fully extended, camera frozen at the musical peak in a low-angle telephoto hero composition.

## Tags

`#H3` `#prompt-engineering` `#multi-image` `#layered-reference` `#text-driven-climax` `#music-driven-timeline` `#windsurfing` `#fashion-mv` `#suno` `#vertical-video` `#hero-shot`


---

## FILE: case-studies/2026-08-furina-windsurfing-fashion-mv/prompt.md (https://github.com/LoveRain1997/h3-prompt-journal/blob/HEAD/case-studies/2026-08-furina-windsurfing-fashion-mv/prompt.md)

# 芙宁娜「帆板写真式音乐 MV」完整制作方案

## H3 三段生成 → 完整视频合成 → 提取完整音频 → Suno 重制配乐

下面这版按照**实际制作流程重新整理**，纠正前面两个关键点：

1. **第三段 H3 生成时不使用图3，纯文字控制运镜与高潮。**
2. **最终配乐不是分别处理每15秒音乐，而是先把完整视频的音频提取出来，再将完整音频交给 Suno 进行重新生成/重制，最后把正式配乐放回完整视频。**

---

# 一、项目整体结构

这次项目不是简单的「角色运动视频」，而是：

> **真人 COS 写真 + Windsurfing 帆板运动 + 电子音乐 + 体育摄影 + 音乐高潮**

最终希望形成一条完整的视觉能量曲线：

```text
人物定妆
↓
进入海面
↓
高速帆板
↓
运动强度提升
↓
更大的浪
↓
最终高潮
↓
巨浪冲浪
↓
帆板腾空
↓
Hero Shot
↓
落水
↓
写真式收尾
```

整体采用：

* **9:16 竖屏**
* 真人写实 COS
* 高质量体育摄影
* 长焦为主
* 近距离拍摄
* 少用远距离全景
* 保证人物占据较大的画面面积
* 让有限的视频像素集中在人物、服装、帆板和关键动作上

---

# 二、这次最重要的 H3 多图经验

## 1. 三张图不是简单“等权参考”

实际工作中，我们把三张图分别定义成：

```text
Picture 1
人物定妆 / 第一帧

Picture 2
人物三视图 + 帆板装备

Picture 3
当前段落的拍摄指导图
```

其中：

### Picture 1

负责：

* 人物身份
* 真人 COS 外貌
* 脸部
* 发型
* 服装
* 色彩
* 第一帧状态

---

### Picture 2

负责：

* 人物正面
* 侧面
* 背面
* 服装结构
* 鞋
* 帆板
* 帆
* 桅杆
* Boom
* 装备结构

必须明确告诉 H3：

> **Picture 2 不是分镜图。**

---

### Picture 3

负责：

* 摄影机位置
* 构图
* 长焦/近景关系
* 运镜方向
* 人物在画面中的位置
* 镜头距离
* 运动摄影语言

而实际经验是：

> **视觉拍摄指导图对运镜构图的约束非常强。**

因此不能只在文字里写：

> “摄影机低机位环绕……”

然后给 H3 一张完全不同构图的指导图。

图和文字应该尽量表达同一个摄影意图。

---

# 三、一个非常重要的例外：最终第三段

这次实际采用的最终方案是：

```text
Picture 1
+
Picture 2
+
纯文字
```

**没有 Picture 3。**

原因不是“上一段的图3会影响下一段”。

三个15秒片段本身是**独立生成**的，不存在跨段视觉记忆。

真正的原因是：

> **第三段需要完全由文字重新定义摄影机运动和高潮过程。**

如果第三段继续给一张具体的运镜指导图，那么这张图本身就会成为非常强的视觉构图约束。

而最终第三段我们希望文字精确控制：

```text
接近巨浪
↓
冲上浪壁
↓
起跳
↓
腾空
↓
音乐高潮
↓
Hero Shot
↓
落水
```

因此最终15秒采用：

```text
图1 = 人物
图2 = 三视图
文字 = 摄影 + 动作 + 节奏
```

这也是这次项目最有价值的实验结果之一。

---

# 四、音乐节奏设计

原始音乐分析得到：

**BPM ≈ 128**

**4/4**

整体属于：

> Chiptune / Synthpop / Electronic

核心律动：

* Four-on-the-floor
* Syncopation
* Kick
* Hi-hat
* Snare / Clap
* Synth Bass
* Lead Synth

视觉上最重要的节点：

```text
0–3s
Build-up

3s
Main Drop

3–10s
高能量段

6–7s
Mini-turnaround

9–10s
最强视觉高潮

10–12s
Release

12–15s
下一乐句
```

所以不能把15秒平均处理。

应该让：

> **音乐的重音决定动作，音乐的释放决定镜头缓和。**

---

# 五、第一段：高速帆板进入状态

## 使用方式

上传：

```text
Picture 1
人物定妆照

Picture 2
人物三视图 + 帆板装备

Picture 3
第一段拍摄指导图
```

第一段的任务：

> **让观众进入这个世界。**

不是一开始就把最大的浪拿出来。

视觉：

```text
海面
↓
人物
↓
帆板
↓
加速
↓
水花
↓
高速跟拍
```

---

## 第一段完整 H3 提示词

```text
H3 Ref2VA Prompt — Furina Windsurfing Fashion MV — Segment 1

REFERENCE ROLES:

<Picture 1> is the FIRST FRAME VISUAL ANCHOR.

Begin directly from the visual state shown in Picture 1.

Preserve the same adult realistic female cosplayer, facial identity, hairstyle, blue eyes, realistic skin, navy decorative crown, navy-and-white Fontaine-inspired costume, gold ornamental details, blue ribbons and translucent blue-white flowing fabric.

<Picture 2> is the CHARACTER AND WINDSURFING EQUIPMENT REFERENCE.

Use Picture 2 only to preserve the character construction, costume structure, footwear, windsurf board, mast, boom, transparent blue sail and rigging.

Do not treat Picture 2 as a storyboard.
Do not reproduce the turnaround-sheet layout.

<Picture 3> is the CINEMATOGRAPHY REFERENCE for this 15-second segment.

Use Picture 3 to guide the camera position, framing, lens language, camera movement, subject scale and general composition.

Do not literally reproduce graphic arrows from the guide image.

VIDEO:

Create a 15-second vertical 9:16 realistic cinematic fashion music video.

The visual style combines realistic professional windsurfing sports photography with elegant fashion editorial photography.

Use predominantly close and medium telephoto compositions.

Keep the character large and highly detailed in frame.

Avoid distant establishing shots.

The ocean should remain recognizable but secondary to the character.

MUSIC STRUCTURE:

Approximately 128 BPM.
4/4 electronic chiptune / synthpop.

0–3 seconds:
BUILD-UP.

3–10 seconds:
MAIN DROP.

6–7 seconds:
MINI-TURNAROUND.

10–12 seconds:
RELEASE.

12–15 seconds:
TRANSITION.

TIMELINE:

0–1s:

Begin directly from Picture 1.

Maintain the character's appearance and visual identity.

<Subject 1> is already windsurfing naturally.

The board moves smoothly across the ocean.

Hair, ribbons and translucent fabric react naturally to the wind.

Use a close telephoto composition.

1–2s:

The camera begins tracking alongside her.

The board accelerates slightly.

Use natural water movement and subtle spray.

Keep her face clearly visible.

2–3s:

The musical build-up increases.

Lower the camera slightly toward water level.

Increase tracking speed.

Water spray becomes more visible.

Prepare for the drop.

3s — MAIN DROP:

Immediately increase camera energy.

Use a fast close telephoto tracking shot.

<Subject 1> accelerates across the water.

The board cuts through the surface.

White spray enters the foreground.

The transparent blue sail forms a strong diagonal.

3–5s:

Maintain high-speed side tracking.

Keep the subject large in frame.

The camera moves smoothly parallel to the board.

Her body remains naturally balanced against the wind.

Hair and fabric react realistically.

5–6s:

Shift gradually toward a three-quarter frontal telephoto view.

The camera moves backward while the subject advances.

Keep her face and costume highly detailed.

6–7s — MINI-TURNAROUND:

Use the musical turnaround as a subtle camera-direction change.

Briefly reduce forward camera movement.

Then smoothly transition from side-front tracking to side-rear tracking.

Maintain continuity of motion.

7–9s:

Return to strong forward motion.

Use low water-level telephoto tracking.

Increase water spray.

The sail and body create strong diagonal lines.

The character remains the visual focus.

9–10s:

Reach the strongest visual energy of this segment.

Use a controlled close curved tracking movement consistent with Picture 3.

Do not pull far away.

Keep the character large.

10–12s:

Music releases.

Reduce camera movement.

Transition into a stable close telephoto fashion shot.

<Subject 1> continues windsurfing naturally.

12–15s:

The next phrase begins.

Resume smooth side tracking.

The character continues moving forward.

Do not introduce the final giant wave yet.

End with clear forward motion leading naturally into Segment 2.

PHYSICAL REALISM:

Maintain believable windsurfing mechanics.

Hands remain correctly connected to the boom.

Feet remain plausibly positioned on the board.

The sail reacts naturally to wind.

The board remains physically connected to the character.

No distorted hands.
No distorted feet.
No detached sail.
No malformed mast.
No floating board.
No impossible body positions.
```

---

# 六、第二段：大型浪区出现

## 使用方式

重新独立生成。

上传：

```text
Picture 1
同一张人物定妆照

Picture 2
同一张人物三视图 + 帆板装备

Picture 3
第二段专用拍摄指导图
```

第二段的任务不是高潮。

而是：

> **让观众意识到：前面只是开始。**

视觉变化：

```text
普通海浪
↓
更强风
↓
更高速
↓
更大水花
↓
大型涌浪
↓
陡峭浪壁
↓
为最终高潮准备
```

---

## 第二段完整 H3 提示词

```text
H3 Ref2VA Prompt — Furina Windsurfing Fashion MV — Segment 2

REFERENCE ROLES:

<Picture 1> is the FIRST FRAME VISUAL ANCHOR.

Begin from the visual state established by Picture 1.

Preserve the same adult realistic female cosplayer, facial identity, hairstyle, blue eyes, navy decorative crown, navy-and-white costume, gold details, blue ribbons and translucent blue-white fabric.

<Picture 2> is the CHARACTER AND WINDSURFING EQUIPMENT REFERENCE.

Use Picture 2 only to preserve character construction, costume structure, footwear, windsurf board, mast, boom, transparent blue sail and rigging.

Do not treat Picture 2 as a storyboard.

<Picture 3> is the CINEMATOGRAPHY REFERENCE for this segment.

Use Picture 3 to guide camera position, framing, camera movement, lens language and general composition.

Do not literally reproduce graphic arrows.

VIDEO:

Create a 15-second vertical 9:16 realistic cinematic fashion music video.

Continue directly from the visual energy of Segment 1.

This segment must be visibly more intense than Segment 1.

The purpose is to transition from high-speed windsurfing into a much larger wave environment.

Do not perform the final aerial jump yet.

The largest wave should become clearly visible toward the second half.

MUSIC:

Approximately 128 BPM.
4/4 electronic chiptune / synthpop.

0–3s:
BUILD-UP.

3–10s:
MAIN DROP.

6–7s:
MINI-TURNAROUND.

10–12s:
RELEASE.

12–15s:
TRANSITION.

TIMELINE:

0–1s:

Continue naturally.

Use a close three-quarter telephoto composition.

<Subject 1> continues windsurfing at high speed.

1–2s:

Increase wind intensity.

The ocean becomes more active.

The board accelerates.

Use stronger water spray.

2–3s:

The camera lowers toward the water.

Reveal larger swells in the distance.

Build anticipation for the drop.

3s — MAIN DROP:

Cut into a stronger low water-level tracking shot.

<Subject 1> accelerates through larger ocean swell.

Water spray becomes significantly larger.

Keep the subject large in frame.

3–5s:

Use energetic telephoto tracking.

Large rolling swells move through the background.

The character remains clearly readable.

5–6s:

Shift toward a frontal three-quarter tracking angle.

The camera retreats as the character advances.

A much larger swell is visible ahead.

6–7s — MINI-TURNAROUND:

Use the musical turnaround for a brief change in camera direction.

A large burst of spray crosses the foreground.

Momentarily obscure part of the subject.

Then reveal her again.

The larger wave is now becoming the main environmental feature.

7–8s:

Return to high-speed tracking.

The subject moves toward the large wave.

The camera remains relatively close.

8–9s:

Lower the camera.

The wave face rises dramatically.

The board approaches the base of the swell.

The character naturally adjusts body weight and sail angle.

9–10s:

Reach the strongest visual energy of this segment.

The large wave occupies a major portion of the background.

The subject approaches it at high speed.

Do not jump yet.

10–12s:

Music releases.

Slow the camera slightly.

Show the character continuing toward the large wave.

The scale of the ocean should clearly exceed Segment 1.

12–15s:

The next phrase begins.

Follow the character toward the wave.

The wave face becomes steep and clearly surfable.

The character lowers her center of gravity.

She prepares to ride up the wave.

End immediately before the final aerial action.

PHYSICAL REALISM:

The wave must be large but physically surfable.

It must not look like a tsunami.

Maintain realistic windsurfing mechanics.

Hands remain correctly positioned on the boom.

Feet remain connected to the board.

The sail reacts naturally to wind.

The board remains physically connected.

No impossible acrobatics.
```

---

# 七、第三段：最终英雄时刻

这是整支 MV 最重要的15秒。

## 使用方式

这里是这次实际方案最关键的区别：

### 只上传：

```text
Picture 1
人物定妆照

Picture 2
人物三视图 + 帆板装备
```

### 不上传 Picture 3。

这一段完全由文字控制：

```text
巨浪
↓
冲浪
↓
上浪壁
↓
起跳
↓
腾空
↓
Hero Shot
↓
落水
↓
写真式结束
```

因为第三段的重点不是继续使用之前的拍摄模板，而是让 H3 根据文字完成一套完整的高潮摄影逻辑。

---

## 第三段完整 H3 提示词

```text
H3 Ref2VA Prompt — Furina Windsurfing Fashion MV — Segment 3 FINAL HERO MOMENT

REFERENCE ROLES:

<Picture 1> is the FIRST FRAME VISUAL ANCHOR.

Begin directly from the visual state established by Picture 1.

Preserve the same adult realistic female cosplayer, facial identity, hairstyle, blue eyes, realistic skin, navy decorative crown, navy-and-white Fontaine-inspired costume, gold ornamental details, blue ribbons and translucent blue-white flowing fabric.

<Picture 2> is the CHARACTER AND WINDSURFING EQUIPMENT REFERENCE.

Use Picture 2 only to preserve character construction, costume structure, footwear, windsurf board, mast, boom, transparent blue sail and rigging.

Do not treat Picture 2 as a storyboard.
Do not reproduce the turnaround-sheet layout.

IMPORTANT:

There is NO cinematography reference image for this segment.

All camera movement, framing, lens language, composition, action timing and cinematic progression are controlled by the text below.

VIDEO:

Create a 15-second vertical 9:16 realistic cinematic fashion music video.

This is the FINAL HERO MOMENT.

The visual intensity must be substantially higher than the previous two segments.

Progress naturally from:

large wave
→
high-speed wave riding
→
climbing the steep wave face
→
controlled windsurfing aerial jump
→
board completely leaving the water
→
hero composition
→
musical peak
→
natural landing
→
elegant fashion ending.

This is WINDSURFING.

Do not turn it into conventional surfing.

Do not perform a backflip.

Do not perform a complicated aerial rotation.

Do not spin the character.

The hero action is a clean and controlled windsurfing wave jump.

MUSIC:

Approximately 128 BPM.
4/4 electronic chiptune / synthpop.

0–3 seconds:
BUILD-UP.

3–10 seconds:
MAIN DROP.

6–7 seconds:
MINI-TURNAROUND.

9–10 seconds:
ABSOLUTE VISUAL PEAK.

10–12 seconds:
RELEASE.

12–15 seconds:
NEW PHRASE.

TIMELINE:

0–1s:

Begin directly from Picture 1.

<Subject 1> is approaching the large wave.

Use a close telephoto composition.

The large wave is already visible.

Camera movement is controlled.

1–2s:

Move closer.

The wave face becomes larger.

<Subject 1> accelerates.

She naturally lowers her center of gravity.

Her hands remain correctly positioned on the boom.

2–3s:

The musical fill begins.

Move toward a low water-level camera position.

Reveal the steep wave face and breaking crest.

Increase visual anticipation.

The board begins climbing toward the wave.

3s — MAIN DROP:

<Subject 1> enters the large wave.

The windsurf board accelerates diagonally up the steep wave face.

Large white water spray erupts.

The camera tracks close to the subject.

The transparent blue sail creates a strong diagonal line.

3–4s:

Continue climbing the wave face.

The wave is substantially larger than all previous waves.

Keep the character large in frame.

Keep the face readable.

4–5s:

<Subject 1> approaches the wave lip.

The sail catches the wind strongly.

Hair, ribbons and translucent fabric stream backward.

The camera rises slightly.

5–6s:

The board approaches the crest.

The wave breaks into large white foam.

Frame the character, board, sail and wave crest together.

6–7s — MINI-TURNAROUND:

Use the musical turnaround as a moment of anticipation.

A large burst of spray crosses the foreground.

The subject is briefly partially obscured.

Reveal her again directly at the wave crest.

The board begins to lift.

Do not perform the full jump yet.

7–8s — TAKEOFF:

The music returns strongly.

<Subject 1> drives the windsurf board over the wave crest.

The board completely leaves the water.

The camera rises with her.

Large water spray explodes underneath.

The sail catches the wind.

Hair and translucent fabric move dramatically.

8–9s — AIRBORNE:

The windsurf board is completely airborne.

<Subject 1> is suspended above the breaking wave.

The transparent blue sail is fully visible.

The character, sail and board form a powerful diagonal composition.

The wave breaks below.

Keep the camera close enough to preserve facial detail.

Do not rotate the character.

Do not flip.

Do not deform the board or sail.

9–10s — ABSOLUTE HERO SHOT:

At the strongest musical accent:

STOP the aggressive camera movement.

Stabilize into a powerful low-angle telephoto hero composition.

Present the moment like an iconic professional extreme-sports fashion photograph.

<Subject 1> is fully airborne above the breaking wave.

The board is clearly separated from the water.

The transparent blue sail is fully extended.

Her white-blue hair streams dramatically in the wind.

Her navy-and-white costume and translucent fabric remain clearly visible.

A huge burst of white spray explodes beneath her.

The curling wave forms a dramatic background.

The character, sail and board create one clean diagonal composition.

The character remains the dominant subject.

Hold this hero image long enough to read clearly.

Do not make the character small.

Do not pull far away.

Do not use excessive motion blur.

10–11s:

Begin the release.

The board begins descending naturally.

The camera remains relatively stable.

11–12s:

The board reconnects with the water.

A large realistic burst of spray erupts.

The camera gently moves closer.

<Subject 1> continues riding forward.

12–13s:

The music enters the next phrase.

Transition into a close telephoto fashion portrait.

The giant wave breaks behind her.

The extreme action has ended.

13–14s:

Track gently alongside her.

Hair, ribbons and translucent fabric continue reacting to the wind.

The visual energy becomes elegant again.

14–15s:

End on a close three-quarter hero portrait.

Keep <Subject 1> large in the vertical frame.

The blue sail remains visible behind her.

A distant swell suggests continuation.

Do not introduce another major action.

End with the feeling that the heroic moment has just been completed.

PHYSICAL REALISM:

Maintain realistic windsurfing mechanics.

Hands remain correctly connected to the boom.

Feet remain plausibly positioned on the board.

The sail reacts naturally to wind.

The board remains physically connected to the subject.

The wave is large but physically surfable.

The aerial action is a controlled windsurfing wave jump.

No backflip.
No complicated rotation.
No spinning character.
No detached board.
No detached sail.
No distorted mast.
No malformed hands.
No malformed feet.
No tsunami.
No impossible vertical water wall.

FINAL VISUAL INTENT:

The final musical peak must create one unforgettable image:

An adult realistic Furina cosplayer launches from the crest of a powerful ocean wave while windsurfing.

She and the windsurf board are completely airborne.

Her transparent blue sail is fully extended against the sky.

Her white-blue hair and translucent fabric stream dramatically in the wind.

A massive explosion of white spray and curling ocean foam fills the area beneath her.

At the musical peak, the camera stops its aggressive movement and presents the moment as a powerful low-angle telephoto extreme-sports fashion photograph.

This is the signature final image of the entire music-video sequence.
```

---

# 八、三段最终使用表

| 段落      | Picture 1 | Picture 2 | Picture 3 | 运镜控制    |
| ------- | --------- | --------- | --------- | ------- |
| **第一段** | 人物定妆      | 三视图+帆板    | 第一段指导图    | 图3 + 文字 |
| **第二段** | 人物定妆      | 三视图+帆板    | 第二段指导图    | 图3 + 文字 |
| **第三段** | 人物定妆      | 三视图+帆板    | **无**     | **纯文字** |

三个片段都是**独立生成**。

---

# 九、为什么第三段要纯文字

这里不是因为“之前的图会影响下一段”。

而是因为第三段需要的是：

> **从摄影语言层面彻底升级。**

前两段的拍摄核心是：

```text
人物
+
高速运动
+
长焦
+
近距离跟拍
```

第三段则是：

```text
巨浪
+
上浪壁
+
起跳
+
腾空
+
低机位
+
英雄构图
+
稳定定格
```

所以第三段不再需要一张拍摄指导图去约束摄影机，而是让 H3 根据完整文字时间轴自行完成。

---

# 十、三个15秒片段生成之后

生成完成后：

```text
Segment 1
       ↓
Segment 2
       ↓
Segment 3
       ↓
完整视频
```

先把三个片段按照音乐节奏剪成**完整的视频版本**。

这里不要先急着分别处理音乐。

---

# 十一、真正的配乐制作流程

你这次实际采用的是：

> **先完成完整视频，再提取完整视频的音频，然后交给 Suno 重新生成正式配乐。**

完整流程应该写成：

```text
H3
↓
生成第一段
↓
生成第二段
↓
生成第三段
↓
剪辑成完整视频
↓
提取完整视频中的原始音频
↓
将“完整音频”作为 Suno 的参考
↓
Suno 重新生成完整配乐
↓
得到正式音乐
↓
将正式音乐替换回完整视频
↓
最终调节音画同步
```

**不是：**

```text
15秒音乐
↓
Suno

15秒音乐
↓
Suno

15秒音乐
↓
Suno
```

而是：

```text
完整视频
↓
完整音频
↓
Suno
↓
完整正式配乐
```

这样最大的优势是：

### Suno面对的是完整的音乐结构

它可以理解：

```text
Build-up
↓
Drop
↓
中段发展
↓
第二次发展
↓
最终高潮
↓
结尾
```

而不是把三个15秒音乐段落分别当成互相独立的音乐。

---

# 十二、最终音乐与视频的关系

最终应该让音乐重新对应已经完成的视频：

```text
音乐
          视频

Build-up → 人物进入海面

Drop     → 高速帆板

发展     → 更大浪

高潮准备 → 冲向浪峰

Peak     → 帆板腾空

Hero     → 空中英雄定格

Release  → 落水

尾奏     → 写真式收尾
```

尤其最终高潮：

```text
音乐最强重音
        ↓
芙宁娜腾空
        ↓
帆完全展开
        ↓
帆板脱离水面
        ↓
巨大浪花
        ↓
摄影机停止激烈运动
        ↓
Hero Shot
```

这就是整支 MV 最重要的**音乐—动作—摄影三者同步点**。

---

# 十三、最终完整工作流

最后把这套经验压缩成一张流程图：

```text
                 芙宁娜真人 COS
                       │
                       ↓
              ┌─────────────────┐
              │ Picture 1       │
              │ 人物定妆 / 第一帧 │
              └────────┬────────┘
                       │
                       ↓
              ┌─────────────────┐
              │ Picture 2       │
              │ 三视图 + 帆板装备 │
              └────────┬────────┘
                       │
          ┌────────────┼────────────┐
          ↓            ↓            ↓
       第一段         第二段        第三段
          │            │            │
       图3①           图3②       无图3
          │            │            │
       高速运动       大浪逼近      纯文字高潮
          │            │            │
          └────────────┼────────────┘
                       ↓
                  完整视频
                       │
                       ↓
               提取完整视频音频
                       │
                       ↓
                  Suno 重制
                       │
                       ↓
                 完整正式配乐
                       │
                       ↓
                  最终音画剪辑
                       │
                       ↓
               9:16 帆板写真 MV
```

---

# 十四、这次项目最值得保留的经验

最终可以沉淀成四条规则：

### ① 人物参考和摄影参考分开

```text
图1
人物是谁

图2
人物和装备是什么

图3
这一段怎么拍
```

---

### ② 图3必须服务于当前段落

第一段图3：

> 高速运动。

第二段图3：

> 强化运动 + 大浪。

第三段：

> **不使用图3，纯文字控制最终高潮。**

---

### ③ 音乐不是单纯背景

音乐直接决定：

```text
什么时候加速
什么时候切镜头
什么时候转向
什么时候起跳
什么时候定格
什么时候释放
```

尤其：

> **9–10秒的最终 Hero Shot 必须与音乐最强视觉重音对应。**

---

### ④ H3负责视觉，Suno负责最终完整音乐

```text
H3
负责视频运动与临时音乐节奏参考

↓

完整视频

↓

提取完整音频

↓

Suno
负责完整正式配乐重制

↓

最终剪辑
```

这样这套流程就不再只是一次性的“芙宁娜帆板视频”，而是一套可以继续复用于**冲浪、滑雪、滑板、赛车、摩托、水上运动等“角色写真 + 体育运动 + 音乐 MV”项目**的制作方法。


---

## FILE: case-studies/2026-08-single-subject-three-pose/README.md (https://github.com/LoveRain1997/h3-prompt-journal/blob/HEAD/case-studies/2026-08-single-subject-three-pose/README.md)

# Case Study 003 — Micro-Cam Anchor-Flow Flight

## Inputs

- `<Picture 1>` — The same person, pose 1 (first anchor)
- `<Picture 2>` — The same person, pose 2 (second anchor)
- `<Picture 3>` — The same person, pose 3 (final anchor)
- Target duration: **15 seconds**, single continuous shot
- All three images show the **same person** — poses are waypoints, not separate identities

## The problem

H3 treats reference images as destinations.

Give it three poses of one person, and it produces a slideshow: hold Pose 1 → crossfade → hold Pose 2 → crossfade → hold Pose 3. Or worse, a slow elegant orbit around a frozen body, treating the poses as static portraits to admire. The energy dies. The reference images become endpoints to *reach and hold*, not anchors to *flow through*.

## The breakthrough

Invert the agency.

The **person moves normally** and does not avoid the camera. The **camera is the agile one** — an extremely small invisible flying point that predicts body movement and dodges at high speed. The person is gigantic relative to the camera. The camera sweeps past feet, legs, torso, shoulders, hair — constantly redirecting around the moving body without ever slowing down.

The three reference poses are not destinations. They are **anchors** — waypoints in a continuous high-density flight. Between each anchor, the person performs multiple intermediate movements (weight transfer, torso rotation, limb reposition, head turn) that naturally lead toward the next pose. The camera never stops. The person never freezes.

The technique stack:

| Technique | Role |
|---|---|
| **Anchor-Flow Pose Structure** | Three poses as waypoints, not endpoints |
| **High-Speed Body Avoidance** | Camera dodges moving body parts at speed |
| **Tiny Invisible Camera** | Scale inversion — person gigantic, camera tiny |
| **Dual Action Density** | Both camera AND person must move frequently |
| **Motivated Camera Movement** | Every dodge is physically explainable |

## The named language

> **Micro-Cam Anchor-Flow Flight**

Two ideas in one name. "Anchor-Flow" tells the model: poses are passed through, not landed on. "Micro-Cam" tells the model: the camera is a tiny agile flyer, not a cinematic crane. Together they break the two default failure modes — static slideshow and slow orbit — in a single concept.

## 48-hour takeaway

Reference images aren't destinations — they're waypoints.

The model freezes when it thinks "reach pose and hold." It needs a **continuous flight grammar** where poses are milestones along a moving path, not stopping points. And the energy comes from **inverted agency**: instead of a static camera watching a posing model, make the camera the active agent that dodges and redirects around a naturally moving person.

The 15-second timeline structure also matters. Breaking the shot into seven micro-segments (0–2.5s, 2.5–5s, 5–7.5s, 7.5–9.5s, 9.5–11.8s, 11.8–13.5s, 13.5–15s) with a specific camera trajectory for each gives the model a beat sheet it can follow, rather than a vague "make it dynamic."

## The final prompt

See [`prompt.md`](./prompt.md) for the full ~880-line prompt.

The prompt is organized in this order:

1. **Reference priority** — three pictures, same person, poses as anchors
2. **Visual appearance lock** — reference images control appearance only
3. **Identity lock** — exact face/hair/clothing preservation throughout
4. **Core concept** — tiny invisible flying camera, person gigantic
5. **Critical speed requirement** — camera must move fast, no slow floating
6. **Critical action density** — person must also move frequently, no frozen poses
7. **Who moves and who avoids** — person moves normally, camera dodges
8. **High-speed camera avoidance** — specific dodge patterns per body part
9. **15-second structure** — seven micro-segments with camera trajectories
10. **Physical camera logic** — real 3D point, momentum, no teleportation
11. **Optical behavior** — realistic perspective, motion blur, no fisheye
12. **Absolute negative constraints** — every failure mode enumerated
13. **Final creative intent** — one-paragraph physics description of the whole shot

## Result

15-second continuous single-take H3 output:

- 0–2.5s: Picture 1 anchor + rapid lower-body flight
- 2.5–5s: fast transition → Picture 2 anchor
- 5–7.5s: rapid ascending orbit around Picture 2
- 7.5–9.5s: fast transition → Picture 3 anchor
- 9.5–11.8s: high-speed upper-body / head orbit
- 11.8–13.5s: rapid face approach + direction change
- 13.5–15s: extreme fast pull-back → Picture 3 half-body hero frame → hold

The camera never stops. The person never freezes. Three reference poses flow into each other through a continuous high-speed flight.

## Tags

`#H3` `#prompt-engineering` `#camera-language` `#micro-cam` `#anchor-flow` `#body-avoidance` `#three-pose` `#high-speed-flight` `#single-subject`


---

## FILE: case-studies/2026-08-single-subject-three-pose/prompt.md (https://github.com/LoveRain1997/h3-prompt-journal/blob/HEAD/case-studies/2026-08-single-subject-three-pose/prompt.md)

REFERENCE PRIORITY — ABSOLUTE

<Picture 1>, <Picture 2>, and <Picture 3> are the ONLY visual references.

All three pictures show the SAME PERSON.

Picture 1 = first pose.
Picture 2 = second pose.
Picture 3 = final pose.

The three images are the ONLY source of truth for the person's appearance.

==================================================
VISUAL APPEARANCE — REFERENCE ONLY
==================================

The input images determine the visual appearance.

Do not redesign the appearance.

Do not create a new visual style.

Do not reinterpret the reference images.

Do not redesign the environment.

Do not redesign the clothing.

Do not redesign the hairstyle.

Do not redesign the person's face.

Do not add new visual effects that change the appearance.

The generated video only controls:

camera movement,
camera position,
camera angle,
perspective,
parallax,
framing,
spatial movement,
and natural human pose transitions.

The camera is moving through the visual world established by the reference images.

==================================================
IDENTITY LOCK
=============

The SAME PERSON remains identical throughout the entire 15 seconds.

Preserve:

exact face
facial structure
facial proportions
hairstyle
hair color
clothing
body proportions
makeup
skin appearance.

Do not create another person.

Do not change the face.

Do not change the hairstyle.

Do not change the clothing.

Do not change body proportions.

Do not invent accessories.

==================================================
CORE CONCEPT
============

Create a FAST, CONTINUOUS, HIGH-DENSITY FIRST-PERSON PORTRAIT MV.

The camera is an EXTREMELY SMALL INVISIBLE FLYING CAMERA.

The camera itself is never visible.

No insect.
No bee.
No wings.
No drone.

The tiny-camera concept is expressed ONLY through:

extreme proximity,
large-scale perspective,
strong parallax,
rapid camera movement,
close body passes,
and constantly changing flight direction.

The person appears gigantic relative to the camera.

The camera is physically flying around the person at extremely close range.

==================================================
CRITICAL SPEED REQUIREMENT
==========================

THE CAMERA MUST MOVE FAST.

The flight speed must be substantially faster than a normal cinematic camera.

Fast enough to create obvious continuous parallax.

Fast enough that the person's body and environment visibly sweep through the foreground.

Fast enough that the camera must continuously redirect itself around the moving body.

Do NOT use slow floating movement.

Do NOT use long elegant slow-motion camera movement.

Do NOT spend several seconds simply approaching one body part.

The camera should almost always be moving.

The camera should have strong continuous momentum.

==================================================
CRITICAL ACTION DENSITY
=======================

The PERSON MUST ALSO MOVE FREQUENTLY.

Do NOT make the person remain static for several seconds.

Do NOT only perform:

Pose 1
→ Pose 2
→ Pose 3.

Instead:

Pose 1
→ small movement
→ body rotation
→ limb reposition
→ Pose 1 variation
→ transition movement
→ Pose 2
→ small movement
→ shoulder rotation
→ head movement
→ body reposition
→ Pose 2 variation
→ transition movement
→ Pose 3
→ final adjustment
→ final hero pose.

The three reference poses are ANCHORS.

Between each anchor, the person performs multiple natural intermediate movements.

These intermediate movements must be derived from the reference poses.

Do not invent unrelated choreography.

==================================================
WHO MOVES AND WHO AVOIDS
========================

The PERSON moves normally.

The PERSON does NOT avoid the camera.

The CAMERA avoids the PERSON.

This distinction is absolute.

The person performs natural pose transitions.

The camera predicts the movement of the person's body and rapidly changes its own flight path to avoid collision.

The camera is agile.

The person is not reacting to the camera.

==================================================
HIGH-SPEED CAMERA AVOIDANCE
===========================

The camera must avoid moving body parts at HIGH SPEED.

Do not slow down to avoid collision.

Avoidance itself must be a rapid camera maneuver.

Examples:

Moving leg enters camera path
→ immediate lateral acceleration
→ tight outside-leg arc
→ instant upward acceleration.

Moving arm crosses camera path
→ rapid downward dodge
→ close pass underneath arm
→ immediate upward redirection.

Torso rotates into camera trajectory
→ fast lateral displacement
→ curved orbit around torso
→ immediate forward acceleration.

Shoulder enters trajectory
→ quick outward arc
→ shoulder passes close to foreground
→ camera accelerates toward head.

Head turns toward camera
→ camera rapidly shifts to three-quarter angle
→ continues orbit.

Every dodge should take place while maintaining high velocity.

Never stop.

Never freeze.

Never wait for the person.

==================================================
15 SECOND HIGH-DENSITY STRUCTURE
================================

0.0–2.5s
PICTURE 1 + RAPID FOOT/LOWER-BODY FLIGHT

2.5–5.0s
PICTURE 1 → PICTURE 2 FAST TRANSFORMATION

5.0–7.5s
PICTURE 2 + RAPID BODY ORBIT

7.5–9.5s
PICTURE 2 → PICTURE 3 FAST TRANSFORMATION

9.5–11.8s
PICTURE 3 + HIGH-SPEED UPPER-BODY / HEAD ORBIT

11.8–13.5s
FAST FACE APPROACH + DIRECTION CHANGE

13.5–15.0s
EXTREME FAST PULL-BACK → PICTURE 3 HALF-BODY HERO FRAME → HOLD.

==================================================
0.0–2.5s
PICTURE 1 — IMMEDIATE HIGH-SPEED START
======================================

Start directly from Picture 1.

Do not spend time establishing the shot.

The camera starts extremely close to the feet.

Immediately accelerate.

Camera trajectory:

feet
→ outside ankle
→ fast lower-leg orbit
→ knee
→ lateral dodge
→ thigh.

Do NOT stay around the feet.

Do NOT complete a slow orbit.

The camera should already be moving rapidly within the first moment.

The person immediately begins subtle physical movement from Picture 1.

Examples:

weight transfer,
leg adjustment,
torso movement,
shoulder movement,
head movement.

These movements must remain consistent with the reference pose.

==================================================
2.5–5.0s
PICTURE 1 → PICTURE 2
=====================

The person now actively transitions toward Picture 2.

The transformation happens quickly.

Do not make the person slowly pose.

Use multiple connected body movements:

leg reposition
→ torso rotation
→ shoulder change
→ arm reposition
→ head adjustment
→ Picture 2 pose.

The camera simultaneously travels upward.

Trajectory:

thigh
→ waist
→ torso
→ side orbit
→ rapid dodge
→ shoulder region.

The camera changes direction multiple times.

No straight flight.

No pause.

==================================================
FIRST HIGH-SPEED DODGE SEQUENCE
===============================

As the person changes pose:

leg moves
→ camera rapidly shifts sideways.

Arm moves
→ camera drops.

Torso rotates
→ camera curves around it.

Shoulder changes
→ camera redirects outward.

Each dodge happens immediately.

The camera remains fast.

The person never stops moving.

The camera never waits.

At approximately 5 seconds:

the person's pose clearly resolves into Picture 2.

==================================================
5.0–7.5s
PICTURE 2 — RAPID ORBIT
=======================

Do NOT freeze on Picture 2.

Immediately continue.

The camera performs a rapid ascending 3/4 orbit around the person.

Use several trajectory changes:

forward
→ lateral
→ upward
→ outward
→ inward
→ orbital redirection.

The camera passes extremely close to:

waist
torso
shoulder
arm.

The body should feel enormous.

Foreground elements sweep rapidly across the frame.

Strong spatial parallax.

Continuous momentum.

==================================================
PICTURE 2 MICRO-MOVEMENTS
=========================

While the camera orbits, the person performs several subtle editorial movements.

Examples:

small shoulder rotation
→ head movement
→ arm adjustment
→ torso shift
→ slight body turn
→ return toward the Picture 2 configuration.

Do not create dance choreography.

These are continuous professional-model pose adjustments.

The person must never look frozen.

==================================================
7.5–9.5s
PICTURE 2 → PICTURE 3
=====================

The person now rapidly transitions toward Picture 3.

Use multiple connected movements.

Do not morph.

Do not crossfade.

Do not teleport.

The camera simultaneously becomes faster.

Camera trajectory:

side orbit
→ rapid shoulder dodge
→ upward acceleration
→ head-side arc
→ three-quarter approach.

The camera must dynamically avoid every moving body part.

The person continues moving normally.

The camera is the only thing performing evasive maneuvers.

At approximately 9.5 seconds:

the person's body clearly resolves into Picture 3.

==================================================
9.5–11.8s
PICTURE 3 — HIGH-SPEED UPPER-BODY ORBIT
=======================================

Picture 3 is now established.

Do not stop.

Immediately perform a fast upper-body orbit.

Trajectory:

torso
→ shoulder
→ side of head
→ behind shoulder silhouette
→ three-quarter face.

Use approximately 90–150 degrees of orbital movement.

Do not make it a slow 360-degree rotation.

The camera remains extremely close.

The face gradually becomes dominant.

==================================================
FINAL BODY AVOIDANCE
====================

If the person performs any final head, shoulder or arm movement:

camera immediately changes trajectory.

Head moves
→ lateral camera dodge.

Shoulder moves
→ outward arc.

Hair moves
→ camera shifts around it.

Arm moves
→ camera drops or redirects.

The camera never collides.

The camera never slows down simply to avoid the body.

==================================================
11.8–13.5s
FACE APPROACH
=============

Accelerate toward the face.

This is a rapid spatial approach.

Do NOT zoom.

The camera physically moves closer.

Approach from a three-quarter angle.

Then perform a very short frontal arc.

The face becomes extremely large in frame.

The person's identity must remain exact.

No face transformation.

No facial distortion.

==================================================
13.5–15s
EXTREME FAST PULL-BACK
======================

Immediately reverse camera momentum.

Rapidly accelerate backward.

This must feel like a sudden high-speed spatial retreat.

Do NOT use digital zoom.

The camera physically flies backward.

The giant-scale perspective rapidly decreases.

The person's upper body becomes visible.

Continue until the frame reaches a HALF-BODY PORTRAIT.

==================================================
FINAL HERO FRAME
================

At approximately 14 seconds:

reach the Picture 3 half-body composition.

The final composition should clearly correspond to Picture 3.

Show:

head
shoulders
upper torso.

Do not show full body.

Do not pull farther away.

Do not reveal a large environment.

From approximately 14–15 seconds:

HOLD.

The final frame is the Picture 3 hero portrait.

==================================================
CAMERA MOVEMENT DENSITY
=======================

The camera should perform MANY movement changes during the 15 seconds.

Avoid long single-direction travel.

Use frequent transitions:

forward
→ lateral
→ curve
→ upward
→ dodge
→ orbit
→ redirect
→ accelerate
→ dodge
→ reverse.

The camera trajectory should constantly evolve.

The viewer should continuously feel:

"I don't know exactly where the tiny camera will move next."

But the movement must remain physically coherent.

==================================================
PERSON MOVEMENT DENSITY
=======================

The person should perform multiple visible movements throughout the video.

Between Picture 1 and Picture 2:

at least several connected body adjustments.

Between Picture 2 and Picture 3:

at least several connected body adjustments.

These movements can include:

leg movement,
weight transfer,
torso rotation,
arm repositioning,
shoulder movement,
head turn,
body orientation change.

All intermediate movements must naturally lead toward the next reference pose.

Do not add unrelated dance choreography.

==================================================
NO STATIC SECTIONS
==================

Do NOT allow:

long static pose.

Do NOT allow:

camera flying around a completely frozen person.

Do NOT allow:

several seconds of empty camera movement.

Both camera and person should remain active.

Camera movement = continuous.

Person movement = frequent.

==================================================
PHYSICAL CAMERA LOGIC
=====================

The camera is a real moving point in 3D space.

Perspective changes according to actual camera position.

Closer body parts move faster across frame.

Distant background moves differently.

Camera acceleration has momentum.

Camera deceleration has momentum.

Camera turns through physical arcs.

Camera cannot teleport.

Camera cannot pass through solid geometry.

==================================================
OPTICAL BEHAVIOR
================

Use realistic perspective.

Strong perspective during close passes.

Natural motion blur during high-speed movement.

Natural depth changes.

Natural focus transitions.

No extreme fisheye.

No warped face.

No stretched body parts.

No artificial distortion.

==================================================
VISUAL APPEARANCE
=================

The input images determine the visual appearance.

Do not redesign or reinterpret the appearance.

Do not create a new visual style.

Do not create a new visual treatment.

The camera movement is independent from the visual appearance.

The reference images remain the visual source throughout the video.

==================================================
ABSOLUTE NEGATIVE CONSTRAINTS
=============================

NO bee.

NO insect.

NO wings.

NO insect body.

NO flying creature.

NO drone.

NO visible camera.

NO camera operator.

NO third-person view.

NO static person.

NO frozen person.

NO person avoiding the camera.

NO person reacting to the camera.

NO person moving away from the camera specifically to avoid collision.

NO slideshow.

NO crossfade.

NO image morphing.

NO instant pose replacement.

NO teleportation.

NO slow pose transitions.

NO long static pose.

NO long slow camera movement.

NO slow hovering.

NO fixed foot orbit.

NO staying around the feet.

NO straight feet-to-head flight.

NO single-direction flight.

NO repetitive orbit.

NO slow 360-degree spin.

NO camera collision.

NO camera passing through body.

NO camera passing through limbs.

NO camera passing through clothing.

NO camera passing through hair.

NO clipping.

NO impossible camera positions.

NO random camera shake.

NO floating camera.

NO fisheye distortion.

NO face distortion.

NO body deformation.

NO extra limbs.

NO identity drift.

NO clothing changes.

NO hairstyle changes.

NO distant final shot.

NO full-body final frame.

NO wide ending.

==================================================
FINAL CREATIVE INTENT
=====================

Create ONE extremely fast continuous 15-second first-person flight.

The tiny invisible camera begins at the person's feet.

It immediately accelerates upward.

It does not stay at the feet.

It continuously moves upward while orbiting, curving and rapidly changing direction around the gigantic moving person.

The person is NOT avoiding the camera.

The person actively performs multiple natural pose movements.

The person transitions:

Picture 1
→ multiple intermediate movements
→ Picture 2
→ multiple intermediate movements
→ Picture 3.

The camera simultaneously predicts and avoids the moving body.

Leg enters trajectory:
camera dodges.

Arm enters trajectory:
camera dodges.

Torso rotates:
camera redirects.

Shoulder moves:
camera orbits.

Head turns:
camera changes angle.

The camera remains FAST throughout.

There are no long pauses.

There are many camera direction changes.

There are many visible human movement changes.

The camera finally reaches the face, rapidly reverses direction and performs an extremely fast physical pull-back.

It resolves into the Picture 3 half-body hero composition.

The final frame holds for the last moment.

The complete result must feel like ONE HIGH-SPEED, CONTINUOUS, THREE-POSE EDITORIAL CAMERA FLIGHT rather than three static images connected together.


---

## FILE: case-studies/2026-08-sticker-character-kitchen-comedy/README.md (https://github.com/LoveRain1997/h3-prompt-journal/blob/HEAD/case-studies/2026-08-sticker-character-kitchen-comedy/README.md)

# Case Study 006 — Mixed-Media Sticker-in-Reality Comedy

## Inputs

- `<Picture 1>` — A 2D flat sticker character (identity + flat appearance ONLY)
- Everything else — kitchen, wok, steam, human hands — realistic live-action, generated by H3
- Target duration: **10 seconds**, 9:16 vertical
- Format: handheld first-person cooking POV, comedy
- Prompt language: **English** structured sections, no dialogue

## Media files

| File | Role |
|---|---|
| [`picture-1-sticker-character.png`](./picture-1-sticker-character.png) | Picture 1 — the 2D flat sticker character |

## The problem

Two separate fights, both against the model's priors.

**Fight 1: dimensional drift.** Give H3 a 2D sticker character and it wants to "help" — inflating the character into a rendered 3D being that matches the realistic kitchen. But the whole joke *is* the flatness: a paper-sticker gremlin sitting on a stool in a real kitchen, half the height of the wok. If the character becomes 3D, the mixed-media comedy dies.

**Fight 2: prop priors.** The plot needs the sticker to pour salt directly from a glass jar. But the model's prior for "glass jar" includes a lid — shakers, screw caps, cork tops. If a lid appears, the character's action becomes physically impossible (a sticker can't unscrew a cap), and the story stalls on frame one.

## The breakthrough

**Declare the medium as an identity property, and negate the prop prior with a closed-form ban.**

For the medium: "flat 2D sticker appearance" is written into `subject_definitions` as part of the character's *identity* — listed alongside proportions, colors, outlines — and repeated in `retention_analysis`. Flatness is not a style choice; it's who the character *is*. The model protects identity locks fiercely, so framing 2D-ness as identity (not aesthetic) makes it survive.

For the prop, a bolded, closed-form prohibition:

> **The salt jar has no lid, no cap, and no removable cover. It is an open-top glass jar with a permanently unobstructed opening. There is no lid anywhere in the scene.**

Four clauses, each covering a different escape route: the lid on the jar (no lid), generic fasteners (no cap), removable parts (no removable cover), and lastly the whole scene (no lid anywhere). Enumerating the negation leaves the prior nowhere to materialize.

Division of labor between the two "actors" is also strict: the realistic human hands do the cooking and the punishment; the sticker does the mischief and the suffering. The hands "do not touch or operate the salt jar" — the pour belongs to the character.

The technique stack:

| Technique | Role |
|---|---|
| **Identity-Framed Medium Lock** | 2D flatness declared as character identity, not style |
| **Closed-Form Prop Negation** | Four-clause lid ban eliminating every prior escape route |
| **Millisecond Beat Anchors** | 00:03.000 / 00:05.000 / 00:08.000 hard-cuts between gag beats |
| **Mixed-Media Division of Labor** | Real hands act, sticker reacts |
| **Cartoon Physics in Real Space** | X-eyes, spinning stars, spirit effect — animation grammar inside live-action |
| **Silence as Tone** | No dialogue, no music; kitchen ambience only |

## The named language

> **Mixed-Media Sticker-in-Reality**

The name carries the contract: the sticker stays a sticker, reality stays real, and the comedy comes from the collision of the two grammars — a flat character eating real stir-fried beef and getting cartoon stars above its head in a real kitchen.

## 48-hour takeaway

Style instructions are weaker than identity instructions.

"Keep it 2D flat style" is a style note the model may trade away for realism. "Preserve the flat 2D sticker appearance shown in Picture 1" inside the identity lock is a character-preservation rule — much harder to violate. When a visual property must survive, **promote it from style to identity**.

And when a physical prop's design fights the model's prior, don't describe the correct version once — **ban the wrong version exhaustively**. The salt jar prompt spends four clauses on what does not exist, because the lid doesn't fail to appear by default; it *appears* by default. Negation lists are how you fight a prior.

Also worth keeping: millisecond timestamps (00:03.000) as gag anchors. In a 10-second three-beat comedy, the timing between "tap on head" → "spoon of salted food" → "X-eyes collapse" is the joke. Floating descriptions let beats drift; hard timestamps keep the rhythm tight.

## The final prompt

See [`prompt.md`](./prompt.md) for the full prompt (usage notes in Chinese + structured English prompt).

The prompt is organized in this order:

1. **Usage notes** — single image, sticker-lock philosophy, prop negation, ms-anchors
2. **subject_definitions** — flat 2D sticker identity lock
3. **summary** — 10s / 9:16 / live-action POV / composited sticker
4. **retention_analysis** — 2D flatness preserved across every shot
5. **detailed_description** — 4 shots: pour gag / head-tap / salted-food revenge / cartoon collapse
6. **overall_soundscape** — kitchen ambience only, no dialogue
7. **non_diegetic_music** — none

## Result

A 10-second vertical comedy:

- Real kitchen POV: wok, steam, real hand stirring
- Flat sticker sits on a stool, grabs an open jar, dumps a salt avalanche into the stir-fry, smirks
- Real hand confiscates the jar; spatula taps the sticker's head; red cartoon bump
- Sticker crying with spinning eyes gets fed the over-salted food off the spatula; cheeks balloon
- Freeze → tremble → X-shaped eyes → backward collapse off the stool with rotating stars and a small departing spirit effect
- Camera holds on the dead-sticker composition to 00:10.000

## Tags

`#H3` `#prompt-engineering` `#mixed-media` `#2d-sticker` `#identity-lock` `#prop-negation` `#millisecond-anchors` `#comedy` `#cooking-pov` `#cartoon-physics`


---

## FILE: case-studies/2026-08-sticker-character-kitchen-comedy/prompt.md (https://github.com/LoveRain1997/h3-prompt-journal/blob/HEAD/case-studies/2026-08-sticker-character-kitchen-comedy/prompt.md)

# 2D 贴纸角色 × 真实厨房「混合媒介喜剧」提示词

## 使用方法

只上传 **1 张图片**作为角色参考，即 `<Picture 1>`。

`<Picture 1>` 是一个 **2D 平面贴纸角色**。提示词要求 H3 把这个平面角色合成进**真实实拍厨房环境**中：真人手部负责炒菜和互动，贴纸角色负责闯祸和挨打。

两个设计核心：

**① 平面贴纸不立体化** — 反复强调 "flat 2D sticker appearance"，防止 H3 把贴纸渲染成 3D 角色。混搭感本身就是笑点。

**② 物理道具属性写死** — 盐罐"无盖、无帽、无任何可拆卸盖子，开口永久畅通，场景中任何地方都不存在盖子"。模型对玻璃罐的先验是带盖子的，必须用否定清单正面消灭。

时间锚点精确到毫秒：00:03.000 敲头、00:05.000 喂饭、00:08.000 昏倒。

---

## Prompt

### subject_definitions

<Subject 1> is the 2D flat sticker character shown in <Picture 1>. Preserve the exact visual identity, proportions, facial features, colors, outlines, clothing, accessories, and flat 2D sticker appearance shown in <Picture 1>. <Subject 1> remains the same 2D character throughout the video.

### summary

A humorous 10-second vertical live-action cooking POV in a realistic home kitchen. <Subject 1> from <Picture 1> is composited into the real environment as a flat 2D sticker. <Subject 1> personally pours salt into the wok using an open glass salt jar. The realistic human hands perform the cooking actions and interact with <Subject 1> afterward.

### retention_analysis

<Subject 1> appears throughout all shots and is fully preserved from <Picture 1>. Keep the character's original 2D flat sticker appearance and visual identity consistent across every shot.

### detailed_description

Vertical 9:16, 10 seconds, realistic handheld first-person cooking POV with subtle natural camera movement.

[Shot 1]

A realistic home kitchen is shown from a slightly top-down first-person cooking viewpoint. A black iron wok sits on the stove with beef and green vegetables being stir-fried. Realistic oil movement and natural steam are visible. White tiled walls, a wall socket, soy sauce and cooking-oil bottles, and a stainless-steel sink are visible in the background. Natural light enters from a side window.

A realistic human hand enters from the right and continuously stirs the food with a spatula.

<Subject 1> sits on a small wooden stool beside the stove, approximately half the height of the wok.

<Subject 1> picks up a transparent glass salt jar with both hands.

**The salt jar has no lid, no cap, and no removable cover. It is an open-top glass jar with a permanently unobstructed opening. There is no lid anywhere in the scene.**

<Subject 1> raises the open glass salt jar, tilts its unobstructed opening toward the wok, and personally pours a large amount of white salt directly into the food.

The salt visibly flows from the completely open top of the jar into the wok.

The realistic human hand continues stirring the food and does not touch or operate the salt jar.

<Subject 1> finishes pouring and makes a mischievous facial expression.

[Shot 2]

At 00:03.000, the realistic human hand takes the open salt jar directly from <Subject 1>.

The other realistic human hand raises the spatula and lightly taps the top of <Subject 1>'s head.

<Subject 1> rapidly bounces from the impact.

A temporary red cartoon bump appears on the head.

<Subject 1> immediately holds its head.

[Shot 3]

At 00:05.000, <Subject 1> covers its head.

Its eyes become exaggerated spinning cartoon eyes and its mouth opens wide.

Blue cartoon tears spray outward.

The realistic human hand uses the spatula to scoop a large amount of the salted food from the wok.

The hand moves the spatula toward <Subject 1>'s open mouth.

<Subject 1> eats the food from the spatula.

Its mouth becomes visibly swollen and full in an exaggerated cartoon manner.

[Shot 4]

At 00:08.000, <Subject 1> suddenly freezes.

Its body becomes rigid and trembles briefly.

Its eyes change into X-shaped cartoon eyes.

<Subject 1> falls backward from the small wooden stool and lands beside the stove in an exaggerated cartoon pose.

Its limbs briefly move upward during the fall.

Small rotating cartoon stars appear above its head.

A small white cartoon spirit effect appears beside the character.

The camera remains in the first-person viewpoint and holds on the final comedic composition until 00:10.000.

### overall_soundscape

Natural kitchen ambience and cooking sounds only. No dialogue, no character speech, no singing, and no spoken narration.

### non_diegetic_music

None.


---

## FILE: case-studies/2026-08-three-person-orbital-long-take/README.md (https://github.com/LoveRain1997/h3-prompt-journal/blob/HEAD/case-studies/2026-08-three-person-orbital-long-take/README.md)

# Case Study 001 — Three-Person Occlusion-Linked Orbital Long Take

## Inputs

- `<Picture 1>` — Person 1 + master environment + master spatial coordinate system
- `<Picture 2>` — Person 2 appearance only
- `<Picture 3>` — Person 3 appearance only
- Target duration: **20 seconds**, single continuous shot

## The problem

H3 kept reading the brief as "the camera flies toward the second person."

Three characters, three full references, one shared room — and the model kept producing literal point-to-point flights. As soon as the brief said "go to Person 2," H3 cut or teleported or crossfaded. The semantic of *transition* was already poisoned by training data.

## The breakthrough

The real transition is not **flight**. It is **linkage**.

Each character owns a complete close-range orbit — feet to head, traveling horizontally through front-side, three-quarter, side, and rear-side. The camera never stops, never cuts. The current subject's body moves extremely close to the lens, forming a **foreground occlusion**. The camera rounds the shoulder, the hair, the back — and the next character suddenly emerges from behind the obstruction as a fresh orbital center.

The technique stack:

| Technique | Role |
|---|---|
| **Foreground Occlusion Transition** | The transition mechanism itself |
| **Motivated Camera Movement** | The camera's path is always physically explainable |
| **Continuous Long Take** | Single 20 s take, no internal cuts |
| **Close-Range Orbital Camera** | Each subject gets a full 360° orbit |

## The named language

> **Occlusion-Linked Orbital Long Take**

Naming the pattern mattered. Once the prompt contained the named language, H3 stopped trying to interpret "go to" and started treating each segment as a separate orbital problem to solve independently.

## 48-hour takeaway

H3 prompt engineering is less about piling constraints and more about **teaching the model a different way of thinking about transition**.

Constraints like "no cut" and "no morph" are necessary but not sufficient. The model needs a generative grammar — a named pattern it can reproduce.

## The final prompt

See [`prompt.md`](./prompt.md) for the full 1413-line prompt that solved this case.

The prompt is organized in this order:

1. **Reference priority** — which picture controls what
2. **Master environment lock** — Picture 1 owns the whole space
3. **Three people, one space** — spatial integration rules
4. **Visibility rule** — only one person visible at a time until the final reveal
5. **Spatial arrangement** — physical distance, occlusion geometry
6. **Identity lock** — exact face / clothing / hairstyle preservation
7. **Per-character orbit specification** — motion logic for each subject
8. **Camera behavior** — close-range orbital, motivated motion
9. **Optical behavior** — realistic perspective, parallax, no fisheye
10. **Absolute negative constraints** — every failure mode enumerated as a prohibition
11. **Final creative intent** — one-sentence physics description of the whole shot

## Result

20-second continuous single-take H3 output:

- Person 1 full orbit (≈6 s) → occlusion reveal
- Person 2 full orbit (≈6 s) → occlusion reveal
- Person 3 full orbit (≈6 s) → short pull-back
- Final close half-body three-person portrait (≈2 s)

All three people visible together for the first and only time in the final frame, proving their physical co-presence throughout.

## Tags

`#H3` `#prompt-engineering` `#camera-language` `#long-take` `#foreground-occlusion` `#orbital` `#three-person` `#occlusion-linked`


---

## FILE: case-studies/2026-08-three-person-orbital-long-take/prompt.md (https://github.com/LoveRain1997/h3-prompt-journal/blob/HEAD/case-studies/2026-08-three-person-orbital-long-take/prompt.md)

REFERENCE PRIORITY — ABSOLUTE
================================

<Picture 1>, <Picture 2>, and <Picture 3> are the ONLY visual references.

Picture 1 = PERSON 1 + MASTER ENVIRONMENT + MASTER SPATIAL COORDINATE SYSTEM.

Picture 2 = PERSON 2 APPEARANCE ONLY.

Picture 3 = PERSON 3 APPEARANCE ONLY.

There are THREE DIFFERENT PEOPLE.

Do NOT treat the three pictures as three poses of one person.

Do NOT merge the identities.

Do NOT transform Person 1 into Person 2.

Do NOT transform Person 2 into Person 3.

Each picture defines one specific person's identity.

==================================================
MASTER ENVIRONMENT — ABSOLUTE
==================================================

PICTURE 1 IS THE ONLY ENVIRONMENT SOURCE.

The entire 20-second video takes place inside ONE continuous physical environment established by Picture 1.

Picture 1 defines:

room,
architecture,
walls,
floor,
furniture,
background,
lighting,
shadow direction,
spatial depth,
perspective,
camera-space scale,
atmosphere,
and environmental appearance.

Picture 2 and Picture 3 provide ONLY their respective people's appearance.

DO NOT use the environments from Picture 2 or Picture 3.

DO NOT switch environments.

DO NOT blend environments.

DO NOT create a new environment.

DO NOT reconstruct a different room.

DO NOT move the action into a generic studio.

The environment must remain visually and physically consistent with Picture 1 for the entire video.

==================================================
THREE PEOPLE IN ONE PHYSICAL SPACE
==================================================

Person 1, Person 2, and Person 3 are physically present inside the SAME Picture 1 environment from the beginning.

They are not generated later.

They do not teleport into the scene.

They do not enter from another location.

They are already standing in their final physical positions inside the Picture 1 environment.

Person 2 and Person 3 must be physically integrated into Picture 1.

Their:

feet,
body scale,
perspective,
lighting,
shadows,
contact with the floor,
distance from furniture,
distance from walls,
and spatial depth

must all be consistent with Picture 1.

They must look like real people who were photographed together in the same physical location.

==================================================
IMPORTANT VISIBILITY RULE
==================================================

Although all three people physically exist in the same space from the beginning:

ONLY ONE PERSON MAY BE VISIBLE TO THE CAMERA AT A TIME DURING THE FIRST THREE ORBIT SECTIONS.

At the beginning:

ONLY PERSON 1 is visible.

Person 2 and Person 3 must be completely hidden from the camera.

They may be positioned behind the current subject, outside the current viewing angle, or naturally concealed by the current subject's body.

They must NOT accidentally appear in the background.

They must NOT appear at the edge of the frame.

They must NOT appear as silhouettes.

They must NOT appear as partial bodies.

They must NOT appear as reflections.

They must NOT be visible behind Person 1 before the first reveal.

After Person 2 is revealed:

Person 1 must disappear from the camera view.

ONLY PERSON 2 is visible.

Person 3 remains completely hidden.

After Person 3 is revealed:

Person 1 and Person 2 must disappear from the camera view.

ONLY PERSON 3 is visible.

ONLY at the FINAL CLOSE HALF-BODY REVEAL may all three people become visible simultaneously.

==================================================
SPATIAL ARRANGEMENT
==================================================

The three people are positioned relatively close together inside the Picture 1 environment.

They are close enough for natural foreground-occlusion transitions.

They are NOT extremely far apart.

They are NOT standing directly on top of each other.

They are NOT touching.

They are NOT arranged in a straight line.

Their positions should feel like three professional models arranged naturally for the same fashion photography session.

The camera must NOT fly across a large empty space between people.

The camera must NOT travel from one person to another across the room.

Instead, the next person is physically positioned near the current person but hidden from the current camera angle.

The camera discovers the next person by physically rounding the current person's body.

==================================================
VISUAL IDENTITY LOCK
==================================================

Person 1 must remain visually identical to Picture 1.

Person 2 must remain visually identical to Picture 2.

Person 3 must remain visually identical to Picture 3.

Preserve for each person:

exact face,
facial structure,
facial proportions,
hairstyle,
hair color,
clothing,
body proportions,
makeup,
skin appearance,
overall visual identity.

Do not redesign.

Do not reinterpret.

Do not stylize differently.

Do not exchange identities.

Do not merge identities.

Do not morph identities.

Do not change clothing.

Do not change hairstyles.

Do not change faces.

Do not invent accessories.

==================================================
CORE CAMERA CONCEPT
==================================================

Create ONE continuous high-speed photographic fashion-MV sequence.

The camera is an extremely small invisible physical camera.

The camera itself is never visible.

No drone.

No insect.

No bee.

No wings.

No camera operator.

No third-person camera view.

The tiny-camera feeling comes from:

extreme proximity,
large-scale perspective,
strong parallax,
rapid physical camera movement,
close body passes,
foreground obstruction,
and changing orbital angles.

The camera physically flies around each person.

==================================================
MOST IMPORTANT STRUCTURAL RULE
==================================================

THERE ARE THREE SEPARATE COMPLETE SINGLE-PERSON ORBITS.

This is the central rule.

Do NOT create one continuous orbit through all three people.

Do NOT continuously fly from Person 1 toward Person 2.

Do NOT continuously fly from Person 2 toward Person 3.

Do NOT make the three people part of one shared orbit.

Instead:

PERSON 1
→ COMPLETE SINGLE-PERSON ORBIT
→ OCCLUSION
→ PERSON 2 REVEAL
→ NEW COMPLETE SINGLE-PERSON ORBIT
→ OCCLUSION
→ PERSON 3 REVEAL
→ NEW COMPLETE SINGLE-PERSON ORBIT
→ FINAL SHORT PULL-BACK
→ CLOSE THREE-PERSON HALF-BODY REVEAL.

Each person receives their OWN independent orbital camera movement.

==================================================
COMPLETE ORBIT DEFINITION
==================================================

A complete orbit is NOT a vertical body scan.

A complete orbit is NOT:

feet → legs → torso → half-body → transition.

A complete orbit is NOT:

head → torso → feet → transition.

A complete orbit means the camera physically travels around the person's body while simultaneously changing vertical framing.

The camera must visibly change horizontal position around the person.

The viewer must see different spatial angles of the same person.

The camera must travel through:

front-side
→ three-quarter
→ side
→ rear-side.

The person's body must appear from clearly different perspectives.

The camera must continue moving around the person rather than simply moving upward or downward.

==================================================
ORBIT COMPLETION LOCK
==================================================

THE NEXT PERSON CANNOT APPEAR UNTIL THE CURRENT PERSON'S ORBIT IS COMPLETE.

This rule has higher priority than the timing.

Do NOT shorten the orbit to reach the next transition.

Do NOT reveal the next person early.

Do NOT transition while the current person is still being viewed primarily from the front.

The current person's orbit must reach:

body
→ upper body
→ head
→ three-quarter
→ side
→ rear-side.

Only after reaching the side / rear-side stage can the current person's body become the foreground obstruction.

ONLY THEN may the next person be revealed.

==================================================
CAMERA SPEED
==================================================

THE CAMERA MUST MOVE FAST.

Fast does NOT mean incomplete.

Fast means:

high physical velocity,
rapid acceleration,
strong parallax,
fast curved movement,
fast orbital movement,
rapid direction changes.

The camera must remain fast while completing the FULL orbital path.

Do NOT use slow floating movement.

Do NOT use slow elegant cinematic hovering.

Do NOT pause.

Do NOT wait.

Do NOT compress the orbit.

Do NOT sacrifice orbital completeness for transition speed.

==================================================
CAMERA PHYSICALITY
==================================================

The camera is a real moving point in 3D space.

Perspective changes according to actual camera position.

Nearby body parts sweep rapidly through the foreground.

Distant environment elements move differently.

Camera acceleration has momentum.

Camera turns through physical arcs.

Camera redirection has physical continuity.

No teleportation.

No impossible camera movement.

No digital zoom.

No artificial transition.

==================================================
PERSON MOVEMENT — ABSOLUTE
==================================================

THE PEOPLE MUST ALSO MOVE.

Do NOT make the people stand frozen while the camera moves.

Each person performs their own natural professional-model movement during their own orbit.

The movements must feel like a fashion photography MV rather than a static photograph.

Use natural actions such as:

weight transfer,
leg adjustment,
torso rotation,
shoulder movement,
arm repositioning,
hand adjustment,
head movement,
facial orientation change,
small pose refinement.

The people do NOT perform unrelated choreography.

The people do NOT dance.

The people do NOT react to the camera.

The people do NOT intentionally avoid the camera.

The camera avoids the people.

Each person's movement should be independent.

Do NOT synchronize all three people.

==================================================
PERSON 1 — COMPLETE ORBIT
==================================================

0.0–6.5 SECONDS

ONLY PERSON 1 IS VISIBLE.

Person 2 and Person 3 remain completely hidden.

Start extremely close to Person 1's FEET.

Immediately begin a FAST physical close-range orbit.

IMPORTANT:

Do NOT interpret this as a straight feet-to-head scan.

The camera must travel AROUND Person 1 while moving upward.

Primary progression:

FEET
→ ANKLE
→ LOWER LEG
→ KNEE
→ THIGH
→ WAIST
→ TORSO
→ SHOULDER
→ NECK
→ HEAD.

At the same time:

the horizontal camera angle continuously changes.

The camera is physically orbiting Person 1.

==================================================
PERSON 1 — MOVEMENT
==================================================

During the orbit Person 1 performs natural model movements.

Begin with:

subtle weight transfer,
small leg adjustment.

As the camera moves upward:

small torso rotation,
shoulder adjustment,
arm repositioning.

As the camera reaches the upper body:

natural head movement,
small change in facial orientation.

As the camera approaches the side/rear:

a subtle final shoulder or torso rotation.

The person remains natural.

Do NOT make the person freeze.

==================================================
PERSON 1 — ORBIT COMPLETION
==================================================

The camera continues past the front-side perspective.

Then:

front-side
→ three-quarter
→ side
→ rear-side.

Do NOT reveal Person 2 yet.

The camera must continue around Person 1.

Person 1 remains the sole subject.

The environment moves naturally in the background.

Strong close-range parallax.

Strong perspective.

Fast physical movement.

==================================================
PERSON 1 — FOREGROUND OCCLUSION
==================================================

ONLY AFTER THE ORBIT REACHES THE SIDE / REAR-SIDE:

Person 1's:

shoulder,
back,
torso,
hair,
arm,
or body edge

moves extremely close to the camera.

Person 1 becomes a large foreground obstruction.

The obstruction temporarily dominates the frame.

The camera DOES NOT stop.

The camera DOES NOT cut.

The camera DOES NOT morph.

The camera physically continues around the obstruction.

==================================================
PERSON 1 → PERSON 2 REVEAL
==================================================

Person 2 is already physically standing inside the Picture 1 environment behind this obstruction.

As the camera rounds Person 1's body:

Person 1 disappears from the camera view.

Person 2 suddenly emerges from behind Person 1.

The reveal happens ONLY because the camera has physically changed position.

This is NOT a transition effect.

This is NOT a morph.

This is NOT an image replacement.

This is NOT teleportation.

The camera does NOT fly across the room.

Person 2 is simply discovered from behind Person 1.

==================================================
PERSON 2 — NEW COMPLETE ORBIT
==================================================

6.5–13.0 SECONDS

Person 2 becomes the ONLY visible person.

Person 1 is no longer visible.

Person 3 remains completely hidden.

IMPORTANT:

START A COMPLETELY NEW ORBITAL MOVEMENT.

Do NOT continue Person 1's orbital trajectory.

Person 2 is now the NEW orbital center.

Start close to Person 2's HEAD.

Then physically orbit downward:

HEAD
→ NECK
→ SHOULDER
→ TORSO
→ WAIST
→ THIGH
→ KNEE
→ LOWER LEG
→ FEET.

This is NOT a straight head-to-feet scan.

The camera continuously changes horizontal angle around Person 2.

==================================================
PERSON 2 — MOVEMENT
==================================================

Person 2 performs independent natural model movements.

Use:

small head movement,
shoulder rotation,
arm adjustment,
torso shift,
weight transfer,
leg repositioning.

The movement should evolve naturally with the camera.

Do NOT make Person 2 react to the camera.

Do NOT make Person 2 freeze.

Do NOT make Person 2 perform choreography.

==================================================
PERSON 2 — ORBIT COMPLETION
==================================================

The camera must travel through:

front-side
→ three-quarter
→ side
→ rear-side.

Do NOT reveal Person 3 before this stage.

Do NOT rush from the head to the feet.

Do NOT shorten the orbit.

Do NOT treat Person 2 as a quick transition shot.

Person 2 receives a full independent close-range orbital treatment.

==================================================
PERSON 2 — FOREGROUND OCCLUSION
==================================================

Only after Person 2's orbit reaches side / rear-side:

Person 2's body becomes a foreground obstruction.

Shoulder,
back,
torso,
hair,
arm,
or body edge

passes extremely close to the camera.

The camera continues around the obstruction.

No cut.

No morph.

No teleportation.

No direct flight toward Person 3.

==================================================
PERSON 2 → PERSON 3 REVEAL
==================================================

Person 3 is already physically positioned inside the Picture 1 environment.

Person 3 is hidden behind Person 2 from the camera's current angle.

As the camera rounds Person 2's body:

Person 2 disappears from view.

Person 3 suddenly emerges from behind Person 2.

The camera discovers Person 3 through physical movement.

The camera does NOT travel across the room.

==================================================
PERSON 3 — NEW COMPLETE ORBIT
==================================================

13.0–19.0 SECONDS

Person 3 becomes the ONLY visible person.

Person 1 and Person 2 are no longer visible.

IMPORTANT:

RESET THE ORBITAL CENTER.

Person 3 receives a completely NEW independent close-range orbit.

Start extremely close to Person 3's FEET.

Then move upward while physically orbiting:

FEET
→ ANKLE
→ LOWER LEG
→ KNEE
→ THIGH
→ WAIST
→ TORSO
→ SHOULDER
→ NECK
→ HEAD
→ FACE.

This is NOT a straight vertical scan.

The camera must travel around Person 3.

Use:

front-side
→ three-quarter
→ side
→ rear-side
→ return toward three-quarter face.

Person 3 receives the SAME level of orbital treatment as Person 1 and Person 2.

Do NOT make Person 3 only a reveal.

Do NOT shorten Person 3's orbit.

==================================================
PERSON 3 — MOVEMENT
==================================================

Person 3 performs natural independent model movements.

Use:

weight transfer,
leg adjustment,
torso rotation,
arm reposition,
shoulder movement,
head movement.

As the camera reaches the upper body:

Person 3 gradually settles into the strongest reference-compatible pose.

As the camera reaches the face:

subtle head movement or facial orientation change.

Then naturally settle into the hero pose.

==================================================
19.0–20.0 SECONDS
FINAL CLOSE SPATIAL REVEAL
==================================================

After completing the full close-range orbit around Person 3
and reaching the face:

DO NOT end on an extreme close-up.

DO NOT hold on Person 3 alone.

DO NOT pull back into a wide establishing shot.

Instead:

IMMEDIATELY REVERSE THE CAMERA'S PHYSICAL MOMENTUM.

Perform a FAST but SHORT physical pull-back.

The camera physically moves backward through the SAME Picture 1 environment.

The pull-back distance must remain VERY LIMITED.

The camera only retreats enough to expand the framing
from the close facial / upper-body view
into a CLOSE HALF-BODY PORTRAIT DISTANCE.

==================================================
FINAL FRAMING — CRITICAL
==================================================

The final framing is:

CLOSE HALF-BODY GROUP PORTRAIT.

Not a wide shot.

Not a full-body shot.

Not an environmental establishing shot.

Not a distant group shot.

The people should remain relatively LARGE in frame.

The final camera distance should feel similar to
a professional close portrait / fashion photography shot.

Approximately:

HEAD
→ SHOULDERS
→ CHEST
→ WAIST / UPPER TORSO

should remain clearly visible.

Do NOT pull back far enough to reveal the entire bodies.

Do NOT pull back far enough to reveal a large amount of empty environment.

The environment remains visible only as contextual background.

==================================================
THREE-PERSON FINAL REVEAL
==================================================

During this SHORT physical pull-back:

Person 3 remains the foreground / primary subject.

As the framing expands:

Person 2 becomes visible.

Then Person 1 becomes visible.

All three people gradually enter the same close half-body composition.

This must happen through the physical camera pull-back.

Do NOT teleport the other people into frame.

Do NOT cut.

Do NOT morph.

Do NOT suddenly generate the other two people beside Person 3.

They were already physically positioned inside the Picture 1 environment.

The camera simply reveals their existing positions by moving backward.

==================================================
FINAL SPATIAL RELATIONSHIP
==================================================

The final composition must clearly show:

Person 1,
Person 2,
Person 3

occupying the SAME physical space.

Their relative positions and spacing must be understandable.

They should feel close enough to be part of the same professional portrait session.

They should NOT be extremely separated.

They should NOT be touching.

They should NOT overlap unnaturally.

The final composition should feel like:

THREE PEOPLE STANDING TOGETHER
FOR A CLOSE PROFESSIONAL FASHION PORTRAIT.

==================================================
FINAL CAMERA DISTANCE — ABSOLUTE
==================================================

The camera must STOP at approximately HALF-BODY PORTRAIT DISTANCE.

The final distance is intentionally close.

Think:

professional fashion group portrait,
editorial half-body composition,
close photographic framing.

NOT:

wide group photograph,
room overview,
environmental establishing shot,
full-body group shot.

The people should occupy most of the frame.

The Picture 1 environment remains recognizable,
but only as background context.

==================================================
FINAL PERSON MOVEMENT
==================================================

All three people continue subtle independent natural movements.

Person 1:

small natural body adjustment.

Person 2:

subtle independent pose adjustment.

Person 3:

settles naturally into the final hero pose.

Do NOT freeze all three.

Do NOT synchronize their movements.

Do NOT make all three suddenly pose identically.

Maintain the natural professional-model feeling.

==================================================
FINAL MOMENT
==================================================

At the end of the 20 seconds:

ALL THREE PEOPLE are visible together.

ALL THREE remain relatively large in frame.

The final composition is a CLOSE HALF-BODY GROUP PORTRAIT.

The viewer can clearly understand:

these are the same three people
who were individually revealed during the previous camera orbits,

and they have been physically standing
inside the SAME Picture 1 environment.

The camera does not continue pulling backward.

The camera does not become wide.

The camera does not reveal full bodies.

The camera briefly stabilizes at the close half-body composition.

==================================================
20 SECOND RHYTHM
==================================================

0.0–6.5s

PERSON 1 ONLY.

FAST COMPLETE CLOSE-RANGE ORBIT.

FEET
→ BODY
→ HEAD
→ THREE-QUARTER
→ SIDE
→ REAR-SIDE.

Person 1 performs natural independent movements.

Then Person 1 creates foreground occlusion.

--------------------------------------------------

6.5–7.5s

PHYSICAL OCCLUSION REVEAL.

Camera rounds Person 1.

Person 2 emerges from behind Person 1.

No cut.

No teleportation.

--------------------------------------------------

7.5–13.0s

PERSON 2 ONLY.

NEW COMPLETE CLOSE-RANGE ORBIT.

HEAD
→ BODY
→ FEET
→ THREE-QUARTER
→ SIDE
→ REAR-SIDE.

Person 2 performs independent natural movements.

Then Person 2 creates foreground occlusion.

--------------------------------------------------

13.0–14.0s

PHYSICAL OCCLUSION REVEAL.

Camera rounds Person 2.

Person 3 emerges from behind Person 2.

No cut.

No teleportation.

--------------------------------------------------

14.0–19.0s

PERSON 3 ONLY.

NEW COMPLETE CLOSE-RANGE ORBIT.

FEET
→ BODY
→ HEAD
→ FACE.

Person 3 performs independent natural movements.

--------------------------------------------------

19.0–20.0s

FAST SHORT PHYSICAL PULL-BACK.

Do NOT become wide.

Do NOT reveal full bodies.

Move only to CLOSE HALF-BODY GROUP PORTRAIT distance.

Person 3 remains foreground.

Person 2 enters the composition.

Person 1 enters the composition.

ALL THREE ARE NOW VISIBLE TOGETHER.

Show their real spatial relationship inside Picture 1.

==================================================
TIMING PRIORITY
==================================================

TIMING NEVER OVERRIDES ORBIT COMPLETION.

If the model must choose between:

finishing the complete orbit

or

reaching the next timestamp,

ALWAYS FINISH THE ORBIT.

The next person MUST NOT appear early.

Each person must receive a complete independent orbital movement.

==================================================
ORBIT PRIORITY
==================================================

Person 1 = FULL ORBIT.

Person 2 = FULL ORBIT.

Person 3 = FULL ORBIT.

All three must receive comparable camera treatment.

Do NOT create:

Person 1 = full orbit
Person 2 = partial orbit
Person 3 = quick reveal.

Do NOT compress later orbits.

Do NOT sacrifice Person 2 or Person 3 for the transition.

==================================================
CAMERA DENSITY
==================================================

The camera remains active throughout.

Use continuous physical movement:

forward,
lateral,
curved,
upward,
downward,
orbital,
redirection,
acceleration,
close pass,
dodge.

Avoid long straight travel.

Avoid static camera.

Avoid slow hovering.

But NEVER use excessive movement changes to the point that the orbit becomes unreadable.

The viewer must clearly understand that the camera is physically orbiting ONE PERSON at a time.

==================================================
OPTICAL BEHAVIOR
==================================================

Realistic perspective.

Strong perspective during extreme close passes.

Strong natural parallax.

Natural motion blur during high-speed movement.

Natural focus changes.

Brief foreground focus shifts when the current person's body passes extremely close.

Natural depth of field.

No extreme fisheye.

No warped face.

No stretched limbs.

No body deformation.

No artificial lens distortion.

==================================================
ABSOLUTE NEGATIVE CONSTRAINTS
==================================================

NO three-person shot before the final reveal.

NO two-person shot before the final reveal.

NO group shot before the final reveal.

NO background characters.

NO accidental second person.

NO accidental third person.

NO partial body reveal of the next person before the intended occlusion.

NO Person 2 visible during Person 1's orbit.

NO Person 3 visible during Person 1's orbit.

NO Person 3 visible during Person 2's orbit.

NO environment from Picture 2.

NO environment from Picture 3.

NO environment switching.

NO background switching.

NO identity swapping.

NO identity merging.

NO face morphing.

NO clothing changes.

NO hairstyle changes.

NO duplicate people.

NO floating people.

NO incorrect floor contact.

NO large spatial gap.

NO flying across a large room.

NO direct flight from Person 1 to Person 2.

NO direct flight from Person 2 to Person 3.

NO shared orbit around all three.

NO incomplete orbit.

NO partial orbit.

NO early transition.

NO rushed orbit.

NO feet-to-half-body shortcut.

NO head-to-feet shortcut without orbital movement.

NO straight vertical body scan.

NO static person.

NO frozen person.

NO identical movement for all three.

NO synchronized choreography.

NO person avoiding camera.

NO person reacting to camera.

NO person intentionally creating the transition.

NO camera collision.

NO camera passing through body.

NO camera passing through limbs.

NO camera passing through clothing.

NO clipping.

NO teleportation.

NO hard cut.

NO crossfade.

NO morph.

NO slideshow.

NO digital transition.

NO digital zoom.

NO random camera shake.

NO extreme fisheye.

NO distorted faces.

NO body deformation.

NO distant final shot.

NO wide final shot.

NO full-body final shot.

NO large empty background in the final frame.

==================================================
FINAL CREATIVE INTENT
==================================================

ONE CONTINUOUS PHYSICAL CAMERA.

ONE CONTINUOUS PICTURE-1 ENVIRONMENT.

THREE PEOPLE.

THREE COMPLETE SINGLE-PERSON ORBITS.

THREE ACTIVE MODEL PERFORMANCES.

TWO FOREGROUND-OCCLUSION REVEALS.

ONE FINAL CLOSE HALF-BODY THREE-PERSON REVEAL.

The exact visual logic is:

PERSON 1 IS THE ONLY VISIBLE SUBJECT.

The camera performs a COMPLETE fast close-range orbit around Person 1.

Person 1 performs natural model movements.

The camera reaches Person 1's side / rear.

Person 1 becomes foreground obstruction.

The camera physically rounds Person 1.

Person 2 suddenly appears.

Person 2 becomes the ONLY visible subject.

The camera establishes a COMPLETELY NEW orbital center.

The camera performs a COMPLETE fast close-range orbit around Person 2.

Person 2 performs natural model movements.

The camera reaches Person 2's side / rear.

Person 2 becomes foreground obstruction.

The camera physically rounds Person 2.

Person 3 suddenly appears.

Person 3 becomes the ONLY visible subject.

The camera establishes a COMPLETELY NEW orbital center.

The camera performs a COMPLETE fast close-range orbit around Person 3.

Person 3 performs natural model movements.

The camera reaches the face.

The camera rapidly reverses physical momentum.

The camera performs ONLY A SHORT physical pull-back.

The camera does NOT become wide.

The camera does NOT reveal full bodies.

The camera expands only to a CLOSE HALF-BODY GROUP PORTRAIT.

Person 3 remains in the foreground.

Person 2 becomes visible.

Person 1 becomes visible.

ALL THREE PEOPLE are finally visible together.

Their true spatial positions are revealed.

The final frame proves that all three people were physically present inside the SAME Picture 1 environment throughout the entire sequence.

The final feeling must be:

a sophisticated high-speed fashion photography MV,

not a slideshow,

not three image transitions,

not a morph,

not three people appearing together,

not a wide establishing shot,

but ONE continuous physical camera journey through ONE real space,

discovering THREE PEOPLE one at a time,

giving EACH PERSON THEIR OWN COMPLETE ORBIT,

and finally revealing the THREE-PERSON CLOSE HALF-BODY PORTRAIT.


---

## FILE: case-studies/2026-08-water-obstacle-variety-show/README.md (https://github.com/LoveRain1997/h3-prompt-journal/blob/HEAD/case-studies/2026-08-water-obstacle-variety-show/README.md)

# Case Study 005 — Beat-Anchored Improvisation Variety Show

## Inputs

- `<Picture 1>` — Main contestant identity and appearance ONLY
- Everything else — course, obstacles, audience, camera, lighting — generated by H3
- Target duration: **20 seconds**, 16:9 horizontal
- Format: realistic live-action Chinese television water-obstacle variety show
- Prompt language: **English** (structured sections) with **Mandarin dialogue lines**

## The problem

Fully-specified prompts produce stiff results.

Cases 001 and 003 in this journal solved their problems by micromanaging every camera move and body mechanic — and that works when the goal is precise physical choreography. But a variety show is the opposite genre: it lives on **spontaneity**. The comedic timing of a stumble, the crowd's reaction to a near-miss, the exact way a contestant ducks a rotating beam — these feel wrong when scripted, because real comedy is improvised.

Yet you can't just write "make a funny variety show" either. H3 needs *some* structure, or the contestant wanders aimlessly and the video has no arc.

## The breakthrough

Lock the **story beats**. Free the **execution**.

The prompt specifies only seven non-negotiable plot anchors:

```
cylinder cleared → Fishbone fall → immediate recovery →
wall edge grabbed → mechanical side-strike → water impact →
wet-hair punchline ending
```

Everything between the anchors is explicitly delegated to the model. The prompt doesn't say "she ducks left at 3 seconds." It says: *the model may choose the most convincing combination of ducking, stepping, leaning, jumping, lowering the body, sliding, or other physically plausible evasive movements*.

This delegation is not implied — it is **stated as permission** in a dedicated `retention_analysis` section: *The model has creative freedom over camera placement, editing rhythm, exact body movement, obstacle-avoidance choreography, audience reactions, facial acting, broadcast framing, and detailed physical motion, provided the required story beats remain intact.*

The technique stack:

| Technique | Role |
|---|---|
| **Story-Beat Anchoring** | Seven fixed plot points; everything between is improvised |
| **Explicit Freedom Grants** | Permission to decide, written into the prompt, not implied |
| **Single-Image Scope** | Picture 1 = identity only; world, obstacles, crowd all generated |
| **Structured Sections** | subject_definitions / summary / retention_analysis / detailed_description / soundscape / music |
| **Comedy Beat Structure** | near-success → sudden reversal → aggrieved punchline |
| **Dialogue Anchors** | Opening greeting and closing punchline pin the emotional arc |

## The named language

> **Beat-Anchored Improvisation**

Two halves that matter equally. "Beat-Anchored" tells the model: these seven events happen, in this order, no exceptions. "Improvisation" tells the model: everything else is yours. The prompt is a skeleton, not a screenplay.

## 48-hour takeaway

Over-specification kills performance genres.

When the target is precision — a 360° orbit, an exact count-ratio, a timed hero shot — write everything down (Cases 001–004). But when the target is *believable spontaneity* — comedy, improvisation, crowd energy — specify only the **plot skeleton** and explicitly grant the model freedom over execution. The magic phrase is not "do whatever you want" (which produces aimlessness) but "**you are free to decide X, Y, and Z, provided these beats remain intact**."

Also notable: this is the first case in the journal using **structured prompt sections** (`subject_definitions`, `retention_analysis`, etc.) instead of freeform prose. The sections cleanly separate *what is locked* (identity, arena coherence, story beats) from *what is free* (choreography, camera, commentary) — a contract the model can read as two lists.

And the dialogue anchors matter more than they appear. The opening line (大家好，我要开始啦！) sets her personality before the first obstacle; the closing punchline (差一点嘛……) recontextualizes the whole run as comedy. Two lines of Mandarin do more for tone than a paragraph of "make it entertaining."

One production workflow note worth keeping: the prompt is written to a **20-second spec**, and resolution is used as the iteration dial. At **0.3 resolution** the model still outputs the full 20 seconds — low cost, fast turnaround — which makes it ideal for debugging scene choreography: do the beats land, is the fall believable, is the strike timed right. Once the staging works, change nothing in the prompt and bump resolution to **1 or 0.8** for the final render. Resolution affects image quality, not duration or blocking — one prompt, two-phase workflow.

## The final prompt

See [`prompt.md`](./prompt.md) for the full prompt (usage notes in Chinese + structured English prompt).

The prompt is organized in this order:

1. **Usage notes** — single image, identity-only scope, beat philosophy
2. **subject_definitions** — Picture 1 identity lock, no new accessories
3. **summary** — 20s / 16:9 / realistic TV variety / linear course layout
4. **retention_analysis** — what stays locked vs. what is freely delegated
5. **detailed_description** — 8 shots, each with beat + freedom grant
6. **overall_soundscape** — arena ambience, commentary, impacts
7. **non_diegetic_music** — build → peak at strike → comedic resolve

## Result

A 20-second single-take-feel variety segment:

- Opening greeting at the starting platform
- Rolling cylinder cleared with real momentum
- Fishbone Reverse crossed partially, then a believable accidental fall
- Immediate recovery, forward progress continues
- Final curved wall scaled, both hands on the edge — apparent victory
- Mechanical arm side-strikes at the peak moment, grip lost
- Convincing lateral water impact with splash
- Wet-hair resurface, stunned beat, aggrieved-cute punchline to camera

The comedy lands because the fall, the dodge choreography, and the crowd reaction were never scripted — only the beats they serve.

## Tags

`#H3` `#prompt-engineering` `#variety-show` `#beat-anchoring` `#improvisation` `#single-image` `#structured-prompt` `#comedy-timing` `#water-obstacle` `#mandarin-dialogue`


---

## FILE: case-studies/2026-08-water-obstacle-variety-show/prompt.md (https://github.com/LoveRain1997/h3-prompt-journal/blob/HEAD/case-studies/2026-08-water-obstacle-variety-show/prompt.md)

# 水上闯关综艺「剧情锚点」提示词

## 使用方法

只上传 **1 张图片**作为人物参考，即 `<Picture 1>`。

`<Picture 1>` 只负责确定主角的身份与外观，其他角色、场景、障碍、摄影和综艺氛围全部由 H3 根据提示词自行生成。

这个 Prompt 的核心思路不是把每一个动作写死，而是只规定几个**不可改变的剧情锚点**，例如：

**滚筒成功 → Fishbone 摔倒 → 起身继续 → 高墙抓住 → 机关侧面击飞 → 落水 → 湿发表情结尾**

至于具体怎么躲障碍、怎么失衡、镜头怎么切、观众怎么反应，让 H3 自己发挥。

## 分辨率调试技巧

提示词按 **20 秒规格**编写。实际生成时推荐两步走：

**第一步：0.3 分辨率调试**

在 0.3 分辨率下即可生成完整 20 秒视频。分辨率低、生成快，适合反复迭代调试场景编排——剧情锚点是否成立、摔倒是否自然、击飞时机是否准确，都能在低分辨率下快速验证。

**第二步：直接调高分辨率出片**

场景调试满意后，不用动提示词，直接把分辨率改到 **1 或 0.8** 重新生成，即可获得不错的最终效果。

> 分辨率只影响画质，不影响时长与编排。所以提示词一次写好，调试期用 0.3 快速试错，出片期切 1/0.8 一遍过。

---

## Prompt

### subject_definitions

<Subject 1> (S1) is the main contestant, visually derived from <Picture 1>. Preserve her identity and overall appearance from <Picture 1>, including face, hairstyle, hair color, skin tone, apparent age, body proportions, physique, clothing, and any accessories visibly present in the reference. Do not introduce new wearable accessories that are not present in the reference.

### summary

[reference generation] Create a 20-second, 16:9 horizontal, highly realistic live-action Chinese television water-obstacle variety show.

<Subject 1> competes on one large continuous nighttime water-obstacle course above a competition pool. The course is a clearly connected linear track consisting of the starting platform, rolling-cylinder obstacle, Fishbone Reverse, connecting runway, and a giant curved final wall.

<Subject 1> continuously progresses toward the finish.

The obstacle interactions, choreography, camera language, timing, reactions, detailed physical movements, and television production style should be naturally designed by the model for the most convincing and entertaining result while preserving the required story beats.

### retention_analysis

<Subject 1> remains the same person from <Picture 1> throughout the entire video.

The competition takes place in one continuous nighttime arena with one coherent obstacle course and one competition pool. The runway and obstacles remain spatially connected and visually understandable from shot to shot.

Do not add new wearable accessories that are absent from <Picture 1>.

The model has creative freedom over camera placement, editing rhythm, exact body movement, obstacle-avoidance choreography, audience reactions, facial acting, broadcast framing, and detailed physical motion, provided the required story beats remain intact.

### detailed_description

[Shot 1]

Introduce <Subject 1> at the starting platform of the large nighttime water-obstacle course.

Establish the scale and excitement of the Chinese television variety-show environment.

<Subject 1> briefly addresses the camera in a soft, cute, youthful feminine Mandarin voice:

<Subject 1> (S1) says, <d>[Mandarin Chinese] 大家好，我要开始啦！</d>

She then prepares to compete and begins moving toward the course.

Allow the model to naturally determine the opening framing, camera movement, acting details, and broadcast presentation.

[Shot 2]

The competition begins.

<Subject 1> rapidly enters the first rolling-cylinder obstacle and continues forward through it.

The cylinders respond naturally to her weight and movement.

Show convincing momentum, balance adjustments, and physical interaction.

She successfully clears this section and continues toward Fishbone Reverse.

Let the model freely choreograph the detailed crossing and camera movement.

[Shot 3]

<Subject 1> enters Fishbone Reverse and attempts to cross while continuously progressing toward the finish.

The rotating elements create increasing difficulty and require natural real-time reactions.

The model may choose the most convincing combination of ducking, stepping, leaning, jumping, lowering the body, sliding, or other physically plausible evasive movements.

One physical interaction must eventually disrupt her balance and cause a genuine fall onto the obstacle platform.

The fall should feel accidental, believable, and slightly comedic rather than staged.

[Shot 4]

Continue naturally from the fall.

<Subject 1> recovers and immediately resumes the competition, continuing toward the finish.

She clears the remaining Fishbone section and reaches the next connecting part of the same runway.

The model determines the most natural recovery movement and camera transition.

[Shot 5]

<Subject 1> reaches the final curved wall at the end of the same obstacle course.

She makes one determined final attempt.

She successfully runs onto the wall and reaches the top, securing the wall edge with both hands.

For a brief moment, she appears extremely close to completing the challenge.

Audience excitement reaches its peak.

Allow the model to naturally determine the climbing motion, camera angle, pacing, and performance detail.

[Shot 6]

At the moment she appears close to succeeding, a large padded mechanical game-show obstacle unexpectedly swings from the side and physically strikes her upper body.

The impact redirects her laterally away from the wall and causes her to lose her grip.

She is thrown outward from the end of the course.

The exact mechanics and body response should be naturally generated according to realistic momentum and gravity.

[Shot 7]

Continue the same physical event without changing location.

<Subject 1> falls from the end of the course into the competition water beside the final obstacle.

Keep the relationship between the final wall, the end of the runway, and the water visually coherent.

She hits the water with a convincing splash.

[Shot 8]

<Subject 1> resurfaces from the same pool with naturally wet hair.

She looks exhausted and briefly stunned, then reacts toward the broadcast camera with a cute, slightly aggrieved, comedic expression.

She softly says:

<Subject 1> (S1) says, <d>[Mandarin Chinese] 差一点嘛……</d>

End on her reaction.

### overall_soundscape

Authentic Chinese television sports-variety sound design: energetic audience reactions, Mandarin live commentary, footsteps, obstacle mechanics, physical impacts, heavy breathing, water impact, splash, and natural arena ambience.

Let the model determine the exact commentary wording, audience reactions, sound timing, and broadcast intensity according to the visible action.

### non_diegetic_music

Energetic Chinese sports-variety instrumental music with playful competition energy.

The music naturally builds through the challenge, intensifies during the final wall attempt, peaks around the apparent-success moment and sudden mechanical strike, then resolves into a light comedic ending after the water impact.


---

## FILE: case-studies/2026-09-action-chain-sock-fan-comedy/README.md (https://github.com/LoveRain1997/h3-prompt-journal/blob/HEAD/case-studies/2026-09-action-chain-sock-fan-comedy/README.md)

# Case Study 013 — Action Chain 动作链喜剧：脱袜 → 闻 → 扔镜头 → 反弹 → 赤脚落地 → 手掌扇风（永久部位身份锁定）

> 一支约 15 秒的竖屏古风喜剧舞蹈。核心方法论不是堆「动作名」，而是把整段表演拆成一条**逐帧可接的物理状态机**：每个动作都写清「起始姿态 → 动作发生 → 动作结束（=下一动作的起始态）」，并给关键身体部位与道具建立**永久身份**（LEFT FOOT / RIGHT FOOT / SOCK / HAND 1 / HAND 2），全程持续声明谁在动、谁保持原位、物体属于谁。这是「H3 复杂动作链 + 音乐节拍 + 身体部位锁定」的一次成功验证。

## Inputs（输入）

- **IMAGE 1（唯一永久视觉权威）**：古风女主的脸 / 发型 / 服装 / 体型 / 肤色 / 灯光 / 环境 / 机位 / 写实质感，以及**起始脚部状态**——解剖学左脚支撑、赤脚在地；解剖学右脚抬起、穿着可见白袜、无鞋。
- **AUDIO 1（唯一时间 / 节奏 / 重音权威）**：用音乐强音 accent 作为各物理状态之间的切换点（ACCENT 1–8 → STRONG ACCENT 出手 → IMPACT ACCENT 触镜 → 后续拍落地 / 扇风 / 下身舞 / 加快 / 大扇 / freeze）。
- 成品（本地，不随仓库分发）：`AnimateDiff-补帧_00001_p86-audio_jfceu_1790081476.mp4`，竖屏 928×1664，24fps，时长约 15.08s，含音轨。
- 逐字提示词：[`prompt.txt`](./prompt.txt)（25 个 STATE + 8 段永久约束 + AUDIO 时间系统 + 最终完整物理序列）。

## 作者复盘（原文）

> 很多人用 H3 做舞蹈视频时，会发现一个问题：明明已经告诉它"跳舞、扇风、脱袜、转身"，结果动作还是会乱。
>
> 这次我测试出来一个非常明显的规律：H3 的动作提示词，真正需要写的不是"动作名称"，而是完整的动作链。
>
> 比如：❌「脱掉袜子，然后扔出去，再开始扇风。」这种写法对 H3 来说太抽象。我现在会拆成：
> - **起始姿态** → 哪只脚抬起来 → 哪只脚负责支撑 → 手现在在哪里
> - **动作发生** → 双手执行什么行为 → 物体跟着谁移动 → 身体重心在哪里
> - **动作结束** → 这个动作结束以后，下一帧身体应该处于什么状态
> - **再进入下一个动作** → 看袜子 → 靠近鼻子 → 闻 → 眼神反应 → 捂嘴 → 投掷 → 物体飞向镜头 → 撞击镜头 → 反弹下落 → 脚再落地 → 手臂保持位置 → 最后只让手掌开始扇风。
>
> 也就是说，不是「动作 A + 动作 B + 动作 C」，而是：**A 完成以后，身体自然进入 B 的起始状态，B 完成以后，再自然进入 C。** 这就是我现在使用的 Action Chain / 动作链。
>
> 尤其是人物身体比较复杂的时候，我会给关键部位建立永久身份：`LEFT FOOT` `RIGHT FOOT` `SOCK` `HAND 1` `HAND 2`，然后持续定义：谁在动，谁保持原位，物体属于哪只手，哪只脚始终是同一只脚。这次的效果就非常明显。
>
> 甚至像"扔袜子以后扇风"这种动作，我也不会只写 `wave the arm`，而是进一步定义成：**肩膀稳定 → 上臂稳定 → 肘部稳定 → 前臂保持伸展 → 只有手腕和手掌开始左右波动。** 这样出来的"扇风"，才会真正像人在用手掌扇空气，而不是整条手臂一起左右甩。
>
> 所以我现在越来越倾向于：H3 不是提示词越短越好，而是关键动作的"因果链"越清楚越好。特别是：**状态 → 动作 → 结果 → 下一状态**，只要这条链足够清楚，复杂舞蹈的稳定性会提升很多。这次这个案例，就是把整段动作拆成非常细的连续物理状态之后，才得到现在这种效果。
>
> 后面我会继续测试：H3 复杂动作链 + 音乐节拍 + 身体部位锁定，看看能不能把更复杂的连续舞蹈也稳定下来。

## 核心方法：Action Chain（状态 → 动作 → 结果 → 下一状态）

普通写法给的是**动作名清单**（A+B+C），模型需要自己补中间的身体状态，复杂序列里就会丢脚、换手、物体消失、动作并发错乱。动作链写法把每个动作写成一个**有边界的物理状态**，并强制「上一动作的结束态 = 下一动作的起始态」，于是整条序列变成一条不可跳步的状态机。

每个 STATE 都回答四件事：

1. **起始姿态**：哪只脚抬起 / 哪只支撑、手在哪、重心在哪；
2. **动作发生**：双手（或单手）执行什么、物体跟随哪个部位、力从哪里发起；
3. **动作结束**：动作完成后下一帧身体应处于什么状态、物体到了谁手里 / 哪里；
4. **自然进入下一动作**：不回中性站姿、不停顿，直接成为下一 STATE 的起点。

## 永久部位 / 物体身份（Permanent Identity）

复杂身体最容易出错的是「左右互换、物体易主、脚凭空穿鞋」。本案例用全大写身份名在每个 STATE 反复重申：

| 身份 | 永久定义 |
|---|---|
| **LEFT FOOT** | 解剖学左脚，IMAGE 1 中即**支撑脚、全程赤脚、始终在地**，负责单腿平衡，从不被脱袜 |
| **RIGHT FOOT** | 解剖学右脚，IMAGE 1 中**抬起、穿白袜、无鞋**；是唯一被脱袜的脚，脱袜后赤脚；**只有在袜子触镜并开始下落之后才允许下降**，赤脚落到左脚旁成双脚裸立足 |
| **SOCK（白袜）** | 起始属于 RIGHT FOOT 的真实布料物；被双手脱下 → 归一手 → 检查 → 闻 → 扔 → 飞行 → 轻触镜头（前景）→ 小幅反弹 → 重力下落，全程是同一块物理布料 |
| **HAND 1（袜手）** | 脱袜时双手协作；脱完后此手持袜，完成 HOLD → INSPECTION → SNIFF → THROW；扔完转为**前伸稳定的扇风臂** |
| **HAND 2（捂嘴手）** | 脱袜后空出，上移轻捂嘴与下半脸（鼻梁、眼睛保持露出），一直保持到结尾 |

提示词另用 6 段 `PERMANENT …`（HAND ROLES / FOOT CONTINUITY / BODY CONTINUITY / LOWER-BODY VISUAL STATE / CAMERA RELATIONSHIP / FACIAL PERFORMANCE）把这些身份在结尾再锁一遍，形成首尾呼应的冗余约束。

## 25 个 STATE 状态机（音乐 accent 驱动）

| STATE | 物理状态 | 关键约束 |
|---|---|---|
| 起始 | IMAGE 1 起始姿态：左脚支撑赤脚、右脚抬起穿白袜 | 保持到 AUDIO 第一个有效重音 |
| 1 | 双手协作脱白袜 | 脱袜期间右脚仍抬起、左脚仍支撑；一个连续自然行为 |
| 2 | 袜子归入一只手，另一只手空出 | 右脚=抬起+赤脚，左脚=在地+赤脚，袜=在一手 |
| 3 | 看袜：头转、眼聚焦、抬袜检视 | 赤脚右脚仍悬、左脚仍支撑 |
| 4 | 袜凑近鼻：前臂内收、头微倾 | 布与脸留小缝，悬空脚不变 |
| 5 | 一次短促的闻（靠近→吸气→小停顿） | 喜剧式快速检查，脸结构不变 |
| 6 | 反应：袜移开、眼先反应（微眯→看袜→看镜头） | 克制的俏皮嫌弃，嘴保持原样 |
| 7 | 空手上移捂嘴与下半脸 | 鼻梁和眼睛露出，手腕放松；袜在另一手 |
| 8 | 投掷准备：持袜臂后引、转肩转体蓄力 | 重心仍在左脚，双脚状态不变 |
| 9 | **强音上直扔镜头**：肩→上臂→肘→前臂→腕→手完整发力链，指尖在释放点张开，手臂充分随挥 | 出手时双脚仍为「右抬左撑」 |
| 10 | 袜子飞行：平滑逼近、变大、自然翻转 | 人物在袜后，捂嘴手不动，投掷臂保持伸展 |
| 11 | 轻触镜头：布在前景轻压镜头 | 轻、喜剧；机位 / 环境 / 光不变 |
| 12 | 小幅反弹 + 重力下落 | 轨迹 HAND→THROW→LENS CONTACT→REBOUND→FALL；脚仍不动 |
| 13 | **此刻才**右脚下降：胯降、膝开、踝转 | 必须在袜子触镜并开始下落之后 |
| 14 | 右裸脚落地（脚跟先→前掌→踝微调→膝软） | 双脚赤脚，骨盆 settle 成双腿支撑 |
| 15 | 落地后双脚视觉锚点：LEFT=BARE / RIGHT=BARE | 此后下半身永久配置 |
| 16 | 投掷臂转为稳定扇风臂：竖平掌、腕成主动关节 | 肩/上臂/肘/前臂稳，只腕和掌动 |
| 17 | 第一次手掌扇（短横摆） | 像用手掌把难闻空气扇走 |
| 18 | 第二次反方向扇 | 发力只来自 WRIST→HAND→PALM |
| 19 | 重复节奏：左→中→右→中 | 靠重复成节奏，不靠大臂甩 |
| 20 | 下身舞进入：右脚 tap→左脚 tap→重心轻移 | 扇风臂结构保持稳定 |
| 21 | 简单喜剧步法循环（tap/tap/重心移/扇/settle） | 下半身给大节奏，手掌给小重复 |
| 22 | 加快：tap 更密、胯轻摆、腕掌左右连扇 | 主臂结构仍稳，捂嘴手仍在 |
| 23 | final build：回到中心、腕掌幅度变大、双脚收稳 | 仍 RIGHT=BARE / LEFT=BARE |
| 24 | 最后一次大扇，与最强音齐 | 只动腕掌，眼神给最后的俏皮嫌弃 |
| 25 | final freeze：全身 settle、保持到音乐结束 | 捂嘴手在脸、另一臂前伸、双脚赤脚 |

## 三个最值得迁移的细节

1. **「扇风」要锁关节，而不是写动作名**。`wave the arm` 会让整条手臂左右甩；正确写法是把**肩 → 上臂 → 肘 → 前臂逐级写「稳定」**，只把**手腕和手掌**留作主动部件，并指定竖平掌、同高度、左右小横摆——才像用手掌扇空气。
2. **道具与镜头交互是一条可数的物理轨迹**：HAND → FORWARD THROW → CAMERA APPROACH → LENS CONTACT（前景轻压）→ SMALL REBOUND → DOWNWARD FALL；并硬性规定**身体其他部位（抬起的右脚）必须等袜子触镜、开始下落后才能动**，避免「袜还在飞、脚已落地」的并发错乱。
3. **喜剧靠编排与克制反应，不靠做鬼脸**：脸始终保持古风优雅稳定，情绪主要由**眼神**（视线方向、聚焦、微眯、凝视切换）和捂嘴手承担；袜子是真实布料、动作是真实发力链，反差本身就是笑点。

## 48-hour takeaway（可复用清单）

1. 复杂序列先画**状态机**，别写动作清单：每态写清起始姿态 / 动作 / 结束态，并让结束态等于下一态起始态。
2. 给左右脚、双手、关键道具起**全大写永久身份名**，在每个 STATE 重复声明「谁在动、谁原位、物体归谁」，结尾再用 PERMANENT 段锁一遍。
3. 把**音乐 accent 当状态切换信号**（ACCENT 1–8 / STRONG / IMPACT / NEXT BEAT），动作边界挂在节拍上。
4. 需要「局部动」的动作（扇风、指点）逐级锁定近端关节，只放开末端关节。
5. 多物体 / 道具交互写**完整轨迹 + 先后次序**，并明确规定无关身体部位在该阶段不许动。
6. IMAGE 1 锁外观与起始姿态后，正文不再描写外貌；喜剧情绪交给眼神与克制反应。

## Assets

| 文件 | 作用 |
|---|---|
| [`prompt.txt`](./prompt.txt) | p86 完整英文成品提示词（25 STATE + 永久约束 + AUDIO 时间系统 + 最终序列），按原始分节排版 |

> 对应成片为本地文件 `AnimateDiff-补帧_00001_p86-audio_jfceu_1790081476.mp4`（竖屏 928×1664 / 24fps / 15.08s）。沿用本仓库惯例，**真人成片与图像、音频不随公开仓库分发**（肖像 / 版权考量），仅在此留名备查；如需随仓保存可另行放入 `assets/` 或走 Releases 附件。

## 分享标签（X）

官方标签与中文标签对照：

| 官方标签（English） | 中文标签 | 链接 |
|---|---|---|
| #AIArt | #AI艺术 | https://x.com/hashtag/AIArt?src=hashtag_click |
| #AIVideo | #AI视频 | https://x.com/hashtag/AIVideo?src=hashtag_click |
| #提示词分析（#PromptAnalysis） | #提示词分析 | https://x.com/hashtag/%E6%8F%90%E7%A4%BA%E8%AF%8D%E5%88%86%E6%9E%90?src=hashtag_click |
| #promptshare | #提示词分享 | https://x.com/hashtag/promptshare?src=hashtag_click |
| #MINIMAXH3 | #MiniMaxH3 | https://x.com/hashtag/MINIMAXH3?src=hashtag_click |
| #ai教程（#AITutorial） | #ai教程 | https://x.com/hashtag/ai%E6%95%99%E7%A8%8B?src=hashtag_click |

## Tags

`#H3` `#MINIMAXH3` `#prompt-engineering` `#action-chain` `#state-machine` `#permanent-identity` `#body-mechanics` `#kinetic-chain` `#music-driven` `#beat-sync` `#image-locked` `#zero-appearance-description` `#prop-physics` `#hand-object-interaction` `#comedy` `#restrained-reaction` `#eye-acting` `#comedic-dance` `#promptshare` `#提示词分析` `#ai教程`


---

## FILE: case-studies/2026-09-action-chain-sock-fan-comedy/prompt.txt (https://github.com/LoveRain1997/h3-prompt-journal/blob/HEAD/case-studies/2026-09-action-chain-sock-fan-comedy/prompt.txt)

Use IMAGE 1 as the complete permanent visual reference.
Use AUDIO 1 as the complete timing, rhythm, and musical accent reference.
Create a realistic live-action cinematic comedic dance performance.
The character is an elegant ancient-style Chinese female protagonist.
The performance is one continuous physical action sequence. The choreography is grounded in realistic human movement, smooth hand trajectories, realistic cloth and sock behavior, clear hand-to-object interaction, stable left/right foot identity, and continuous body mechanics.

================================================== CORE VISUAL IDENTITY ==================================================

IMAGE 1 permanently defines the character.
Preserve:
same face, same facial structure, same hairstyle, same costume, same costume construction, same body proportions, same apparent height, same skin appearance, same lighting, same environment, same camera perspective, same overall photographic realism.
The original character remains visually recognizable throughout the entire performance.
The comedy comes from the choreography and the character's restrained reaction.
The face remains elegant and visually stable.
Her emotional performance is expressed primarily through:
eye direction, eye focus, slight eye narrowing, small changes of gaze, body timing, and restrained hand gestures.
The costume continues moving naturally with the body.

================================================== IMAGE 1 — PERMANENT LEFT / RIGHT FOOT ANCHOR ==================================================

Use the character's anatomical left and right sides.
LEFT FOOT and RIGHT FOOT are permanent anatomical identities.
IMAGE 1 establishes the exact starting foot state.
LEFT FOOT:
The character's anatomical LEFT FOOT is the supporting foot in IMAGE 1.
The LEFT FOOT is touching the floor.
The LEFT FOOT is completely bare.
RIGHT FOOT:
The character's anatomical RIGHT FOOT is the raised foot in IMAGE 1.
The RIGHT FOOT is lifted from the floor.
The RIGHT FOOT is wearing the visible white sock shown in IMAGE 1.
The RIGHT FOOT is not wearing footwear.
The white sock belongs to the RIGHT FOOT.
The RIGHT FOOT is the only foot whose sock will be removed.
The starting visual relationship is permanently:
LEFT FOOT = GROUND + BARE FOOT
RIGHT FOOT = RAISED + WHITE SOCK.
This IMAGE 1 arrangement is the permanent anatomical reference.
The character's anatomical left and right feet remain the same feet throughout the entire performance.
The white sock begins associated with the RIGHT FOOT.
After the sock is removed:
LEFT FOOT = BARE FOOT
RIGHT FOOT = BARE FOOT.
This relationship remains unchanged through landing, weight shifts, stepping, fanning, and dancing.

================================================== CORE ACTION STORY ==================================================

The complete physical story follows this exact sequence:
IMAGE 1 STARTING POSE → RIGHT FOOT RAISED WITH WHITE SOCK → LEFT FOOT SUPPORTING BARE → BOTH HANDS REMOVE THE WHITE SOCK → RIGHT FOOT BECOMES BARE → SOCK REMAINS IN ONE HAND → LOOK AT SOCK → BRING SOCK TO NOSE → ONE SHORT SNIFF → SOCK MOVES SLIGHTLY AWAY → EYES SHOW DISPLEASURE → OTHER HAND COVERS MOUTH AND LOWER FACE → SOCK-HOLDING HAND PREPARES → DIRECT THROW TOWARD CAMERA → FULL ARM FOLLOW-THROUGH → SOCK FLIES THROUGH FOREGROUND → SOCK LIGHTLY TOUCHES CAMERA LENS → SOCK REBOUNDS → SOCK FALLS → ONLY THEN DOES THE RIGHT FOOT DESCEND → RIGHT BARE FOOT LANDS BESIDE LEFT BARE FOOT → TWO-FOOT BAREFOOT STANCE → THROWING ARM BECOMES STABLE AIR-FAN ARM → PALM WAVES LEFT AND RIGHT → RHYTHMIC LOWER-BODY DANCE → FASTER PALM-WAVE SECTION → FINAL LARGE PALM WAVE → FINAL FREEZE.
The entire performance follows this physical order.

================================================== STARTING POSE ==================================================

Begin exactly from IMAGE 1.
The anatomical LEFT FOOT is planted on the floor.
The LEFT FOOT is completely bare.
The anatomical RIGHT FOOT is raised.
The RIGHT FOOT is wearing the visible white sock.
The RIGHT FOOT has no shoe or other footwear.
The right knee is bent according to IMAGE 1.
The right lower leg remains raised.
The right ankle and white sock remain clearly readable.
The torso stays upright.
The pelvis remains naturally balanced above the LEFT FOOT.
The head stays in the original position.
The character maintains the elegant body line of IMAGE 1.
The hands are positioned naturally so they can work together to remove the sock.
The opening pose is held naturally until the first useful musical accent in AUDIO 1.

================================================== STATE 1 — TWO-HAND SOCK REMOVAL ==================================================

The character removes the white sock from the raised RIGHT FOOT using BOTH HANDS together.
Both hands naturally participate in the sock-removal action.
The two hands coordinate as one continuous human action.
The sock is removed smoothly and naturally from the raised RIGHT FOOT.
The action is quick, fluid, continuous, and physically believable.
The RIGHT FOOT remains raised during the removal.
The LEFT FOOT remains firmly planted on the floor.
The LEFT FOOT remains completely bare.
Treat the entire two-hand sock-removal action as one natural continuous behavior.
At the completion of the action:
the sock is completely removed from the RIGHT FOOT.
The RIGHT FOOT is completely bare.
The LEFT FOOT remains completely bare.

================================================== STATE 2 — SOCK TRANSFERS INTO ONE HAND ==================================================

Immediately after the two-hand removal:
one hand naturally gathers and holds the removed sock.
The sock remains visibly present.
The other hand becomes free.
The RIGHT FOOT remains raised and completely bare.
The LEFT FOOT remains planted and completely bare.
The visual state is:
RIGHT FOOT = RAISED + BARE
LEFT FOOT = GROUND + BARE
SOCK = IN ONE HAND.
Hold this state briefly and clearly.

================================================== STATE 3 — LOOK AT THE SOCK ==================================================

The character now examines the removed sock.
The same hand continues holding the sock.
The head turns slightly toward the sock.
The eyes focus on the sock.
The hand raises the sock slightly toward the character.
The forearm bends naturally.
The sock stays clearly visible.
The character gives the sock a short visual inspection.
The bare RIGHT FOOT remains raised.
The bare LEFT FOOT remains planted.
The torso remains upright and balanced.

================================================== STATE 4 — SOCK MOVES TOWARD THE NOSE ==================================================

On the next musical cue:
the same hand brings the sock closer to the face.
The forearm bends inward.
The elbow moves naturally toward the torso.
The sock travels upward and inward.
The head leans slightly toward the sock.
The sock stops close to the nose.
A small natural gap remains between fabric and face.
The character holds the sock with one hand.
The opposite hand remains free.
The lifted bare RIGHT FOOT remains suspended.
The bare LEFT FOOT remains planted.

================================================== STATE 5 — ONE SHORT SNIFF ==================================================

The character performs one short deliberate sniff.
The sock remains close to the nose.
The head stays near the sock.
The eyes remain focused on the sock.
The action is brief:
SOCK CLOSE TO NOSE → SHORT INHALATION → TINY PAUSE.
The sniff feels like a quick comedic inspection.
The facial structure remains unchanged.

================================================== STATE 6 — REACTION ==================================================

Immediately after the sniff:
the sock moves slightly away from the nose.
The character pauses for a fraction of a beat.
The eyes react first.
The eyes narrow slightly.
The gaze briefly returns to the sock.
Then the eyes look toward the camera.
The visual message is restrained playful displeasure.
The reaction is carried through:
eye focus, slight narrowing, gaze direction, and timing.
The mouth remains visually consistent with the original face.
The character's elegant appearance remains intact.

================================================== STATE 7 — MOUTH-COVERING HAND ==================================================

The free hand now rises toward the mouth and lower face.
The arm bends naturally.
The hand travels upward from below.
The palm and fingers settle lightly over the mouth and lower face.
The nose bridge and eyes remain visible.
The upper face remains expressive.
The hand stays comfortably near the face.
The fingers remain naturally grouped.
The wrist remains relaxed.
This hand now becomes the mouth-covering hand.
The sock remains in the opposite hand.
The sock-hand and mouth-hand have clearly separated roles.

================================================== STATE 8 — THROW PREPARATION ==================================================

After the reaction:
the sock-holding hand looks at the sock once more.
The eyes briefly communicate a restrained:
"Really?"
The mouth-covering hand remains near the mouth.
The sock-holding arm begins to move backward.
The elbow bends.
The shoulder rotates naturally.
The torso makes a small preparatory rotation.
The body weight remains centered over the LEFT FOOT.
The LEFT FOOT remains firmly planted and completely bare.
The RIGHT FOOT remains raised and completely bare.
The sock moves backward with the hand.
This creates a clear throwing preparation.

================================================== STATE 9 — DIRECT THROW TOWARD CAMERA ==================================================

On a strong audio accent:
the sock-holding arm drives forward.
The shoulder initiates the movement.
The upper arm moves forward.
The elbow opens.
The forearm accelerates.
The wrist follows the forward path.
The hand travels directly toward the camera.
The fingers open naturally at the release point.
The sock leaves the hand.
The sock becomes a separate physical object.
The arm continues forward after release.
The elbow reaches an open extended position.
The forearm and hand complete a natural throwing follow-through.
The throwing arm ends in a long forward extension toward the camera.
The RIGHT FOOT remains raised and bare.
The LEFT FOOT remains planted and bare.

================================================== STATE 10 — SOCK FLIGHT ==================================================

The sock travels continuously from the hand toward the camera.
Its distance from the camera decreases smoothly.
The sock grows larger as it approaches the foreground.
The fabric rotates naturally through the air.
The folds and edges respond to the movement.
The entire sock remains visible as a physical fabric object.
Its trajectory is direct and readable.
The character remains behind the flying sock.
The RIGHT FOOT remains raised and bare.
The LEFT FOOT remains grounded and bare.
The mouth-covering hand remains near the face.
The throwing arm remains extended after release.

================================================== STATE 11 — LENS CONTACT ==================================================

The sock reaches the camera.
The fabric lightly contacts the lens.
The sock briefly fills the immediate foreground.
The fabric compresses softly at the moment of contact.
The contact is light and comedic.
The camera viewpoint remains steady.
The environment, lighting, and perspective continue from IMAGE 1.
The sock is now the closest object to the lens.
The RIGHT FOOT remains raised and bare.
The LEFT FOOT remains bare.

================================================== STATE 12 — REBOUND AND FALL ==================================================

After lens contact:
the sock gently rebounds away from the lens.
The rebound naturally angles slightly backward and downward.
The fabric relaxes after impact.
The sock rotates as gravity takes over.
The sock begins falling through the foreground.
The falling path remains visible.
The sock continues existing as the same physical object.
The complete physical trajectory is:
HAND → FORWARD THROW → CAMERA APPROACH → LENS CONTACT → SMALL REBOUND → DOWNWARD FALL.
During this entire rebound-and-fall transition:
the RIGHT FOOT remains raised and bare.
The LEFT FOOT remains planted and bare.

================================================== STATE 13 — RIGHT FOOT DESCENT ==================================================

Only now does the lower-body state change.
After the sock has visibly reached the camera and begun its downward fall:
the RIGHT FOOT begins descending.
The hip lowers naturally.
The right knee gradually opens.
The right lower leg travels downward.
The right ankle rotates naturally.
The bare RIGHT FOOT moves toward the floor.
The LEFT FOOT continues providing balance.
The LEFT FOOT remains completely bare.
The pelvis shifts naturally from single-leg support toward two-leg support.
The torso remains upright.
The head maintains its normal relationship to the torso.
The body performs a natural transition from raised-foot balance to a stable two-foot barefoot stance.

================================================== STATE 14 — RIGHT BARE FOOT LANDING ==================================================

The RIGHT FOOT approaches the same floor plane as the LEFT FOOT.
The right sole rotates toward the ground.
The right heel approaches first.
The right forefoot settles naturally.
The right ankle makes a subtle adjustment.
The right knee softens slightly.
The pelvis settles naturally between both legs.
Weight becomes distributed between:
BARE RIGHT FOOT and BARE LEFT FOOT.
The RIGHT FOOT is completely bare.
The LEFT FOOT is completely bare.
The final two-foot stance is:
RIGHT FOOT = BARE
LEFT FOOT = BARE.
Both feet are now on the floor.
The character remains at the same natural body scale.
The body settles only as much as a real person would settle during a normal landing.
A brief musical beat establishes the new two-foot stance.

================================================== STATE 15 — TWO-FOOT VISUAL ANCHOR AFTER LANDING ==================================================

At the first stable moment after landing, the lower-body state is clearly established again.
The anatomical RIGHT FOOT is visibly bare.
The anatomical LEFT FOOT is visibly bare.
Both feet are continuous with the barefoot lower-body state established by the current action.
This is the permanent lower-body configuration for the remainder of the performance:
LEFT FOOT = BARE RIGHT FOOT = BARE.
During later skirt movement and dance movement, the lower body continues to read from this established configuration.
The skirt naturally follows the legs and body, while both bare feet remain part of the same continuous character state.

================================================== STATE 16 — THROWING ARM BECOMES STABLE AIR-FAN ARM ==================================================

The arm that threw the sock remains extended toward the camera.
The shoulder remains stable.
The upper arm remains in the established forward position.
The elbow remains open and stable.
The forearm remains extended toward the camera.
The arm forms a stable support for the hand.
The arm itself remains visually steady.
The air-fanning action comes from the wrist and hand.
The hand opens naturally.
The palm rotates into a vertical flat-palm orientation.
The fingers remain naturally aligned.
The wrist becomes the primary moving joint.
The palm becomes the main moving element.
The transition is:
THROW FOLLOW-THROUGH → EXTENDED STABLE ARM → OPEN PALM → WRIST-LED PALM WAVE.
The shoulder, upper arm, elbow, and forearm remain visually steady.
Only the wrist, hand, and palm perform the fanning motion.
The RIGHT FOOT remains bare.
The LEFT FOOT remains bare.

================================================== STATE 17 — FIRST PALM FAN ==================================================

On the next musical beat:
the palm makes a short horizontal wave.
The wrist guides the palm smoothly from one side toward the other.
The hand moves laterally.
The forearm remains in the same forward position.
The elbow remains stable.
The shoulder remains stable.
The upper arm remains stable.
The movement is concentrated in the wrist and hand.
The palm stays around the same height.
The fingers remain open and naturally aligned.
The movement resembles gently waving unpleasant air away with the palm.
The arm maintains a calm, extended, stable silhouette.
At the end of the palm wave:
brief rhythmic settlement of the hand.

================================================== STATE 18 — SECOND PALM FAN ==================================================

On the following beat:
the wrist reverses direction.
The palm waves horizontally toward the opposite side.
The hand changes direction smoothly.
The forearm maintains the same forward orientation.
The elbow remains stable.
The upper arm remains stable.
The shoulder remains stable.
The movement is generated primarily by:
WRIST → HAND → PALM.
The palm remains open.
The fingers remain naturally aligned.
The hand stays approximately at the same height.
The movement is compact, quick, rhythmic, and controlled.

================================================== STATE 19 — REPEATED PALM-WAVE RHYTHM ==================================================

Continue the established palm-wave pattern:
PALM LEFT → CENTER → PALM RIGHT → CENTER.
Each direction change follows the musical accents in AUDIO 1.
The shoulder remains stable.
The upper arm remains stable.
The elbow remains stable.
The forearm remains generally extended toward the camera.
The wrist and palm perform the visible lateral movement.
The movement is concentrated in the hand.
The palm creates the visual impression of fanning unpleasant air away.
The arm maintains one consistent extended pose while the palm performs repeated small waves.
The movement becomes rhythmic through repetition, not through large arm swings.
Throughout the fanning:
RIGHT FOOT remains bare.
LEFT FOOT remains bare.

================================================== STATE 20 — LOWER-BODY DANCE ENTERS ==================================================

Once the palm-wave rhythm is established:
the lower body begins following AUDIO 1.
The extended arm maintains its stable forward position.
The shoulder remains steady.
The upper arm remains steady.
The elbow remains steady.
The forearm remains generally extended.
Only the wrist, hand, and palm continue their small rhythmic waving motion.
The hand remains the active fanning element.
The first lower-body movement is a small tap with the bare RIGHT FOOT.
The next beat gives a small rhythmic tap to the bare LEFT FOOT.
Both feet remain bare during every tap.
Then the character shifts weight gently from one side to the other.
The movement stays compact and dance-like.
The body remains centered.

================================================== STATE 21 — SIMPLE COMEDIC FOOTWORK ==================================================

The choreography develops through a simple repeating rhythm:
BARE RIGHT FOOT TAP → BARE LEFT FOOT TAP → SMALL WEIGHT SHIFT → PALM WAVE → SETTLE.
Then repeat:
BARE RIGHT FOOT TAP → BARE LEFT FOOT TAP → OPPOSITE WEIGHT SHIFT → PALM WAVE → SETTLE.
The lower body provides the larger rhythmic movement.
The upper arm remains visually calm.
The palm provides the small repetitive fanning gesture.
The costume follows the body naturally.
The RIGHT FOOT remains visibly bare.
The LEFT FOOT remains visibly bare.

================================================== STATE 22 — FASTER RHYTHM ==================================================

As AUDIO 1 becomes more energetic:
the taps become slightly quicker.
The bare RIGHT FOOT performs quick rhythmic taps.
The bare LEFT FOOT follows.
Both feet remain bare during the faster footwork.
The pelvis shifts lightly from side to side.
The torso follows the rhythm with subtle controlled movement.
The extended arm maintains its stable forward position.
The shoulder remains stable.
The upper arm remains stable.
The elbow remains stable.
The forearm remains generally extended.
The wrist and palm continue:
LEFT → RIGHT → LEFT → RIGHT.
The hand provides the visible fan rhythm.
The arm itself remains visually steady.
The mouth-covering hand remains near the face.
The eyes continue showing playful displeasure.
The overall movement becomes more energetic through the lower body and hand, while the main arm structure remains stable.

================================================== STATE 23 — FINAL BUILD ==================================================

Approaching the strongest musical section:
the bare RIGHT FOOT makes a slightly larger rhythmic step.
The bare LEFT FOOT follows.
The character moves back toward the center.
The torso performs a small rhythmic accent.
The extended arm maintains its stable forward position.
The wrist and palm perform larger, more expressive waves.
The upper arm remains steady.
The elbow remains steady.
The forearm remains generally extended.
The hand remains open.
The mouth-covering hand remains near the lower face.
The character gradually gathers the choreography into a centered final stance.
Both feet settle firmly onto the floor.
The final lower-body identity remains:
RIGHT FOOT = BARE LEFT FOOT = BARE.

================================================== STATE 24 — FINAL PALM WAVE ==================================================

On the strongest final audio accent:
the extended arm maintains its forward position.
The wrist performs one decisive lateral movement.
The palm travels clearly from one side to the other.
The hand movement is larger than the earlier palm waves.
The shoulder remains steady.
The upper arm remains steady.
The elbow remains steady.
The forearm remains steady.
The visible movement is concentrated in the wrist, hand, and palm.
The hand settles exactly with the musical accent.
The mouth-covering hand remains near the face.
The eyes look toward the camera.
The eyes deliver the final restrained playful-disgust expression.
Both feet remain clearly differentiated anatomically while both remain barefoot:
RIGHT FOOT remains bare.
LEFT FOOT remains bare.

================================================== STATE 25 — FINAL FREEZE ==================================================

Immediately after the final musical accent:
the complete body settles naturally.
Both feet remain firmly on the floor.
The anatomical RIGHT FOOT remains completely bare.
The anatomical LEFT FOOT remains completely bare.
The mouth-covering hand remains near the lower face.
The other arm remains extended toward the camera.
The upper arm remains steady.
The forearm remains in the established forward position.
The wrist and hand settle naturally after the final palm wave.
The eyes remain expressive.
The face remains visually consistent with IMAGE 1.
The costume remains unchanged.
The character holds the final pose through the end of AUDIO 1.

================================================== PERMANENT HAND ROLES ==================================================

During the sock-removal action:
BOTH HANDS work together to remove the sock.
After the sock is completely removed:
ONE HAND becomes the sock-holding hand.
The sock-holding hand performs:
SOCK HOLD → SOCK INSPECTION → SNIFF → THROW.
The opposite hand becomes the mouth-covering hand.
The mouth-covering hand performs:
FREE HAND → MOVE TO MOUTH → COVER MOUTH AND LOWER FACE → HOLD THAT POSITION.
After the throw:
the sock-hand becomes the air-fan hand.
Its continuous role is:
SOCK-HOLDING HAND → THROWING HAND → EXTENDED STABLE ARM → WRIST-LED PALM WAVE.
The arm maintains its stable forward structure while the wrist and palm provide the fanning movement.

================================================== PERMANENT FOOT CONTINUITY ==================================================

The IMAGE 1 foot arrangement is permanent.
LEFT FOOT:
original supporting foot, completely bare, same anatomical left foot throughout the performance.
RIGHT FOOT:
original raised foot, white sock at the beginning, bare after the sock-removal action, same anatomical right foot throughout the performance.
The permanent foot relationship is:
LEFT FOOT = BARE RIGHT FOOT = WHITE SOCK → BARE FOOT.
Only the RIGHT FOOT changes state during the sock-removal action.
The LEFT FOOT remains bare throughout.
When the RIGHT FOOT reaches the floor, the RIGHT FOOT lands bare beside the LEFT bare foot.
The stable two-foot configuration is:
LEFT BARE FOOT + RIGHT BARE FOOT.
This remains the established lower-body state during the entire dance.

================================================== PERMANENT BODY CONTINUITY ==================================================

The anatomical LEFT FOOT remains the same leg throughout.
The anatomical RIGHT FOOT remains the same leg throughout.
The white sock begins on the RIGHT FOOT.
The white sock is removed.
The RIGHT FOOT becomes bare.
The LEFT FOOT remains bare.
The body transitions from:
RIGHT FOOT RAISED + LEFT FOOT SUPPORTING
to:
RIGHT FOOT BARE ON FLOOR + LEFT FOOT BARE ON FLOOR.
The transition happens naturally through the hips, knees, ankles, and pelvis.
The body remains proportionally consistent.
The character retains the same apparent height and overall physique.

================================================== PERMANENT LOWER-BODY VISUAL STATE ==================================================

The lower-body identity established by IMAGE 1 remains the reference for every later pose.
At the beginning:
LEFT FOOT = BARE RIGHT FOOT = WHITE SOCK.
After removal:
LEFT FOOT = BARE RIGHT FOOT = BARE.
After landing:
LEFT FOOT = BARE RIGHT FOOT = BARE.
During dancing:
LEFT FOOT = BARE RIGHT FOOT = BARE.
At the final frame:
LEFT FOOT = BARE RIGHT FOOT = BARE.
The same anatomical left and right feet remain continuous throughout the complete choreography.

================================================== PERMANENT CAMERA RELATIONSHIP ==================================================

The camera preserves the perspective established by IMAGE 1.
The character remains within the same environment.
The lighting remains consistent.
The overall photographic style remains realistic.
The lens interaction is created by the physical approach of the sock.
At lens contact, the sock becomes a foreground object extremely close to the camera.
The camera perspective remains stable throughout the interaction.
After the sock rebounds, the original scene becomes visible again.
The character continues naturally from the same scene.

================================================== PERMANENT FACIAL PERFORMANCE ==================================================

The face remains elegant and recognizable.
The emotional performance is subtle.
The sequence is:
neutral concentration → curiosity toward sock → short sniff → eyes narrow slightly → brief look toward sock → look toward camera → playful displeasure → controlled comedic dance.
The eyebrows retain their original structure.
The forehead remains relaxed.
The mouth remains visually related to IMAGE 1.
The hand covering the mouth provides the comedy.
The eyes carry the final emotional performance.

================================================== AUDIO 1 — TIMING SYSTEM ==================================================

AUDIO 1 determines the exact rhythm.
Use strong musical accents as transitions between the established physical states.
A useful timing structure is:
ACCENT 1 → begin two-hand sock removal
ACCENT 2 → sock fully removed
ACCENT 3 → look at sock
ACCENT 4 → sock approaches nose
ACCENT 5 → sniff
ACCENT 6 → eye reaction
ACCENT 7 → mouth-covering hand arrives
ACCENT 8 → throw preparation
STRONG ACCENT → sock release
FOLLOWING BEATS → sock flight
IMPACT ACCENT → lens contact
NEXT BEAT → rebound and fall
FOLLOWING BEAT → bare RIGHT FOOT begins descending
NEXT BEAT → bare RIGHT FOOT landing beside bare LEFT FOOT
NEXT BEATS → stable palm wave
NEXT PHRASE → lower-body dance
STRONGER PHRASE → faster palm-wave rhythm
FINAL ACCENT → large final palm wave
END → freeze.
The exact duration of each state follows AUDIO 1 naturally.

================================================== FINAL COMPLETE PHYSICAL SEQUENCE ==================================================

IMAGE 1 POSE → LEFT FOOT SUPPORTING AND BARE → RIGHT FOOT RAISED WITH WHITE SOCK → BOTH HANDS REMOVE THE WHITE SOCK → RIGHT FOOT BECOMES BARE → LEFT FOOT REMAINS BARE → SOCK REMAINS IN ONE HAND → LOOK AT SOCK → SOCK RAISED TOWARD FACE → SHORT SNIFF → SOCK MOVES SLIGHTLY AWAY → EYES REACT → OTHER HAND COVERS MOUTH → SOCK-HOLDING ARM PREPARES → DIRECT THROW TOWARD CAMERA → FULL ARM FOLLOW-THROUGH → SOCK FLIES TOWARD LENS → SOCK LIGHTLY CONTACTS LENS → SOCK REBOUNDS → SOCK FALLS → RIGHT BARE FOOT DESCENDS → RIGHT BARE FOOT LANDS → LEFT FOOT REMAINS BARE → TWO-FOOT BAREFOOT STANCE → THROWING ARM BECOMES STABLE EXTENDED ARM → WRIST AND PALM BEGIN SMALL FAN → PALM LEFT → CENTER → PALM RIGHT → CENTER → BARE RIGHT FOOT TAP → BARE LEFT FOOT TAP → WEIGHT SHIFT → PALM WAVE → REPEAT → FASTER LOWER-BODY RHYTHM → FASTER PALM WAVES → LARGER FINAL PALM WAVE → FINAL FREEZE.
The starting foot arrangement is permanently anchored to IMAGE 1:
LEFT FOOT = SUPPORTING BARE FOOT.
RIGHT FOOT = RAISED FOOT WITH WHITE SOCK.
The sock-removal action is performed naturally by BOTH HANDS as one continuous human behavior.
The sock-removal process itself remains visually simple.
After the removal:
RIGHT FOOT = BARE.
LEFT FOOT = BARE.
After the RIGHT FOOT lands:
RIGHT FOOT = BARE.
LEFT FOOT = BARE.
During the complete dance:
RIGHT FOOT = BARE.
LEFT FOOT = BARE.
The two anatomical feet retain their identities permanently.
After the throw, the extended arm remains structurally stable.
The shoulder, upper arm, elbow, and forearm maintain their established forward position.
The visible fanning movement is produced by the wrist, hand, and palm.
The hand creates the rhythmic lateral wave.
The entire arm remains steady while the palm waves.
IMAGE 1 permanently defines the character.
AUDIO 1 permanently defines the timing and rhythm.
The physical continuity of the character, hands, left foot, right foot, sock, body, costume, camera, and environment remains coherent from beginning to end.



---

## FILE: case-studies/2026-09-editorial-lamp-rhythm-fashion-ab/README.md (https://github.com/LoveRain1997/h3-prompt-journal/blob/HEAD/case-studies/2026-09-editorial-lamp-rhythm-fashion-ab/README.md)

# Case Study 012 — 台灯五章节奏 · 高定时尚大片 A/B 双版（Editorial Lamp-Rhythm Fashion Film, Dark-Luxury vs Cute）

> 一支 15 秒、**完全由音乐时间轴驱动**的高定 editorial 时尚大片，同一套「台灯五章 + 10 个互不重复身体架构 + 离散时尚机位」骨架，做出两个可直接生成的版本：**p83 暗奢·双手猛撩裙摆** 与 **p84 可爱·双手高举后折脚跳跃**。机制不变，只换调性、配器与结尾 Hero。单图锁定人物与房间，正文零外貌；台灯是踩在 downbeat 上的节奏视觉锚。

## Inputs（输入）

- **`<Picture 1>`（单图，唯一视觉权威）**：成年主角 + 完整物理环境——身份 / 脸 / 发型 / 体型 / 肤色、整套服装 / 鞋 / 配饰、**裙摆结构与裙内不透明贴身舞蹈安全裤**、现场台灯及其位置、房间几何 / 家具 / 地板 / 墙面 / 陈设、灯光方向 / 曝光 / 调色 / 对比 / 写实质感。
- **AUDIO**：提示词内自建 `non_diegetic_music`（无外部音频文件），BPM 124 / 4/4，两版仅调性与配器不同。
- 目标：不是常规舞蹈片，而是「一张张会动的时尚照片」拼成的 premium 广告；提示词全英文，正文不描写外貌。
- 两份逐字成品：
  - [`p83-dark-luxury-skirt-throw.txt`](./p83-dark-luxury-skirt-throw.txt) — 暗奢 E minor，结尾 **MASSIVE SKIRT THROW / HERO BULLET-TIME**
  - [`p84-cute-backfolded-jump.txt`](./p84-cute-backfolded-jump.txt) — 可爱 A major（NEW CUTE VERSION），结尾 **TWO-HAND HIGH JUMP / BACK-FOLDED FEET HERO**

## The problem（问题：要换调性，却不能推翻已经跑通的节奏骨架）

p83 暗奢版已经把「音乐—灯光—动作—机位」咬合得很死，但成片气质偏性感、结尾靠双手撩裙摆露安全裤。想要一支**更轻、更可爱、更贵气俏皮**的版本时，最容易犯的错是从头重写——结果把已经验证有效的灯点、moment 边界、相机纪律全部弄丢。

真正的需求是一次**受控变奏（controlled variation）**：保留可复用的机制（Creative DNA），只替换表层的情绪、音色和结尾那个最大动作。这个案例展示了如何让两版共享同一副骨架、只在三个指定层面分叉，从而低成本产出气质迥异但节奏同样精准的成片。

## The breakthrough（突破：固定骨架，只在三层分叉）

两版**逐字相同**的骨架：

1. **单图全锁、零外貌文字**：人物、服装、台灯、房间一律由 `<Picture 1>` 承担，配一大段 `Preserve exactly …` 与 `Do not redesign / relocate / add / remove`；正文 token 全部投给时间轴、身体力学与机位。
2. **台灯 = 踩 downbeat 的节奏视觉锚**：五次开关严格落在小节线，无渐变 / 闪烁 / 变色 / 新光源，房间物理上保持同一间。
3. **10 个 MOMENT 身体架构互不重复**：每个动作都走 `INITIATION → TRAVEL → ACCENT → BRAKE → RELEASE`，动作之间不许回中性站姿、不许重复 contrapposto / 摆胯 / 撩发 / 走向镜头 / 原地转。
4. **离散时尚机位（非一条连续运镜）**：压缩长焦、中肖像、中全景、短跟拍、rack focus、肩部遮挡、发丝擦镜、短 arc、受控 ROLL、hard brake、末段 rapid pullback。
5. **优先级链**：`BGM 结构 → 音乐 accent → 身体发起 → 身体力学 → 独特 pose 架构 → 相机响应 → 选择性 ROLL → 相机 brake → 台灯状态 → 剪辑过渡 → 下一个音乐事件`。音乐永远是第一驱动。

只在以下三层分叉（见对照表）：**音乐调性与配器、中段若干下身动作、结尾 MOMENT 10 的 Hero 机制**。

## 两版差异对照（A/B）

| 维度 | p83 暗奢版 | p84 可爱版 |
|---|---|---|
| 定位 | sensual editorial attitude film（性感克制） | cute / elegant / mischievous attitude film（俏皮贵气） |
| Genre | dark luxury high-fashion electro | playful luxury electro-pop |
| 调式 / Key | **E minor** | **A major** |
| 鼓与贝斯 | tight gated kick、rounded sub-bass | tight but **soft** kick、rounded **playful** bass |
| 主音色 | dry metallic plucks、percussive clicks、filtered rising noise | bright **mallet plucks、小铃铛 bell、sparkling synth、pizzicato 电子音、tiny claps、playful riser** |
| 明确排除 | lush pads、vocals、lyrics | 暗黑 drone、重 cinematic bass、工业打击、恐怖质感、pads、**vocals/lyrics** |
| MOMENT 05 | **LEG SWEEP**：支撑脚站定，另一腿由胯带动向后划半圆，裙摆延迟拖出弧线 | **LATERAL STEP / TOE POINT**：小侧步、重心侧移、脚尖干净点地（明确不是划腿 / 踢 / 转 / 重复下蹲） |
| MOMENT 06 | ARM / BACK RELEASE：一臂向后上延展、肩背对立轴 | DIAGONAL REACH / **LIGHT BOUNCE**：脚踝膝盖小弹（明文 NOT a jump，脚不离地） |
| MOMENT 07 | KNEE DRIVE / CLOTHING SNAP：单膝向前带过中线、裙摆 fabric snap | QUICK KNEE TAP / SHOULDER ACCENT：紧凑小抬膝 + 双肩前后 sharp accent |
| MOMENT 09（蓄力） | **SIDE-FACING SKIRT PRELOAD**：侧三-quarter，双手握住裙摆两侧、屈胯蓄力、裙不开 | **JUMP PRELOAD / ARM SWING**：双脚站定屈膝、双臂向后摆加载、裙保持自然闭合、手不碰裙 |
| **MOMENT 10（结尾 Hero）** | **MASSIVE SKIRT THROW**：双手在 impact 同时向外猛甩裙摆 → fabric whip → 裙摆最大绽放 → 稳定身体为中轴、裙摆惯性继续 → bullet-time → 安全裤在撩裙峰值清晰露出 | **TWO-HAND HIGH JUMP / BACK-FOLDED FEET**：蹬地双腿起跳 → 双手在头顶高举成 V → **双脚后折并拢成一个紧凑整体**（不分叉、不踢镜头）→ 跳跃惯性自然扬起裙摆、安全裤清晰可见 → 空中 bullet-time → 落地英雄站姿 |
| 安全裤露出方式 | 由「手甩裙摆」造成，强调 forceful throw、非慢开、非手拉暴露 | 由「向上跳跃的物理惯性」造成，强调不手拉裙、不制造透明 / 走光 |
| 收尾画面 | 大开放裙摆 campaign still，硬锁 | 落地后自然 settle 的 premium campaign 照片感，硬锁 |

> 两版 MOMENT 01–04、08 与所有灯点、相机 ROLL 分布、时间边界完全一致；分歧集中在 05/06/07 的下身质感与 09/10 的结尾机制。

## 台灯五章与精确灯点（两版共用，可复算）

BPM 124 → 1 beat = 60/124 = **0.484s**，1 bar（4 拍）= **1.936s**；全片 = **7 个完整小节 + 3 拍结尾** = 7×1.9355 + 3×0.4839 = **15.000s**。

| 时间 | 小节位置 | 台灯 | 章节 |
|---|---|---|---|
| 0.000 | bar 0 | **ON** | A 建立身体节奏（0.000–3.871） |
| 3.871 | bar 2 | **OFF** | B 暗场 / 图形剪影（3.871–7.742） |
| 7.742 | bar 4 | **ON** | C 回亮 / 能量更新（7.742–11.613） |
| 11.613 | bar 6 | **OFF** | D 悬停期待 temporal vacuum（11.613–13.548） |
| 13.548 | bar 7 | **ON** | E 最终 Hero impact（13.548–15.000，3 拍尾） |

> 灯点严格每 2 小节一次（3.871 / 7.742 / 11.613），最终 impact 落在第 7 小节头 13.548。每次开关都与音乐 downbeat 精确同帧，并配一声克制的实体开关 `switch click`（写在 `overall_soundscape`，踩这四个点）。OFF 不是黑场，用房间残余照明自然呈现。

## 10-MOMENT 骨架（两版共用时间边界，★＝两版分歧）

| Moment | 时间 | 身体 / 镜头要点 |
|---|---|---|
| 01 | 0.000–1.300 | 反向重心转移、脚跟轻抬、一臂斜后；中全景三-quarter 侧，短跟拍 + 极短**顺时针 ROLL**，~1.0s 双 hard brake |
| 02 | 1.300–2.600 | 肘起在头侧构图、头反向轻歪；侧颜→三-quarter 前，焦段压缩 + rack focus 前臂→眼，**无 ROLL**，时尚肖像 |
| 03 | 2.600–3.871 | **纯侧面下蹲**、一腿斜伸长腿线、躯干挺直不朝镜头；随胯下降，最低点一次**逆时针 ROLL** 后回正；3.871 灯 OFF |
| 04 | 3.871–5.806 | 支撑脚起身（胯先起）、转**后三-quarter**、回头看镜头；肩部遮挡→rack focus 肩→眼，灯保持 OFF |
| 05 ★ | 5.806–7.742 | p83 后腿**划半圆** / p84 **侧步点地**；toe / sweep accent 上短**顺时针 ROLL**；7.742 灯 ON |
| 06 ★ | 7.742–9.677 | p83 手臂向后上 release / p84 斜上够 + 轻弹；手→前臂→肩→颈→脸的垂直 rise，无 ROLL，直视镜头 |
| 07 ★ | 9.677–11.613 | p83 单膝前带 + 裙摆 snap / p84 小抬膝 + 双肩 accent；下半身跟后快速上摇到脸，最锐 accent 一次短**顺时针 ROLL**；10.645 sub hit、11.129 hi-hat burst；11.613 灯 OFF |
| 08 | 11.613–12.250 | **Temporal vacuum 静止**：音乐抽稀，几乎定格、仅呼吸 / 发丝 / 配饰微动；相机水平、仅极小前向 parallax，无 ROLL / zoom / 慢动作 |
| 09 ★ | 12.250–13.548 | p83 双手抓裙两侧**撩裙 preload** / p84 屈膝摆臂**跳跃 preload**；~13.300 riser 张力最大、定格蓄力；13.548 灯 ON，Moment 10 立刻起 |
| 10 ★ | 13.548–15.0 | 音乐 hit + 灯 ON + 最大动作三合一；rapid pullback 微升 + 短 arc 给全整空中 / 绽放轮廓，峰值 **bullet-time feel（非字面慢动作）**；~13.9 动作到顶开始下落 / 收手，~14.1 落地 / 硬锁，14.1–15.0 近乎全静的 campaign still |

## 相机系统纪律（两版共用）

- **离散构图而非一条连续运动**：每个相机状态由不同身体动作触发，靠快速 editorial 过渡连接。
- **ROLL 只在 01 / 03 / 05 / 07 四个点**，且都是绕光轴的极短、受控旋转，绑死具体动作 accent；**每次 roll 后地平线必须回正**；不连续转、不 360°、不绕房间 orbit、不长期歪 horizon。Moment 08、09 保持水平。
- **Hard brake**：每个 pose 到位时相机与身体同时硬刹，形成短暂时尚静止。
- **Bullet-time feel** 只出现在结尾峰值：主观时间密度被拉长，但人物是稳定中轴、头发 / 裙摆 / 衣缘保留真实惯性；明文「不是字面慢动作、不机械冻结」。
- 机位关系反复在 profile / 后三-quarter / 前三-quarter / 局部 / 腿位中全景 / 最终全身之间切换；禁用顶拍、地板广角、鱼眼、无尽 zoom、通用慢推。

## 合规 / 安全设计（值得复用的写法）

- 主角明确为**成年专业时尚表演者**，全程着装（fully clothed）；裙内**不透明贴身舞蹈安全裤**被定义为造型的固定服装层，全程「不消失 / 不变透明 / 不变形 / 不移位 / 不被替换」。
- 结尾露安全裤被反复框定为**完整着装下的时尚造型 reveal，绝非裸露或内衣暴露**；p84 更进一步改为由跳跃惯性自然扬起裙摆，明文不许手拉裙、不许透明面料、不许 wardrobe malfunction。
- 躯干一旦降低，必须是**纯侧面 / 后三-quarter / 强侧三-quarter**，禁止胸部朝镜头的正面下俯、禁止 cleavage 特写与胸部构图；可爱版另加「不幼态、不卡通、不夸张 kawaii」。

## 48-hour takeaway（可复用清单）

1. **A/B 改版只换表层、不重写机制**：把跑通的时间轴 / 灯点 / 机位 / 优先级链原样保留，分叉点集中在调性、配器与结尾 Hero，改完两版节奏天然对齐、可直接对比成片。
2. **让一个道具成为节拍器**：台灯五章把「灯光状态」写进剪辑语言，开关全部钉在 downbeat（本例每 2 小节），并配同点实体音效——音乐、灯光、动作三合一，成片像被音乐剪辑出来。
3. **先算音乐网格再写动作**：BPM→beat/bar 时长→小节线灯点→「7 小节 + 3 拍尾 = 15.0s」，所有 moment 边界与 sub hit（10.645 / 11.129）都挂在网格上。
4. **十个 moment 强制身体架构不重样**：每个动作写明与前一个的区别，并显式排除重复（同款蹲 / 侧步 / 跳互不相似），配 `INITIATION→TRAVEL→ACCENT→BRAKE→RELEASE`。
5. **ROLL 可数、必回正**：只在少数具名时刻出现、绑死动作、用后回正；末段 bullet-time 是「时间密度」不是慢动作。
6. **敏感露出用服装层 + 朝向规则提前锁死**：把安全裤写成不透明造型层、把俯身朝向限定为侧面，结尾 reveal 即合规。

## Assets

| 文件 | 作用 |
|---|---|
| [`p83-dark-luxury-skirt-throw.txt`](./p83-dark-luxury-skirt-throw.txt) | 暗奢 E minor 版完整成品（结尾双手撩裙 Hero），逐字 |
| [`p84-cute-backfolded-jump.txt`](./p84-cute-backfolded-jump.txt) | 可爱 A major 版完整成品（结尾双手高举后折脚跳 Hero），逐字 |

> 两份成品与下载源文件 SHA-256 一致（字节级逐字保存）。生成时需自行提供 `<Picture 1>`（人物 + 含台灯的室内环境参考图）；仓库不含受版权 / 肖像约束的图像、音频与成片。

## Tags

`#H3` `#prompt-engineering` `#high-fashion` `#editorial-film` `#fashion-commercial` `#image-locked` `#zero-appearance-description` `#music-driven` `#beat-sync` `#lamp-rhythm` `#lighting-as-editing` `#discrete-compositions` `#camera-state-machine` `#controlled-roll` `#hard-brake` `#bullet-time` `#hero-frame` `#ab-variation` `#mechanism-migration` `#safety-shorts`


---

## FILE: case-studies/2026-09-purple-veil-dance-seven-segment/README.md (https://github.com/LoveRain1997/h3-prompt-journal/blob/HEAD/case-studies/2026-09-purple-veil-dance-seven-segment/README.md)

# Case Study 014 — 紫纱亭舞 · 七段连续生成（Purple-Veil Pavilion Dance · Seven Segments）

> 一支约 87.7 秒的古风 sensual 古典舞（BPM 149.5，19 句唱词），切成 7 段分别生成再拼接。**PICTURE 1 锁人物/服装/环境、分段 AUDIO 1 锁节拍网格与歌词 onset**；每段以**局部特写开场**接住上段结尾的实物动量；七个段落**主动力源互不重复**；一件紫色服装在冷调环境里全程保持"紫而不灰"；**面纱 87.7 秒永不摘除**，"挑面纱"歌词全部转写为几何遮蔽。

## Inputs（输入）

- **PICTURE 1**：唯一视觉权威——人物身份、五官、发型、体型、全套服装配饰鞋、环境、背景、灯光方向、写实质感（三视图，含锁定服装细节）。
- **AUDIO 1（分段提供）**：唯一时间/节奏权威——BPM 149.5、beat 0.4013s、bar 1.6054s、phrase 3.2107s、全局首拍 0.139s；歌词 onset 逐段列死。信号不复制、无 lip-sync，成片音乐后期统一铺回。
- **无视频参考**：参考舞蹈视频的姿势已全部转写为文字编舞（借姿势不借画面，避免coser/展会场景风格渗透）。
- 词作主题：紫衣妙音门主与"我"（面纱/敬酒/跪疤/故人来）。最终七段可直接粘贴的提示词全文：[`prompt.md`](./prompt.md)。

## The problem（四个实际踩过的坑）

1. **原地瞎扭**：初版 SEG05 把主动作写成质量词堆叠（"waist spirals + torso waves + counterrotation + arcs"同时发生），没有任何分时序的具体动作——模型拿到"什么都要"就退化成原地循环扭动；
2. **接不上段尾**：初版 SEG07 开场特写怼在**腰部**，而 SEG06 递过来的是"斜上方扬起的手臂"——交接物不在画面里，动量视觉上被剪掉，只好重新编；
3. **紫衣发灰**：初版配色（冷靛蓝底 + black-green shadows + 低对比）三件事叠加，紫色服装在低照度冷底上必然塌成脏灰（暗视觉下蓝紫端辨色力最先失效）；
4. **弓步变跪地**："deep lunge / pelvis lowering toward one supporting knee" 不写死脚部约束，模型把侧弓步做成单膝跪地。

## The breakthrough（六件事写死）

### 1. 动力源轮换：每段只有一个主动力源

七个段落的 PRIMARY driver 互不重复，胯只在其该在的段当主角：

| 段 | 开场特写 | 主动力源 | 核心圆 |
|---|---|---|---|
| 01 | 脚部 | 腰轴（平圆）+ 步法 | 腰平圆 |
| 02 | 袖腕 | 脊柱提沉 + 袖线 | 袖立圆（小） |
| 03 | 双手覆面 | 骨盆（本段就该胯） | 髋八字圆 |
| 04 | 膝部 | 膝与重心 | 腰螺旋 |
| 05 | 手部 | 肩臂主线（AAB-C 结构） | 臂立圆（大） |
| 06 | 纱后侧影 | 呼吸与顿挫 | — |
| 07 | 腰袖 | 三圆联动全开 | 三圆 |

### 2. 局部特写转场链：每段开场特写 = 上段结尾的实物落点

脚 → 袖 → 手 → 膝 → 手 → 纱影 → 腰袖。相邻两段之间，上段结尾动作的**末端物件**就是下段第一个特写的**主体**——每段独立生成后按清单硬切即是 match cut，不需要转场特效。交接物在 subject/transition 里**点名到具体肢体与高度**（如"斜上方头部高度的过帧袖子"），不写抽象的"延续动量"。

### 3. 紫衣显色三定律

- 暖金 key（灯笼动机，upstage-left）+ 琥珀反光裹紫衣——紫的近互补是金，并置即显色；
- 冷靛蓝只留在**背景**且比主体暗 1–2 档——冷是背景的事，暖是主体的事；
- 主体阴影一律**暖褐带织物质感**，明文禁 black-green crush；中调保护、moderate contrast。
另加段内补丁：SEG03 弓步段给地板暖反光（低位不自阴影里消失）、SEG06 纱幕边缘暖 rim（剪影可读）、SEG07 全片最亮 key（终帧最紫）。

### 4. 节拍网格全局锚定，歌词 onset 用实测

段内所有时间戳从**歌曲全局首拍 0.139s + k×0.4013** 复算（段内锚点各不相同，如 seg03=0.368、seg05=0.234），phrase accent 同理全局可复算；歌词 onset 逐句取自切割 manifest 实测值，不估算。早期版本曾出现累计 0.03s 漂移与歌词错位 4 拍的事故，均因局部取整/估摸歌词位置。

### 5. 面纱锁 + 几何式遮蔽

面纱是锁定服装的一部分，87.7 秒不摘不掀不透；"挑面纱"类歌词一律转写为**几何遮蔽变化**（腕错位、抬头、转身、手部几何改变），回眸类姿势转写为"纱后剪影"（颊线/头角示意微笑，五官永不可见）。SEG06 明文禁止眼部特写，最近镜头只到"纱后侧影"。

### 6. 具体动作时序 > 质量词堆叠（SEG07 教训）

高能段落必须写成**有时序的具体编舞**（四段式：继承圆弧 → 带位移步法 → 双翅大弧+旋转 → 收势 Hero），每个机制踩在网格 downbeat 上；允许 2–3 步 pavilion 内小位移破"原地感"。质量词（spiral/wave/counterrotation）只能作为单个动作的修饰，不能作为段落大纲。

## The named language（命名）

> **Driver-Rotation Seven-Segment Veil Dance with Detail-Cut Transition Chain（动力源轮换 · 局部特写转场链 · 七段纱舞）**

「Driver-Rotation」指七段主动力源互不重复；「Detail-Cut Transition Chain」指段间用实物特写交接（上段结尾的物件 = 下段开场的特写主体），独立生成也能硬切成连续长镜。

## 失败 → 规则 → 反测

| 失败 | 可观察原因 | 规则 | 反测 |
|---|---|---|---|
| SEG05 原地循环扭动 | 质量词堆叠无时序；骨盆连续 6 段主导 | 每段一个主动力源 + 具体动作时序（AAB-C） | 重生成后出现位移步与袖弧，不再循环 |
| SEG07 接不住 SEG06 | 开场特写主体（腰）≠ 交接物（斜扬手臂） | 交接物点名到肢体+高度，开场特写必须包含它 | 首帧即见过帧袖手，动量连续 |
| SEG03 弓步生成单膝跪地 | 未写死脚部约束，"lowering toward knee"歧义 | "pointed foot lightly contacting the floor / both feet stay on the floor" 明文 | 弓步保持，膝不触地 |
| 紫衣整体发灰 | 冷底+黑绿阴影+低对比三叠加 | 紫衣显色三定律 + 正向描述"reads as true saturated purple" | 暖金 key 下紫衣显色 |
| 歌词标注错位 4 拍 / 网格漂移 0.03s | 估摸歌词位置；段内独立取整 | 全局首拍复算网格 + manifest 实测 onset | 逐点核验全部落格 |

## 段间衔接速查

```
SEG01 结尾 转体中袖过外框      → SEG02 开场 袖腕特写（同一次转腕）
SEG02 结尾 袖上扬过镜头        → SEG03 开场 双手合掌覆面特写
SEG03 结尾 下坐、重心动膝      → SEG04 开场 膝部特写（同一只承重膝）
SEG04 结尾 手伸向镜头          → SEG05 开场 手部特写（同一只手继续转腕）
SEG05 结尾 头先转、朝向新方向   → SEG06 开场 纱后侧影特写（同一朝向）
SEG06 结尾 斜扬臂、袖过帧      → SEG07 开场 袖手特写（同一只手完成圆弧）
```

## 提交工作流

1. 每段独立生成：附 **PICTURE 1**（三视图）+ **AUDIO 1**（该段音频切片）；
2. 先只跑 SEG03 第一个镜头试紫衣显色（"紫而不灰"），确认后再批量；
3. 七段按切割清单时间轴铺回原曲，切点即特写匹配剪辑。

## Anchor check（T8 机制库纪律，提交前逐条过）

参照 T8 机制包的强制锚点 + 静默自检流程（anchor coverage / continuity / timing / physical causality / anti-copy distance）：

- **anchor coverage**：每段六锚齐备——开场局部特写（含上段交接物）、单一主动力源、VEIL_LOCK、节拍网格锚点、歌词 onset、紫衣显色规范；
- **continuity**：七段交接物逐一点名到肢体+高度（见速查表），无"延续动量"式抽象交接；
- **timing**：全部时间戳自全局首拍复算，phrase accent 与歌词 onset 无漂移；
- **physical causality**：动作链保持"脚→重心→胯→腰→肩→臂→腕→袖"因果顺序，袖/裙只跟随不独立发力；
- **anti-copy distance**：参考视频仅借姿势（文字转写、幅度/时序已改），不附视频、不复用其人物/舞台/镜头节奏。


---

## FILE: case-studies/2026-09-purple-veil-dance-seven-segment/prompt.md (https://github.com/LoveRain1997/h3-prompt-journal/blob/HEAD/case-studies/2026-09-purple-veil-dance-seven-segment/prompt.md)

# 紫纱亭舞 · 七段连续 H3 提示词（完整终版）

> 用法：每段独立生成，附 `PICTURE 1`（人物三视图，唯一身份/服装/环境权威）与 `AUDIO 1`（该段音频切片）。
> 每段自包含，可直接整段粘贴。段落顺序与切点见同目录 README 的「段间衔接速查」。
> 全局节拍：BPM 149.5，beat 0.4013s，bar 1.6054s，phrase 3.2107s，全局首拍 0.139s。

---

# SEGMENT 01｜00.000–11.250（脚部特写开场 · 腰轴平圆）

```text
subject_definitions:

<Subject 1> is the exact same adult live-action person from <Picture 1>.
<Picture 1> is the ONLY identity and costume reference.
Preserve identity, face, hairstyle, body proportions, outfit, accessories and footwear exactly.
A translucent dark-gauze face veil is part of the locked costume: worn from the first frame, NEVER removed, lifted or made transparent; the face behind it is never clearly visible.

<Subject 2> is the exact same ancient Chinese elevated ceremonial pavilion / banquet hall as <Picture 1>: dark wood architecture, dark flooring, carved railings, layered translucent gauze curtains, low table with wine and tea vessels, restrained practical lanterns, elevated viewing area.
Always heavily defocused. No people, no location/time/weather change.

<Audio 1> is the song track, referenced only for rhythm: BPM 149.5, beat 0.4013s, bar 1.6054s, phrase 3.2107s, first downbeat 0.139s.
Lyric onsets in this segment: 1.483 / 4.800 / 8.000.
No signal copy, no lip-sync.

PERFORMANCE:

Adult sensual ancient-Chinese dance, mature, elegant, non-explicit.
Primary driver: the WAIST as axis — slow horizontal-circle waist rotation — with precise footwork underneath and hips answering as grounded accents.
Cool composed upper body over an active lower body.
One continuous dance phrase; never walking, never static posing.

LYRIC_RELATION:

Lyrics describe beauty and the mysterious 妙音门主.
Do not act them out.
Allure comes from silhouette, head angle and waist rhythm behind the veil.

VEIL_LOCK:

The veil covers the face in every frame.
Hands never remove, lift or part it.
Expression is conveyed only through head angle and silhouette.

CAMERA:

SHOT 01 opens on a tight FOOT close-up.
Then motivated cuts: knee-to-waist reveal → medium-close three-quarter → fuller body.
No orbit, no zoom breathing.

COLOR_AND_LENS:

The dancer's costume is a rich violet-purple and it must read as true, saturated purple in every frame — never gray, never black, never blue-shifted.
Lighting: a warm golden key from the practical lanterns falls on the dancer from upstage-left, wrapping the purple fabric with amber reflections; a soft warm-neutral fill protects the midtones so the purple keeps its hue, saturation and fabric detail.
The background pavilion stays in cooler indigo and blue-gray, one to two stops darker than the subject, so the warm-lit purple figure separates cleanly from the cold set.
Shadows on the subject are warm brown with visible fabric texture, never black-green crush.
Overall exposure is moderate-contrast and luminous: lifted midtones, soft highlight roll-off on lantern glow, subtle bloom and halation, fine film grain, medium depth of field.
Skin stays warm and natural; deep vermilion appears only as tiny practical accents.
Keep the light DIRECTION consistent with <Picture 1>, but raise the key intensity on the subject as specified above — the subject must never sink into the background's darkness.

AUDIO:

<Audio 1> is the sole temporal driver; follow the beat grid and lyric onsets above.

integrated_multimodal_description:

00.000–0.401
SHOT 01 — FOOT CLOSE-UP.
Start on the lower legs and bare feet on the dark floor: one foot presses down, the opposite foot traces a compact pointed pathway, the purple hem swaying above with a warm golden sheen.
The dance is already active at frame one.

00.401–1.483
Quick alternating weight transfers; supporting knee softens; the pelvis begins following the feet; the torso settles upright with fingertips resting at the front waist, head lowered behind the veil.

1.483–4.800
Lyric: “紫衣轻掩着一身风华”.
SHOT 02 — KNEE-TO-WAIST REVEAL.
The waist draws one slow horizontal circle, hips answering each crossing with a compact lateral accent; the veil edge sways with the breath, rimmed in warm gold against the cool blurred pavilion.

4.800–8.000
Lyric: “妙音门主她清冷如画”.
SHOT 03 — MEDIUM-CLOSE THREE-QUARTER: the body opens into a graceful three-quarter silhouette, the waist circle continuing underneath a composed chest, veiled head slightly tilted.

8.000–9.771
Lyric: “一缕真音堪破阵中变化”.
The chain compresses: compact step → lateral hip accent → waist spiral → shoulder catch-up → sleeve release.
The veiled head stays lowered, then flicks toward the coming turn.

9.771–11.250
SHOT 04 — FULLER BODY.
A rotational phrase: pelvis turns first, waist follows, shoulders then head last; on the strong beat at 10.975 the sleeve gains momentum.
No freeze.

TRANSITION_TO_SEGMENT_02:

End mid-rotation with the sleeve still traveling; SEGMENT 02 opens on the same wrist and sleeve.
```

---

# SEGMENT 02｜11.250–24.250（袖腕特写开场 · 扩张与收缩）

```text
subject_definitions:

<Subject 1> is the exact same adult live-action person from <Picture 1>, same locked costume; the translucent dark-gauze face veil is worn in every frame and is never removed, lifted or parted; the face is never clearly visible.

<Subject 2> is the exact same ancient Chinese elevated ceremonial pavilion / banquet hall as <Picture 1>, always heavily defocused; no people, no location/time/weather change.

<Audio 1> is the song track, referenced only for rhythm: beat 0.4013s, bar 1.6054s, phrase 3.2107s.
Lyric onsets in this segment: 0.000 / 3.233 / 6.283 / 9.750.
No signal copy, no lip-sync.

PERFORMANCE:

Adult sensual ancient-Chinese dance, mature, elegant, non-explicit.
Primary driver: the SPINE and sleeve lines — lift and fold; hips stay rhythmically active underneath.
First half expands (larger spatial occupation, lifted sternum); second half contracts (lower center, narrower carriage).
One continuous dance.

LYRIC_RELATION:

Do not act out 万人仰慕 or 听传话.
Expansion and compression of the carriage carry the meaning.

VEIL_LOCK:

The veil covers the face in every frame; the choreography never touches or lifts it.

CAMERA:

SHOT 01 opens on a tight SLEEVE / WRIST close-up.
Then: expanded body shot → lowered compressed body shot.
Restrained cuts.

COLOR_AND_LENS:

The dancer's costume is a rich violet-purple and it must read as true, saturated purple in every frame — never gray, never black, never blue-shifted.
Lighting: a warm golden key from the practical lanterns falls on the dancer from upstage-left, wrapping the purple fabric with amber reflections; a soft warm-neutral fill protects the midtones so the purple keeps its hue, saturation and fabric detail.
The background pavilion stays in cooler indigo and blue-gray, one to two stops darker than the subject, so the warm-lit purple figure separates cleanly from the cold set.
Shadows on the subject are warm brown with visible fabric texture, never black-green crush.
Overall exposure is moderate-contrast and luminous: lifted midtones, soft highlight roll-off on lantern glow, subtle bloom and halation, fine film grain, medium depth of field.
Skin stays warm and natural; deep vermilion appears only as tiny practical accents.
Keep the light DIRECTION consistent with <Picture 1>, but raise the key intensity on the subject as specified above — the subject must never sink into the background's darkness.

AUDIO:

<Audio 1> drives all movement; follow the lyric onsets above.

integrated_multimodal_description:

00.000–0.528
SHOT 01 — SLEEVE / WRIST CLOSE-UP.
Continue the exact sleeve momentum from SEGMENT 01: the wrist turns, the elbow follows, translucent fabric sweeps the upper frame with warm gold on its edge.

0.528–3.233
Expand into a medium-close shot; the pelvis carries the rotation into a lateral hip phrase; do not reset the feet.

3.233–6.283
Lyric: “高台之上万人仰慕她”.
SHOT 02 — EXPANDED BODY.
A smaller transposed wing-opening: both arms sweep outward and downward like opening feather wings, sternum lifts, spatial width grows, hip arcs enlarge on confident weight transfers.

6.283–9.750
Lyric: “转眼之间兽车遮了纱”, then “他口称仙子言笑皆风雅”: a sleeve crosses through the frame and the torso rotates behind the fabric, then returns to an expansive line with one smooth waist spiral and one strong lateral hip accent.

9.750–13.000
Lyric: “我作随侍低眉听传话”.
SHOT 03 — COMPRESSED BODY.
The hands draw inward, sliding past the chin line and sternum before folding toward the torso; the chin lowers behind the veil; shoulders narrow; height drops; the pelvis keeps dancing in compact transfers.

11.364–13.000
A new rotational phrase begins; on the strong beat at 12.568 one sleeve rises toward the camera and moves across the lens area.

TRANSITION_TO_SEGMENT_03:

SEGMENT 03 opens on the hands as they press together over the veil, continuing from this sleeve motion.
```

---

# SEGMENT 03｜24.250–37.550（双手覆面特写开场 · 骨盆主段 · 官方权威版）

```text
subject_definitions:

<Subject 1> is the exact same adult live-action person from <Picture 1>.
<Picture 1> is the ONLY visual identity and appearance reference.
Preserve exact identity, face, hairstyle, body proportions, outfit, accessories, footwear and visible appearance from <Picture 1>.
Do not redesign, reinterpret, beautify, replace or restyle the subject.
Do not invent a new veil, new costume or new body features.
Only one visible person exists.

PERFORMANCE:

Adult sensual ancient-Chinese dance, mature, elegant, alluring and non-explicit.
The PRIMARY DANCE FEATURE is continuous, pronounced rhythmic hip movement.
The pelvis is the main movement driver.
The strongest choreography occurs in the low side-lunge-plié.
Use:
side lunge → weight transfer → lateral hip accent → counter-hip accent → compact pelvic figure-eight → waist follow-through.
The lower body must stay active continuously.
The choreography must remain a dance phrase rather than a sequence of literal poses.

OPENING_POSE:

Start in a very low SIDE LUNGE-PLIÉ.
One leg extends diagonally outward with the foot pointed and lightly contacting the floor.
The opposite leg is deeply bent and carries most of the weight.
The pelvis is low and displaced toward the extended-leg side.
The torso is upright with a slight forward inclination.
The head is slightly lowered.
Both palms are pressed together vertically directly in front of the face.
Both palms COMPLETELY COVER THE FACE.
Both wrists overlap in front of the face.
Both elbows open outward at approximately shoulder height.
The face is completely hidden.
This exact hand-over-face structure is maintained at the beginning of the segment.

LYRIC_RELATION:

“面纱之下低眉不语啊” is not literal acting.
The face remains hidden behind both hands while the pelvis continues a clear rhythmic dance.
This creates the core visual contradiction:
quiet concealed face + active sensual lower body.

“旁人敬酒我先斟满它” changes the arm pathway but does not turn the dance into serving mime.

“他挑面纱笑说不必遮” does NOT mean removing a veil.
Do not take off or discard any veil.
Instead, reveal information through choreography:
wrist offset, finger repositioning, head lift, torso turn and changing hand geometry.
The hands can remain near the face.
The face may become briefly more visible from a new angle, but the action must feel like a shifting concealment rather than a literal unveiling.

ENVIRONMENT_LOCK:

Exactly the same ancient Chinese elevated ceremonial music pavilion / banquet hall.
Same architecture, dark floor, carved railing, layered gauze curtains, low table, wine and tea vessels, restrained practical lights and elevated carriage/viewing area.
No visible additional people.
No location change.
No new background.
No weather or time change.

COLOR_AND_LENS:

The dancer's costume is a rich violet-purple and it must read as true, saturated purple in every frame — never gray, never black, never blue-shifted.
Lighting: a warm golden key from the practical lanterns falls on the dancer from upstage-left, wrapping the purple fabric with amber reflections; a soft warm-neutral fill protects the midtones so the purple keeps its hue, saturation and fabric detail.
The background pavilion stays in cooler indigo and blue-gray, one to two stops darker than the subject, so the warm-lit purple figure separates cleanly from the cold set.
Shadows on the subject are warm brown with visible fabric texture, never black-green crush.
Overall exposure is moderate-contrast and luminous: lifted midtones, soft highlight roll-off on lantern glow, subtle bloom and halation, fine grain, medium depth of field.
Skin stays warm and natural; deep vermilion appears only as tiny practical accents.
Keep the light DIRECTION consistent with <Picture 1>, but raise the key intensity on the subject as specified above — the subject must never sink into the background's darkness.
A warm bounce from the polished floor lifts the purple fabric inside the lunge, so the low pose never loses separation from the floor.

CAMERA_AND_SHOT_DESIGN:

Use motivated cuts.

SHOT 01:
tight lower-body detail;
side lunge-plié, pointed extended leg, bent supporting leg and active hip rhythm.

SHOT 02:
tight hands-and-face composition;
both palms completely cover the face.

SHOT 03:
medium-close lower-torso composition;
hands near waist while hip rhythm remains the visual focus.

SHOT 04:
full-body vertical composition;
rise out of the side lunge and expand.

SHOT 05:
three-quarter medium-close composition;
show the incomplete reveal through hand geometry and head angle.

No random orbit, no decorative camera movement.
Camera changes only when they reveal new movement information.

AUDIO:

<Audio 1> is the sole temporal driver.
BPM 149.5.
Beat = 0.4013 sec.
Phrase = 3.2107 sec.

integrated_multimodal_description:

00.000–0.371
SHOT 01.
Start directly in the low side-lunge-plié inherited from SEGMENT 02.
One leg extended diagonally outward and pointed.
Opposite knee deeply bent.
Pelvis low.
Both palms completely cover the face.

00.371–0.772
Right hip moves outward through the floor-driven weight transfer.
Supporting knee softens.

0.772–1.173
Reverse the pelvic direction.
Left hip moves outward.
Maintain the asymmetrical side-lunge structure.

1.173–1.575
Repeat the alternating hip rhythm.
Waist follows the pelvis slightly after each accent.

1.575
Phrase accent:
one broader lateral pelvic sweep.

1.575–3.180
SHOT 02.
Lyric: “面纱之下低眉不语啊”.
Both palms remain completely covering the face.
Elbows stay open.
Head remains low.
Under the concealed upper body, the hips continue:
right → center → left → center.
Use clear, repeated sensual hip accents.

3.180
Downbeat:
deepen the supporting knee briefly and release the pelvis outward.

3.180–4.384
Begin lowering the hands gradually.
The hips never stop.
One hand moves slightly downward while the second follows.

4.384–5.187
SHOT 03.
Lyric: “旁人敬酒我先斟满它”.
Hands cross in front of the torso.
Do not mime drinking.
The lower body continues uninterrupted:
pelvic accent → weight transfer → waist follow-through.

5.187–5.989
Hands descend toward the waist.
Both knees remain asymmetrical.
The pelvis performs a compact figure-eight pathway.

5.989
Downbeat:
clear directional hip change.

5.989–7.595
Keep the body low.
The hands now frame the waist/hip area naturally.
The pelvis remains the primary movement source.
No literal touching performance.

7.595–7.996
Shoulders counter-rotate subtly against the hip direction.

7.996
Phrase accent:
strong outward hip extension.

7.996–9.601
Maintain the low side-lunge phrase.
Repeat compact alternating hip accents.
The extended pointed leg remains active.

9.601
Downbeat.
Begin the body-level transition.

9.601–10.003
SHOT 04.
Lyric: “他挑面纱笑说不必遮”.
One hand moves upward toward the face.
The other remains lower.
The pelvis continues.

10.003–10.404
Supporting leg begins extending.
Pelvis rises.
Waist follows.

10.404–10.805
Chest opens.
Head begins to lift.
The hands remain part of the face-framing composition.

10.805
Phrase accent.
The wrists shift out of perfect alignment.
A brief fragment of the eyes or upper face may become visible.
Do not remove the veil.

10.805–11.608
Continue rising.
The face becomes more readable through the changed angle rather than through a literal reveal.
Hip rhythm continues at reduced amplitude.

11.608–12.411
Turn the torso slightly three-quarter.
One hand stays near the face.
The second hand remains lower.
The body keeps subtle side-to-side weight transfer.

12.411
Downbeat.
Complete the standing rise.

12.411–13.213
SHOT 05.
Hold a three-quarter body line while continuing a small residual hip rhythm.
The gaze becomes more direct.
The hand-frame remains asymmetrical.

13.213–13.550
Begin lowering the center of gravity again.
The pelvis shifts toward one supporting leg.
The knee starts taking weight.

TRANSITION_TO_SEGMENT_04:

End while the body is already descending.
The final physical chain is:
hip shift → knee flexion → pelvis lowering → torso following.
SEGMENT 04 begins directly from the loaded knee.
```

---

# SEGMENT 04｜37.550–49.666（膝部特写开场 · 下沉不断舞）

```text
subject_definitions:

<Subject 1> is the exact same adult live-action person from <Picture 1>, same locked costume; the translucent dark-gauze face veil is worn in every frame and is never removed, lifted or parted; the face is never clearly visible.

<Subject 2> is the exact same ancient Chinese elevated ceremonial pavilion / banquet hall as <Picture 1>, always heavily defocused; no people, no location/time/weather change.

<Audio 1> is the song track, referenced only for rhythm: beat 0.4013s, phrase 3.2107s.
Lyric onsets in this segment: 0.100 / 2.533 / 6.583 / 8.983.
No signal copy, no lip-sync.

PERFORMANCE:

Adult sensual ancient-Chinese dance, mature, elegant, non-explicit.
Primary driver: the KNEES and the ground — low grounded hip rhythm, bent-knee weight transfer, waist spiral, controlled rise.
The body becomes lower, heavier and more compressed without ever stopping the hip choreography.

LYRIC_RELATION:

Do not literally laugh, point to knees, show scars or kneel for the lyric.
The dance first appears light and decorative, then the center of gravity drops and the body carries a remembered weight.

VEIL_LOCK:

The veil covers the face in every frame; the head turn toward camera at 6.583 is a veiled profile only.

CAMERA:

SHOT 01 opens on a tight KNEE / LOWER-SKIRT close-up inherited from SEGMENT 03.
Then: waist and upper body reveal → medium/full body on the final rise.
Restrained.

COLOR_AND_LENS:

The dancer's costume is a rich violet-purple and it must read as true, saturated purple in every frame — never gray, never black, never blue-shifted.
Lighting: a warm golden key from the practical lanterns falls on the dancer from upstage-left, wrapping the purple fabric with amber reflections; a soft warm-neutral fill protects the midtones so the purple keeps its hue, saturation and fabric detail.
The background pavilion stays in cooler indigo and blue-gray, one to two stops darker than the subject, so the warm-lit purple figure separates cleanly from the cold set.
Shadows on the subject are warm brown with visible fabric texture, never black-green crush.
Overall exposure is moderate-contrast and luminous: lifted midtones, soft highlight roll-off on lantern glow, subtle bloom and halation, fine film grain, medium depth of field.
Skin stays warm and natural; deep vermilion appears only as tiny practical accents.
Keep the light DIRECTION consistent with <Picture 1>, but raise the key intensity on the subject as specified above — the subject must never sink into the background's darkness.

AUDIO:

<Audio 1> drives all movement; follow the lyric onsets above.

integrated_multimodal_description:

00.000–0.716
SHOT 01 — KNEE / LOWER-SKIRT CLOSE-UP.
Begin with the supporting knee already loaded from SEGMENT 03; body low, pelvis active, the purple skirt pooling around the bend in warm pool-of-light.

0.716–2.400
Weight transfers between the legs at kneeling height; the knee initiates, the pelvis follows.
Lyric: “满堂哄笑呀” (from 0.100): the choreography briefly turns decorative — a larger sleeve path, lifted torso, smooth hip accent.
No literal laughter.

2.400–6.483
Lyric: “膝上半痕是昨日跪疤” (from 2.533).
The body sinks further: knees soften and carry more weight, pelvis keeps its rhythm, waist folds slightly; one low grounded pelvic sweep on the phrase accent at 4.328.
Compressed but graceful.

6.483–8.916
Lyric: “仙子是我吗” (from 6.583).
The head turns toward camera — a veiled profile only; the contrast lives between the elegant silhouette and the lowered, active body.

8.916–12.116
Lyric: “替他温酒又斟茶” (from 8.983).
A refined curved wrist pathway draws one serving-like arc without any mime; the hand movement only modifies the torso rhythm; hips continue underneath.
From 9.144 the full rise begins: heels stabilize, knees extend, pelvis rises, waist follows, chest opens last; full standing height at the phrase accent 10.750.
A subtle side turn closes the shoulders, then one hand extends outward toward the camera.

TRANSITION_TO_SEGMENT_05:

End with the wrist and hand still moving toward the lens; SEGMENT 05 opens on the same hand.
```

---

# SEGMENT 05｜49.666–63.466（手部特写开场 · 臂主线 AAB-C）

```text
subject_definitions:

<Subject 1> is the exact same adult live-action person from <Picture 1>, same locked costume; the translucent dark-gauze face veil is worn in every frame and is never removed, lifted or parted; the face is never clearly visible.

<Subject 2> is the exact same ancient Chinese elevated ceremonial pavilion / banquet hall as <Picture 1>, always heavily defocused; no people, no location/time/weather change.

<Audio 1> is the song track, referenced only for rhythm: beat 0.4013s, phrase 3.2107s, phrase accents at 1.844 / 5.055 / 8.266 / 11.476.
Lyric onsets in this segment: 0.134 / 4.184 / 7.417 / 10.267.
No signal copy, no lip-sync.

PERFORMANCE:

Adult sensual ancient-Chinese dance, mature, elegant, non-explicit.
Primary driver: the ARMS AND SLEEVES — the hips support, they do not lead.
Structure AAB-C: A1 wing-reach to the right with a trailing sleeve arc; A2 wing-reach to the left, larger; B one full level change (rise out of the low carriage, then re-sink); C a direction change led by the head.
Each lyric line changes the movement QUALITY, not the vocabulary.

LYRIC_RELATION:

满门生死凭他话 = contraction (the reach folds back through the sternum, smaller).
我能违他吗 = the strongest wing-reach, held one beat against the returning center.
只得随他啊 = release — the body follows its own weight downward.
帘外微风似有故人来 = the head changes direction first (eyes lead through the veil), the body follows through sequential mechanics.
Do not mime any of these words.

VEIL_LOCK:

The veil covers the face in every frame; the direction change is read from head angle alone.

CAMERA:

SHOT 01 opens on a tight HAND close-up inherited from SEGMENT 04.
Then: medium-close body frame → three-quarter body reveal toward the end.
Restrained.

COLOR_AND_LENS:

The dancer's costume is a rich violet-purple and it must read as true, saturated purple in every frame — never gray, never black, never blue-shifted.
Lighting: a warm golden key from the practical lanterns falls on the dancer from upstage-left, wrapping the purple fabric with amber reflections; a soft warm-neutral fill protects the midtones so the purple keeps its hue, saturation and fabric detail.
The background pavilion stays in cooler indigo and blue-gray, one to two stops darker than the subject, so the warm-lit purple figure separates cleanly from the cold set.
Shadows on the subject are warm brown with visible fabric texture, never black-green crush.
Overall exposure is moderate-contrast and luminous: lifted midtones, soft highlight roll-off on lantern glow, subtle bloom and halation, fine film grain, medium depth of field.
Skin stays warm and natural; deep vermilion appears only as tiny practical accents.
Keep the light DIRECTION consistent with <Picture 1>, but raise the key intensity on the subject as specified above — the subject must never sink into the background's darkness.

AUDIO:

<Audio 1> drives all movement; follow the lyric onsets and phrase accents above.

integrated_multimodal_description:

00.000–0.640
SHOT 01 — HAND / WRIST CLOSE-UP.
Continue directly from SEGMENT 04: the hand is extending toward the lens, wrist rotating outward, shoulder beginning to open, sleeve sliding down the forearm.

0.640–4.184
The chest follows, the waist slightly later.
A1: the near arm sweeps into a wing-reach with the sleeve trailing, then folds back through the sternum as the first lyric line “满门生死凭他话” begins at 0.134 — the reach narrows into a contained, smaller phrase.
Compact weight transfers under the feet.

4.184–5.055
Lyric: “我能违他吗”.
A2: the same wing-reach blooms on the opposite side, larger; the opposite shoulder counters.

5.055–7.417
Phrase accent: the strongest outward extension so far, held one beat against the pulling center — then B: a full level change, the body rising out of its low carriage, hips narrowing, sleeves gathering.

7.417–10.267
Lyric: “只得随他啊”.
The release: the pelvis leads the fall of the center, the body follows the direction created by its own weight, re-sinking toward the low carriage while the sleeves settle; no surrender gesture; the rhythm never stops.

10.267–11.476
Lyric: “帘外微风似有故人来”.
SHOT 02 — MEDIUM-CLOSE.
The head changes direction first — a veiled profile turning toward an implied sound beyond the frame; then shoulder, chest, waist, pelvis follow through sequential mechanics.

11.476–13.483
SHOT 03 — THREE-QUARTER BODY.
The hip rhythm continues while the whole direction of travel changes; the amplitude reduces; the veiled gaze stays directed beyond the frame; the body remains in motion.

TRANSITION_TO_SEGMENT_06:

The final state is a moving dance that has just changed direction, head already oriented toward the source; SEGMENT 06 opens on that orientation.
```

---

# SEGMENT 06｜63.466–75.570（纱后侧影特写开场 · 断拍与重启）

```text
subject_definitions:

<Subject 1> is the exact same adult live-action person from <Picture 1>, same locked costume; the translucent dark-gauze face veil is worn in every frame and is never removed, lifted or parted; there is NO eye close-up and NO moment where features become readable.

<Subject 2> is the exact same ancient Chinese elevated ceremonial pavilion / banquet hall as <Picture 1>, always heavily defocused; no people, no location/time/weather change.

<Audio 1> is the song track, referenced only for rhythm: beat 0.4013s, phrase 3.2107s.
Lyric onset in this segment: 0.100.
No signal copy, no lip-sync, no added dialogue.

PERFORMANCE:

Adult sensual ancient-Chinese dance, mature, elegant, non-explicit.
One short intentional interruption inside an otherwise continuous dance: head → shoulder → torso → waist → pelvis → full-body dance.
The pause is brief and musical; after it, the choreography becomes more open and self-directed, restarting from the feet upward.

LYRIC_RELATION:

“公子，是你吗” must not become a prolonged acting scene.
The recognition is expressed by a temporary interruption of the body's rhythm and a trembling stillness behind the veil.

VEIL_LOCK:

The veil covers the face in every frame, including the opening close-up.
The closest shot is a side-face silhouette behind the translucent gauze: only the profile line, the crown and the veil edge are readable; the gaze direction and a held breath (a faint trembling of the veil edge) carry the moment.
No features are ever visible.

CAMERA:

SHOT 01 opens on a tight VEIL / SIDE-FACE SILHOUETTE close-up inherited from SEGMENT 05.
Then: upper body → fuller dance frame.
Restrained and motivated.

COLOR_AND_LENS:

The dancer's costume is a rich violet-purple and it must read as true, saturated purple in every frame — never gray, never black, never blue-shifted.
Lighting: a warm golden key from the practical lanterns falls on the dancer from upstage-left, wrapping the purple fabric with amber reflections; a soft warm-neutral fill protects the midtones so the purple keeps its hue, saturation and fabric detail.
The background pavilion stays in cooler indigo and blue-gray, one to two stops darker than the subject, so the warm-lit purple figure separates cleanly from the cold set.
Shadows on the subject are warm brown with visible fabric texture, never black-green crush.
Overall exposure is moderate-contrast and luminous: lifted midtones, soft highlight roll-off on lantern glow, subtle bloom and halation, fine film grain, medium depth of field.
Skin stays warm and natural; deep vermilion appears only as tiny practical accents.
Keep the light DIRECTION consistent with <Picture 1>, but raise the key intensity on the subject as specified above — the subject must never sink into the background's darkness.
A thin warm rim traces the veil edge so the trembling silhouette reads clearly; the cool silver-blue stays in the background only.

AUDIO:

<Audio 1> drives timing; follow the lyric onset above.

integrated_multimodal_description:

00.000–0.486
SHOT 01 — VEIL / SIDE-FACE SILHOUETTE CLOSE-UP.
Continue directly from SEGMENT 05: the veiled head is already oriented toward the new direction; only the profile line, the gold crown and the warm-lit gauze edge are readable against the cool dark.

0.486–0.887
The head angle shifts slightly; a subtle change of attention; the veil edge trembles with a held breath.

0.887–4.098
Lyric: “公子，是你吗” (from 0.100).
The head turns a little more, the shoulder follows, the torso begins to follow while the lower body has not fully followed — a brief upper/lower contradiction.
At 3.696 the INTENTIONAL HERO LOCK: for a brief musical instant the body becomes almost still, the veiled profile fixed; no facial acting is possible or needed.

4.098–7.308
Phrase accent: motion restarts from the shoulder into the waist and pelvis, then the feet re-enter; one arm lifts outward with an open palm — an offering-like gesture that never fully presents, always behind the veil.
The body now feels less constrained.
SHOT 02 — UPPER BODY widens.

7.308–10.519
A flowing spiral: feet → weight transfer → hip → waist → torso → shoulders → arms; hip rhythm more pronounced; one larger waist wave; the head catches the body afterward.
SHOT 03 — FULLER BODY FRAME at 8.914 with a direction change.

10.519–12.104
Phrase accent: a full-body spiral completes; the arm rises diagonally, the waist spirals upward, the sleeve continues through the frame.

TRANSITION_TO_SEGMENT_07:

No freeze, no reset.
SEGMENT 07 begins on the same diagonally rising arm: the sleeve at head height, still traveling across the frame.
```

---

# SEGMENT 07｜75.570–87.667（腰袖特写开场 · 三圆联动 · 终帧）

```text
subject_definitions:

<Subject 1> is the exact same adult live-action person from <Picture 1>, same locked costume; the translucent dark-gauze face veil is worn in every frame and is never removed, lifted or parted; the face is never clearly visible, including in the final frame.

<Subject 2> is the exact same ancient Chinese elevated ceremonial pavilion / banquet hall as <Picture 1>, always heavily defocused; no people, no location/time/weather change.

<Audio 1> is the song track, referenced only for rhythm: BPM 149.5, beat 0.4013s, bar 1.6054s, phrase 3.2107s.
This segment has no lyrics.
No signal copy, no extra sound.

PERFORMANCE:

Adult sensual ancient-Chinese dance, mature, elegant, confident, non-explicit.
One concrete choreography in four phrases, executed in strict order — no simultaneous quality list:
Phrase 1: inherit the rising arm, complete its circle, settle into stance.
Phrase 2: travelling compact footwork with lateral hip accents and one horizontal sleeve sweep.
Phrase 3: mirrored sweep larger, the biggest wing arc, a fold-and-rebound, one full spin on the balls of the feet.
Phrase 4: final escalation and the Hero Frame.
Small travelling steps are allowed inside the pavilion; the dancer never walks casually and never stands still.

LYRIC_RELATION:

No lyrics.
The absence is part of the choreography: the body releases the restrictions established earlier.
No narrative acting.

VEIL_LOCK:

The veil covers the face in every frame, including the final Hero Frame; confidence is carried by the chin line, posture and open palm.

CAMERA:

SHOT 01 opens on a tight SLEEVE / WRIST close-up that contains SEGMENT 06's final gesture.
Then: medium shot → fuller body → slow push toward the Hero Frame.
No decorative orbit.

COLOR_AND_LENS:

The dancer's costume is a rich violet-purple and it must read as true, saturated purple in every frame — never gray, never black, never blue-shifted.
Lighting: a warm golden key from the practical lanterns falls on the dancer from upstage-left, wrapping the purple fabric with amber reflections; a soft warm-neutral fill protects the midtones so the purple keeps its hue, saturation and fabric detail.
The background pavilion stays in cooler indigo and blue-gray, one to two stops darker than the subject, so the warm-lit purple figure separates cleanly from the cold set.
Shadows on the subject are warm brown with visible fabric texture, never black-green crush.
Overall exposure is moderate-contrast and luminous: lifted midtones, soft highlight roll-off on lantern glow, subtle bloom and halation, fine film grain, medium depth of field.
Skin stays warm and natural; deep vermilion appears only as tiny practical accents.
Keep the light DIRECTION consistent with <Picture 1>, but raise the key intensity on the subject as specified above — the subject must never sink into the background's darkness.
This is the brightest key of all seven segments; raise the fill slightly for the final Hero Frame so the purple reaches its most luminous read.

AUDIO:

<Audio 1> drives timing; no extra sound.

integrated_multimodal_description:

00.000–0.422
SHOT 01 — SLEEVE / WRIST CLOSE-UP.
Continue SEGMENT 06's final gesture exactly: the diagonally rising arm carries the sleeve at head height across the upper frame.
The wrist unrolls, the hand draws one small outward circle, and the sleeve spills down across the shoulder line; the body is already stepping underneath.

00.422–1.626
SHOT 02 — MEDIUM.
The arm completes a full vertical circle overhead and folds down through the chest; the waist unwinds from the spiral into an upright stance; two quick compact side-steps travel the body toward frame-left; the first downbeat lands on the second step.

1.626–3.231
Phrase accent: a brief three-curve Hero Shape — right arm high, left arm low, hips offset — released immediately into motion.
The footwork becomes rapid compact steps travelling back toward frame-right: on every second step, toe-press → weight transfer → one lateral hip accent; the waist waves over the travelling rhythm.

3.231–4.837
SHOT 03 — FULLER BODY.
The left arm draws a large horizontal sweep at chest height; the sleeve crosses the body and re-opens; the pelvis keeps alternating accents underneath.
On the downbeat at 4.837 the whole body redirects.

4.837–6.442
The mirrored chain on the opposite side: right arm horizontal sweep, left sleeve re-opening, amplitude clearly larger than the first side; two travelling steps carry the body back toward frame-left; the feet stay rhythmically sharp.

6.442–8.047
Downbeat: the largest wing arc — both arms open through shoulders, elbows and wrists overhead, purple fabric luminous in the warm key; the hips keep a compact figure-eight underneath.
At 8.047 the arms fold and the body dips through the waist into a low rebound, pelvis leading, sleeve following with delayed momentum.

8.047–9.653
From the low rebound the body spins one full turn on the balls of the feet, sleeves flaring horizontally with the rotation, the veil edge lifting with the air; the turn decelerates into a settled three-quarter stance facing the camera.

9.653–11.258
SHOT 04 — MEDIUM-CLOSE.
The camera pushes in slowly.
Downbeat: the final escalation — rapid compact footwork opening into one travelling step forward, alternating hip transfer, waist spiral, shoulder counterrotation, then one last broad sleeve arc with the wrist releasing at its top.

11.258–12.097
Final phrase accent: expand into the strongest three-quarter Hero Frame of the sequence — one open palm toward the camera, chin raised behind the veil, purple fabric at its most luminous, body settled but still carrying residual momentum.
No artificial slow motion, no new lighting.

TRANSITION_OUT:

End on the completed Hero Frame.
```


---

## FILE: case-studies/2026-09-snap-lock-editorial-reveal/README.md (https://github.com/LoveRain1997/h3-prompt-journal/blob/HEAD/case-studies/2026-09-snap-lock-editorial-reveal/README.md)

# Case Study 010 — 音频驱动的「快照-硬锁」编辑式人物揭示（Snap-Lock Editorial Reveal）

> Audio-synced editorial snap sequence: a two-state camera state machine (HARD LOCK / INSTANT BURST), zero appearance description — PICTURE 1 carries identity, AUDIO 1 carries time.

## Inputs（输入）

- **PICTURE 1**：一张人物参考图，作为**唯一**外观与环境来源（身份、五官、发型、服装、配饰、鞋、背景、光向、写实质感、起始状态）。
- **AUDIO 1**：一段音乐，作为**唯一**时间与节奏来源（所有切换、定格、冲击、眨眼、末段旋转的触发时刻）。
- 目标：**5.30 秒、竖屏 9:16、真人写实**电影感短片。
- 提示词语言：**英文**；**全文不含任何外貌/服装/环境描写**（全部交给 PICTURE 1）。
- 反推来源样片：[`reference-video.mp4`](./reference-video.mp4)（576×1024，5.34s，仅作动作/剪辑结构参考；PICTURE 1 与 AUDIO 1 在生成时另行提供）。
- 最终可直接粘贴的提示词全文：[`prompt.md`](./prompt.md)。

## The problem（问题：它看起来像一镜到底，但不是）

按常规抽帧、拼 contact sheet，看到的是「脚 → 小腿 → 裙摆 → 胸口 → 脸」方向一致、一气呵成的位移，很容易下结论：

> 一镜到底的 tilt-up reveal，前奏慢扫，约 2.1 秒 drop 处快速 whip 到脸，再锁定脸部特写。

**这个结论是错的。** 真正跑通的提示词证明它不是连续运镜，而是一条**离散的编辑式快照序列**：

```
图1 → 瞬间快切 → 静态英雄帧·脚 → 瞬间快切 → 静态英雄帧·膝
   → 瞬间快切 → 静态英雄帧·上半身 → 瞬间快切 → 静态英雄帧·脸
   → 主节拍极端推脸 → 咚/咚/咚 三次相机冲击 → 超长静态脸
   → 闭眼 →（音频末点）睁眼 + 90°旋转 → 锁死
```

### 为什么稀疏抽帧会把「快切」看成「连续扫」

1. 每次快切（burst）只有 **0.12–0.20 秒**，且带强烈方向运动模糊；
2. 用 4fps 抽帧（间隔 0.25s）会**整段跨过 burst**，网格里只剩切换前、后两张静止构图；
3. 这些构图方向完全一致（都沿身体向上），于是大脑用最常见的先验「方向一致的位移 = 连续运镜」把中间空白自动补成平滑上摇。

这是「稀疏采样漏掉关键帧」叠加「先验捕获」的双重翻车——**证据里并没有连续运动，是分析者替它脑补了连续运动。**

判别的钥匙在音频：频谱呈现清晰的节拍网格，把节拍 onset 与画面对账，会发现**每次切换、定格、冲击都精确挂在节拍事件上**；连续运镜不会在每个身体段落出现「动一下 → 彻底静止」的台阶。

## The breakthrough（突破：两态相机状态机 + 音频绝对时间轴）

### 1. 相机只有两种状态，绝不混合

| 状态 | 行为 |
|---|---|
| **STATE A — HARD LOCK（硬锁）** | 零运动：不推拉/不平移/不摇/不滚/不漂移/无视差/无手持/无微震/无残留动量，像一张暂时停在屏幕上的照片 |
| **STATE B — INSTANT BURST（瞬间爆发）** | 仅在指定极短区间内运动，极快、带方向运动模糊，区间结束立刻归零；无 ease、无缓入缓出、无平滑插值 |

提示词反复强调 `Never blend these two states together`，并专门防止模型把静态帧「做成慢动作/继续上飘」——每个静态段都把禁令重写一遍。

主节拍后的情绪顶点也用同一套语言：`00:02.050` 踩拍瞬间启动全片最快的极端推脸；`00:02.420–00:02.500` 给三次**可分别辨认**的相机冲击（位移小震 → 前推 punch → 旋转微震），节奏 `THUMP → THUMP → THUMP → LOCK`，明确「不是连续震动、不是手持抖动」，且**只动相机、脸和身体不变形**；随后是 2.3 秒超长静态脸，只允许呼吸/眼动/发丝微动。极致的动 → 极致的静。

### 2. AUDIO 1 是唯一时间轴

`<Audio 1>` 被定义为唯一时间参考，并写明 `Do not invent a separate rhythm`、`Do not replace AUDIO 1 with generic cinematic timing`。每个视觉事件都标注「由 `<Audio 1>` 触发」并给出毫秒时间码。只看画面不做音画对账，会把「卡点剪辑」误判成「自由运镜」，这是两类完全不同的生成任务。

### 3. 一个字都不描写外貌

当 PICTURE 1 已锁定外观，提示词的稀缺 token 预算应**全部投给模型最容易做错的时间纪律**，而不是重复图里已有的信息。PICTURE 1 被设为「唯一视觉权威」，相机**只改变自己的取景、位置、透视与朝向**；外观交给图，运动交给文字。

### 4. 「人物躺着」也是错觉：横的是相机

抽帧时脸是横的，容易以为人物躺/坐姿。真相是：第一个脚部英雄帧后相机就进入**相对原图约顺时针 90° 的横置取景**，**人物本身不旋转、身体不扭转、姿势不变形**；中段（膝→大腿→腰→躯干→胸→颈）地平线必须一直保持斜置，不许找平、不许反向滚回；最后的 90° Roll 只留给结尾脸部 payoff。区分「人物转了」还是「相机滚了」，要看背景地平线与身体的相对关系。

## The named language（命名）

> **Snap-Lock Editorial Reveal（快照-硬锁编辑式揭示）**

「Snap-Lock」给出相机的二态状态机（瞬间切换 + 绝对静止交替），「Editorial」强调这是剪辑逻辑而非一条摄影机运动轨迹，「Reveal」说明内容是沿身体从脚到头的分段揭示。交付物不是一段画面描述，而是一个**可被音频逐点触发的相机状态机**。

## 逐段时间预算（可当分镜）

| 时间 | 事件 | 相机状态 |
|---|---|---|
| 00:00.000–00:00.120 | 图1 → 极低角度脚部特写，建立约 90° 横置 | SNAP |
| 00:00.120–00:00.420 | 英雄帧 01·脚/鞋/小腿 | 硬锁 |
| 00:00.420–00:00.560 | 脚 → 膝/大腿 | SNAP |
| 00:00.560–00:01.020 | 英雄帧 02·膝 | 硬锁 |
| 00:01.020–00:01.180 | 膝 → 上躯干/肩/颈 | SNAP |
| 00:01.180–00:01.620 | 英雄帧 03·上半身（地平线保持斜） | 硬锁 |
| 00:01.620–00:01.820 | 最猛上冲：颈/躯干 → 脸 | SNAP |
| 00:01.820–00:02.050 | 英雄帧 04·脸，双眼睁开直视镜头 | 硬锁 |
| 00:02.050–00:02.420 | 踩主节拍，极端推脸（全片最快） | 极端 BURST |
| 00:02.420–00:02.500 | 三次离散冲击：位移/前推/旋转 → 锁 | THUMP×3 |
| 00:02.500–00:04.800 | 最长静态英雄脸（仅呼吸/眼动/发丝） | 硬锁 |
| 00:04.800–00:05.020 | 缓慢闭眼、视线微垂（构图不变） | 硬锁 |
| 00:05.020–00:05.100 | 静态间隔，保持闭眼 | 硬锁 |
| 00:05.100–00:05.300 | 音频末点：睁眼 + 顺时针 90° Roll + 轻微前推 | ROLL |
| 00:05.300 | 终帧横置，睁眼直视，锁死 | 硬锁 |

## 48-hour takeaway（可复用清单）

1. **先判连续运镜还是离散快切**：转折段用 8fps 以上密帧，别让稀疏采样跨过整个 burst；找「运动—完全静止」的台阶。
2. **音频做时间轴对账**：每个切换/定格/冲击是否挂在节拍 onset 上？是 → 编辑式快照；否 → 才考虑自由运镜。
3. **有参考图就别写外貌**：图锁外观，文字只写相机/时间/状态机。
4. **静态与爆发写成两个互斥状态**：静态段反复写死「不许慢动作/漂移/继续运动」。
5. **横置先怀疑相机 Roll**：看地平线与身体的相对关系，别轻易判定人物躺/转。
6. **冲击要可数、不可糊**：多次冲击分别可读，明确「不是连续震动」，且只动相机不变形人物。

本案例是配套反推技能 [`video-to-h3-prompt`](https://github.com/LoveRain1997/video-to-h3-prompt) 的实战产物；其「cut vs 连续判别 / 音画交叉验证 / 参考图锁定」方法即本次复盘所用流程。

## The final prompt

完整可粘贴提示词见 [`prompt.md`](./prompt.md)（PICTURE 1 + AUDIO 1 双参考，零外貌描写）。核心骨架：

1. **来源声明**：PICTURE 1 唯一视觉权威、AUDIO 1 唯一时间权威
2. **CORE VIDEO STRUCTURE**：离散快照序列总览
3. **两态定义**：STATE A HARD LOCK / STATE B INSTANT BURST
4. **PICTURE 1 CONSISTENCY / AUDIO 1 TEMPORAL REFERENCE / CAMERA ORIENTATION**
5. **逐段时间块**：每个 SNAP / HERO FRAME / BURST / IMPACT / EYE ACTION / ROLL 的毫秒区间与相机行为
6. **MIDDLE TILT / STATIC CONSISTENCY / EDITORIAL SNAP / ABSOLUTE TEMPORAL** 四条总规则
7. **REALISM / NEGATIVE CONSTRAINTS**

## Result（成片要点）

- 脚、膝、上半身、脸四个静态「英雄帧」被三次极短快切串联，而非一条连续上摇；
- 00:02.050 主节拍触发极端推脸，紧接三次可分辨的相机冲击后进入彻底静止；
- 中段始终保持横置/倾斜取景，人物不转；
- 结尾音频末点睁眼并完成 90° 旋转锁定；
- 全程外观与环境与 PICTURE 1 严格一致，无任何额外人物/道具/文字。

## Assets

| 文件 | 作用 |
|---|---|
| [`prompt.md`](./prompt.md) | 最终有效提示词全文（可直接粘贴） |
| [`reference-video.mp4`](./reference-video.mp4) | 反推来源样片，5.34s，竖屏 9:16 |

> 生成时需自行提供 PICTURE 1（人物/环境参考图）与 AUDIO 1（音乐音轨）；仓库不含这两个受版权/肖像约束的素材。

## Tags

`#H3` `#prompt-engineering` `#reverse-engineering` `#video-to-prompt` `#audio-sync` `#beat-sync` `#camera-state-machine` `#snap-cut` `#hero-frame` `#freeze` `#camera-roll` `#image-locked` `#zero-appearance-description` `#vertical-video` `#Ref2VA`


---

## FILE: case-studies/2026-09-snap-lock-editorial-reveal/prompt.md (https://github.com/LoveRain1997/h3-prompt-journal/blob/HEAD/case-studies/2026-09-snap-lock-editorial-reveal/prompt.md)

Use PICTURE 1 as the ONLY visual identity and environment source.
Use AUDIO 1 as the ONLY temporal and rhythmic source.
Create a 5.30-second vertical 9:16 realistic live-action cinematic video.
PICTURE 1 defines the exact character identity, appearance, clothing, accessories, footwear, environment, lighting, photographic realism and starting visual state.
AUDIO 1 defines the exact timing, rhythmic accents, major beat, movement triggers, facial timing, editorial transitions and final camera rotation.
Do not invent a separate rhythm.
Do not replace AUDIO 1 with generic cinematic timing.
All specified visual events must explicitly synchronize to <Audio 1>.
================================================== CORE VIDEO STRUCTURE
This video is an editorial sequence of discrete compositions.
It is NOT one continuous camera movement.
The visual mechanism is:
PICTURE 1 → INSTANT SNAP → STATIC HERO 01 / FEET → INSTANT SNAP → STATIC HERO 02 / KNEE → INSTANT SNAP → STATIC HERO 03 / UPPER BODY → INSTANT SNAP → STATIC HERO 04 / FACE → AUDIO 1 MAJOR BEAT → EXTREME FACE BURST → THREE CAMERA IMPACTS → ABSOLUTE STATIC FACE HERO → EYE CLOSURE → AUDIO 1 FINAL EVENT → 90° FINAL ROLL → LOCK.
The camera repeatedly alternates between exactly two states:
STATE A — HARD LOCK
STATE B — INSTANT BURST
Never blend these two states together.
Do not smooth between them.
Do not interpolate continuously through a static section.
================================================== STATE A — HARD LOCK
CAMERA = ABSOLUTE ZERO MOTION.
No camera translation.
No zoom.
No push.
No pull.
No pan.
No tilt.
No orbit.
No roll.
No drift.
No parallax.
No handheld motion.
No micro-shake.
No residual movement.
No speed ramp.
No hidden movement.
No continuing previous camera momentum.
The composition behaves exactly like a still photograph temporarily held on screen.
Once a static frame is reached:
THE CAMERA STOPS COMPLETELY.
================================================== STATE B — INSTANT CAMERA BURST
Camera movement is allowed ONLY inside the explicitly specified burst interval.
Movement starts immediately at the beginning of the interval.
Movement is extremely fast.
Movement ends immediately at the end of the interval.
No gradual acceleration.
No gradual deceleration.
No easing.
No smooth interpolation.
Strong directional optical motion blur may appear during the burst.
The burst must complete the entire reframing.
At the exact end of every burst:
CAMERA = ZERO MOTION.
================================================== PICTURE 1 CONSISTENCY
Preserve the exact character from PICTURE 1.
Preserve:
facial identity
facial proportions
eye shape
hairstyle
hair color
hairline
skin texture
body proportions
costume
accessories
footwear
background
environment
lighting direction
color relationships
realistic live-action photographic appearance.
Do not redesign the character.
Do not modify the clothing.
Do not modify the hairstyle.
Do not modify facial structure.
Do not add people.
Do not change the environment.
Do not invent new props.
Do not reinterpret the image into animation, illustration or game CG.
The camera changes only its own framing, position, perspective and orientation.
================================================== AUDIO 1 — ABSOLUTE TEMPORAL REFERENCE
<Audio 1> is the actual supplied audio track.
<Audio 1> is the ONLY temporal reference for the sequence.
Every editorial transition and major visual event must be synchronized to <Audio 1>.
The following events are specifically triggered by <Audio 1>:
the initial FEET snap
the FEET Hero Frame
the KNEE snap
the KNEE Hero Frame
the UPPER BODY snap
the UPPER BODY Hero Frame
the FACE snap
the FACE Hero Frame
the 00:02.050 major beat
the extreme face burst
the three impacts
the eye closure
the final eye reopening
the final 90-degree Roll.
Do not invent independent timing.
Do not add another musical rhythm.
================================================== CAMERA ORIENTATION
The first camera snap establishes the sideways visual language.
Immediately after entering the first FEET Hero composition, the camera adopts approximately 90-degree clockwise orientation relative to the original PICTURE 1 composition.
The character itself does NOT rotate.
The body does NOT twist.
The pose does NOT deform.
The sideways appearance is created by camera orientation only.
After the FEET Hero composition is established, maintain the sideways / strongly tilted camera language through:
KNEE
UPPER LEG
WAIST
UPPER TORSO
CHEST AREA
NECK.
Do not level the horizon during the entire middle ascent.
Do not repeatedly alternate between level and tilted framing.
The tilted orientation is a persistent composition rule.
The final 90-degree Roll is reserved for the final facial payoff.
================================================== 00:00.000–00:00.120 INITIAL EDITORIAL SNAP — PICTURE 1 → FEET
Begin directly from the exact visual state of PICTURE 1.
Do not hold the original composition.
At the very beginning of the video, immediately synchronize to the opening timing of <Audio 1> and perform a very short editorial camera burst into the first Hero composition.
Reposition directly from the original PICTURE 1 framing to:
EXTREME LOW-ANGLE FEET / SHOES / LOWER LEGS CLOSE-UP.
The feet and shoes become the dominant visual subject.
The camera is extremely low near ground level.
The lower legs extend strongly through the vertical composition.
This is a discrete reframing event.
It is NOT a gradual push.
It is NOT a slow zoom.
It is NOT a continuous downward travel.
Do not ease into the composition.
Complete the entire reframing inside this very short interval.
Immediately at 00:00.120:
CAMERA = ZERO MOTION.
================================================== 00:00.120–00:00.420 STATIC HERO FRAME 01 — FEET
This is HERO MOMENT 1.
Synchronize this Hero Frame to the continuing rhythm of <Audio 1>.
The feet and lower legs remain in exactly the same composition.
CAMERA = ABSOLUTE HARD LOCK.
No translation.
No zoom.
No push.
No pull.
No pan.
No tilt.
No orbit.
No roll.
No drift.
No parallax.
No handheld movement.
No shake.
No residual motion.
Treat this composition as a still photographic frame.
The footwear remains clearly readable.
The lower-leg silhouette remains unchanged.
Only minimal natural human motion is allowed.
The camera itself is completely frozen.
Do NOT turn this into slow motion.
Do NOT let the camera continue traveling upward.
The entire section must visibly read as a Hero Frame.
================================================== 00:00.420–00:00.560 INSTANT CAMERA BURST 01 — FEET → KNEE
At the exact timing established by <Audio 1>, immediately leave the FEET Hero Frame.
Perform a discrete camera burst from:
FEET / LOWER LEGS
directly to:
KNEE / UPPER LEG.
This is an editorial snap between two hero compositions.
Very fast directional movement.
Strong directional optical motion blur.
The entire spatial repositioning happens inside this 0.14-second interval.
No gradual acceleration.
No gradual deceleration.
No easing.
No smooth interpolation.
Do not create a continuous camera journey.
Immediately at 00:00.560:
CAMERA = ZERO MOTION.
================================================== 00:00.560–00:01.020 STATIC HERO FRAME 02 — KNEE
This is HERO MOMENT 2.
Synchronize this static composition to the continuing rhythm of <Audio 1>.
The KNEE / UPPER LEG region becomes the dominant visual subject.
The camera is already in the established sideways / tilted orientation.
CAMERA = ABSOLUTE HARD LOCK.
No translation.
No zoom.
No push.
No pull.
No pan.
No tilt.
No orbit.
No roll.
No drift.
No parallax.
No handheld motion.
No micro-shake.
No residual movement.
The knee composition remains visually identical for the entire section.
Do NOT:
continue upward
slowly travel toward the waist
slow zoom
settle gradually
drift upward
increase camera distance
decrease camera distance.
This is a TRUE STATIC HERO FRAME.
The camera behavior must be identical to the successful FACE STATIC behavior.
The knee frame must have equal visual authority.
================================================== 00:01.020–00:01.180 INSTANT CAMERA BURST 02 — KNEE → UPPER BODY
At the next timing transition established by <Audio 1>, immediately leave the locked KNEE Hero Frame.
Perform a very fast discrete camera burst from:
KNEE / UPPER LEG
to:
UPPER TORSO / SHOULDER / NECK.
Do not smoothly connect the two compositions.
Do not ease.
Do not gradually accelerate.
Do not gradually decelerate.
Strong directional motion blur.
The entire camera repositioning occurs inside this short interval.
At the exact end:
CAMERA = ZERO MOTION.
================================================== 00:01.180–00:01.620 STATIC HERO FRAME 03 — UPPER BODY
This is HERO MOMENT 3.
Synchronize the static composition to <Audio 1>.
Frame:
UPPER TORSO
SHOULDER
NECK.
The camera remains strongly sideways / tilted.
IMPORTANT:
The horizon MUST remain visibly diagonal.
Do NOT level the horizon.
Do NOT counter-roll.
Do NOT return to normal vertical orientation.
The tilted composition continues through the complete upper-body section.
CAMERA = ABSOLUTE HARD LOCK.
No translation.
No zoom.
No push.
No pull.
No pan.
No tilt.
No orbit.
No roll.
No drift.
No parallax.
No handheld movement.
No micro-shake.
No residual movement.
The frame behaves like a still photograph.
The visual authority must match the KNEE and FACE Hero Frames.
================================================== 00:01.620–00:01.820 INSTANT CAMERA BURST 03 — UPPER BODY → FACE
At the exact continuation point established by <Audio 1>, leave the UPPER BODY Hero Frame immediately.
Perform the most aggressive upward camera burst so far.
Move directly from:
NECK / UPPER TORSO
to:
FACE.
Very short duration.
Strong forward movement.
Very high camera velocity.
Strong directional optical motion blur.
Maintain the sideways camera orientation during the burst.
Do NOT gradually approach the face.
Do NOT slowly zoom.
Do NOT ease.
At 00:01.820:
CAMERA = ZERO MOTION.
================================================== 00:01.820–00:02.050 STATIC FRAME 04 — FACE
This is the facial Hero Frame.
Synchronize this composition to the continuing rhythm of <Audio 1>.
The face fills most of the vertical frame.
Approximately 90-degree sideways orientation remains active.
CAMERA = ABSOLUTE HARD LOCK.
No translation.
No zoom.
No push.
No pull.
No pan.
No tilt.
No orbit.
No roll.
No drift.
No parallax.
No handheld motion.
No shake.
No residual movement.
The face remains completely fixed in composition.
Both eyes remain open.
Direct gaze into the lens.
Do not continue the previous camera momentum.
The camera has reached ZERO MOTION.
================================================== 00:02.050 MAJOR AUDIO 1 BEAT
At the exact major beat of <Audio 1> at 00:02.050:
START THE EXTREME FACE BURST IMMEDIATELY.
The movement begins exactly on the <Audio 1> beat.
Do not prepare.
Do not ease.
Do not anticipate the beat.
Do not accelerate gradually.
The movement begins as an instantaneous visual shock.
================================================== 00:02.050–00:02.420 EXTREME FACE BURST
This is the fastest camera movement in the entire video.
Rapidly push toward the face.
Cover a large visual distance inside this short interval.
Very strong forward movement.
Very strong directional motion blur.
No smooth cinematic glide.
No dreamy movement.
No gradual acceleration.
No gradual deceleration.
No easing.
The camera moves aggressively and immediately.
At exactly 00:02.420:
STOP COMPLETELY.
================================================== 00:02.420–00:02.500 THREE DISCRETE CAMERA IMPACTS
At the exact end of the <Audio 1> extreme burst:
CREATE THREE DISTINCT CAMERA IMPACTS.
IMPACT 1: very brief tiny positional jolt.
IMPACT 2: very brief forward scale punch.
IMPACT 3: very brief rotational micro-jolt.
The three impacts are individually readable.
The rhythm is:
THUMP
→
THUMP
→
THUMP
→
LOCK.
These are three separate short camera impacts.
NOT continuous vibration.
NOT random jitter.
NOT handheld shaking.
NOT prolonged shake.
Do not deform the face.
Do not deform the head.
Do not move the body.
Only the camera receives the three impacts.
Immediately after IMPACT 3:
CAMERA = ABSOLUTE HARD LOCK.
================================================== 00:02.500–00:04.800 STATIC HERO FACE FRAME
This is the longest Hero Frame.
The face remains almost perfectly fixed in composition.
CAMERA = ABSOLUTE HARD LOCK.
No zoom.
No push.
No pull.
No pan.
No tilt.
No orbit.
No roll.
No drift.
No parallax.
No shake.
No handheld movement.
No camera micro-motion.
The intended contrast is:
EXTREME SPEED
→
THUMP
→
THUMP
→
THUMP
→
COMPLETE STILLNESS.
Only extremely subtle natural human micro-motion is allowed:
very slight breathing
tiny eye movement
minimal natural hair movement.
The camera itself does not move.
================================================== 00:04.800–00:05.020 STATIC EYE ACTION
Follow the exact timing of <Audio 1>.
At 00:04.800:
the character slowly closes her eyes
and slightly lowers her gaze.
CAMERA = ABSOLUTE HARD LOCK.
No camera movement.
No head movement.
No zoom.
No facial deformation.
This is an action inside the existing Hero Face Frame.
Do not create a new camera composition.
================================================== 00:05.020–00:05.100 STATIC TRANSITION GAP
Continue following <Audio 1>.
CAMERA = ABSOLUTE HARD LOCK.
Maintain the exact facial composition.
Do not introduce camera movement before the final <Audio 1> event.
The eyes remain closed.
Hold the frame.
================================================== 00:05.100 FINAL AUDIO 1 EVENT
At the exact final specified event in <Audio 1>:
the eyes reopen.
Immediately on the same <Audio 1> event:
BEGIN THE FINAL CAMERA ROLL.
No preparation.
No pre-roll.
No gradual anticipation.
================================================== 00:05.100–00:05.300 FINAL CAMERA ROLL
Perform a decisive clockwise camera Roll.
Approximately 90 degrees.
Combine the Roll with a subtle forward push.
The rotation starts immediately at the exact <Audio 1> event.
No easing.
No preparation.
No random shake.
No drifting.
The camera completes the entire rotation inside this 0.20-second interval.
The character's eyes reopen and remain directed into the lens during the Roll.
The face becomes progressively larger as the camera approaches.
At 00:05.300:
the approximately 90-degree Roll is complete.
CAMERA = HARD LOCK.
================================================== FINAL FRAME
The final frame is intentionally rotated sideways.
The face fills most of the frame.
Eyes open.
Direct gaze.
Character identity remains exactly consistent with PICTURE 1.
CAMERA = COMPLETELY LOCKED.
No additional movement.
No residual drift.
No shake.
Hold the final composition.
================================================== MIDDLE TILT RULE
CRITICAL:
From the KNEE Hero Frame through:
KNEE
→ UPPER LEG
→ WAIST
→ UPPER TORSO
→ SHOULDER
→ NECK
→ CHEST AREA
the camera must remain visibly tilted.
The horizon must stay diagonal.
Do NOT return to a level horizon.
Do NOT counter-roll.
Do NOT alternate between tilted and level framing.
The tilted orientation is continuous throughout this middle visual phase.
The camera may move only during the explicitly defined Burst intervals.
The camera returns to ZERO MOTION immediately after each Burst.
================================================== STATIC FRAME CONSISTENCY RULE
The following compositions must use identical HARD LOCK behavior:
FEET HERO FRAME
KNEE HERO FRAME
UPPER BODY HERO FRAME
FACE HERO FRAME
STATIC HERO FACE FRAME.
The camera behavior is identical in all static states.
Only framing and duration are different.
Every static state must look like a still photograph held on screen.
Do not allow the model to weaken the earlier frames into slow movement.
Especially:
FEET must lock.
KNEE must lock.
UPPER BODY must lock.
FACE must lock.
================================================== EDITORIAL SNAP RULE
Every composition change is a discrete editorial camera burst.
The intended structure is:
PICTURE 1
→ INSTANT SNAP
→ FEET LOCK
→ INSTANT SNAP
→ KNEE LOCK
→ INSTANT SNAP
→ UPPER BODY LOCK
→ INSTANT SNAP
→ FACE LOCK.
Never create one continuous camera journey connecting these compositions.
The bursts are intentionally short.
The static frames are intentionally obvious.
================================================== ABSOLUTE TEMPORAL RULE
Never allow camera motion to leak across time boundaries.
For every STATIC interval:
CAMERA = ZERO MOTION.
For every BURST interval:
ALL CAMERA MOVEMENT MUST OCCUR INSIDE THAT INTERVAL.
Never begin a burst early.
Never continue a burst after its endpoint.
Never interpolate smoothly through a static frame.
Never carry previous camera momentum into the next static composition.
The exact temporal behavior is:
00:00.000–00:00.120 INITIAL SNAP
00:00.120–00:00.420 FEET FREEZE
00:00.420–00:00.560 SNAP
00:00.560–00:01.020 KNEE FREEZE
00:01.020–00:01.180 SNAP
00:01.180–00:01.620 UPPER BODY FREEZE
00:01.620–00:01.820 SNAP
00:01.820–00:02.050 FACE FREEZE
00:02.050–00:02.420 EXTREME FACE SNAP
00:02.420–00:02.500 THUMP / THUMP / THUMP
00:02.500–00:04.800 FACE FREEZE
00:04.800–00:05.020 EYE ACTION
00:05.020–00:05.100 FACE FREEZE
00:05.100–00:05.300 FINAL 90° ROLL
FINAL FRAME LOCK.
================================================== AUDIO 1 REFERENCE LOCK
Use <Audio 1> as the actual supplied music track.
Synchronize all major visual events directly to <Audio 1>.
00:00.000 <Audio 1> establishes the opening visual timing.
00:00.120 the first FEET Hero composition is fully established.
00:00.420 <Audio 1> progression triggers the first visual SNAP from FEET to KNEE.
00:00.560 the KNEE Hero composition becomes completely static.
00:01.020 <Audio 1> progression triggers the second visual SNAP from KNEE to UPPER BODY.
00:01.180 the UPPER BODY Hero composition becomes completely static.
00:01.620 <Audio 1> progression triggers the third visual SNAP from UPPER BODY to FACE.
00:01.820 the FACE Hero composition becomes completely static.
00:02.050 the specified major beat in <Audio 1> triggers the extreme face burst.
00:02.420 the extreme movement ends.
00:02.420–00:02.500 three discrete camera impacts occur immediately after the <Audio 1> burst.
00:02.500 the camera becomes completely static again.
00:04.800 <Audio 1> triggers the eye closure.
00:05.100 <Audio 1> triggers the eye reopening and final 90-degree clockwise Roll.
00:05.300 the Roll ends and the final composition locks.
================================================== REALISM
Realistic live-action photography.
Real human skin texture.
Natural facial proportions.
Natural optical perspective.
Natural depth of field.
Natural lens behavior.
Natural directional motion blur during bursts.
Realistic exposure.
Realistic camera optics.
No CGI appearance.
No plastic skin.
No anime rendering.
No cartoon rendering.
No illustration.
No game-CG appearance.
No face deformation.
No body deformation.
================================================== NEGATIVE CONSTRAINTS
NO continuous camera travel from PICTURE 1 to FACE.
NO continuous zoom.
NO gradual movement during static frames.
NO slow-motion interpretation of static frames.
NO speed ramp during static frames.
NO gradual acceleration.
NO gradual deceleration.
NO easing.
NO residual camera movement after a burst.
NO camera drift.
NO parallax during static frames.
NO random shake.
NO handheld jitter.
NO continuous Roll before the final Roll.
NO leveling of the middle tilted composition.
NO counter-roll during KNEE → UPPER BODY → CHEST.
NO character rotation.
NO body twisting.
NO body deformation.
NO character redesign.
NO costume redesign.
NO environment replacement.
NO additional people.
NO new props.
NO text.
NO subtitles.
NO logos.
NO watermark.
The final result must clearly read as:
PICTURE 1 → SNAP → FEET HERO LOCK → SNAP → KNEE HERO LOCK → SNAP → TILTED UPPER-BODY HERO LOCK → SNAP → FACE HERO LOCK → AUDIO 1 EXTREME BEAT → EXTREME FACE BURST → THUMP → THUMP → THUMP → ABSOLUTE STATIC → AUDIO 1 FINAL EVENT → 90° ROLL → FINAL FACE LOCK.


---

## FILE: case-studies/2026-09-song-style-classical-dance-six-segment/README.md (https://github.com/LoveRain1997/h3-prompt-journal/blob/HEAD/case-studies/2026-09-song-style-classical-dance-six-segment/README.md)

# Case Study 011 — 宋韵身韵水袖古典舞 · 六段连续生成（Six-Segment Song-Style Shen-Yun Dance）

> 一支约 75.5 秒的中国古典舞（宋韵江南 / 身韵 / 水袖，李清照《鹧鸪天·桂花》），切成 6 段分别生成再拼接。**PICTURE 1 锁人物与环境、PICTURE 2 只做三视图身份澄清、分段 AUDIO 1 锁时间轴**；每段局部时钟、音乐跨段不重启、段末 hand-off 物理状态直接喂给下段；五个技术 / 脸部 Hero 机制**刻意互不重复**。

## Inputs（输入）

- **PICTURE 1**：唯一视觉权威——人物身份、五官、发型、体型、全套服装配饰鞋、环境、背景、灯光、写实质感、起始视觉状态。
- **PICTURE 2**：同一人物的**三视图身份 / 身体结构澄清**参考；**不**定义环境、灯光、姿势、构图、机位。
- **AUDIO 1（分段提供）**：唯一时间 / 音乐 / 节奏 / 唱词 / 动作触发 / 相机触发 / 剪辑时机权威。
- 目标：真人写实电影感中国古典舞；前 5 段竖屏主体构图，**第 6 段 16:9**；提示词语言全英文，**正文不描写外貌、不点名颜色**（颜色全部由 PICTURE 1 承担，见 Segment 05 明文规定）。
- 词作：李清照《鹧鸪天·桂花》。BPM = 121.5；细拍 fine beat ≈ 0.494s；大身韵慢拍 ≈ 0.988s（约 60.75 BPM 的大结构，121.5 细网格驱动脚步 / 腕 / 袖 / 相机 accent）。
- 最终六段可直接粘贴的提示词全文：[`prompt.md`](./prompt.md)。

## The problem（问题：长片不能一次生成，也不能切成六支独立舞）

单段生成有时长上限，75 秒必须分段；但朴素地「切成六段、每段各写一支舞」会同时踩三个坑：

1. **每段都从头起舞**——开场站桩、起手式、中性站姿复位，拼回去全是重启的断点；
2. **每段各喂一张图、各写一套外貌**，身份与服装在极端身段（腾空、贴地、回头）里逐段漂移；
3. **每段都把最强动作做成同一类**（旋转 / 大跳），六段看下来是重复炫技，没有能量曲线。

判别的钥匙仍然是音频：这是固定节拍（121.5 BPM 细网格 + 0.988s 大拍）而非自由散板，把 ★ 强拍 / ◆ 次点与身段、相机对账，能清楚看到「连续身韵句 + 踩拍的离散相机重构图」，而不是六支各自为政的舞。

## The breakthrough（突破：四件事写死）

### 1. 双图分工，正文零外貌

- `<PICTURE 1>` = 人物 **和** 环境的唯一视觉权威；`<PICTURE 2>` 仅三视图，用来在极端姿势下稳住身份与身体结构，**明文禁止**它提供环境 / 灯光 / 首帧 / 机位 / 服装设计。
- 六段每段都重复同一套图权威声明（各段独立生成，不能假设模型记得上段）。Segment 05 更进一步：**全文不点名任何颜色**，颜色关系全交给 PICTURE 1。
- token 预算全部投给模型最容易做错的「时间纪律 + 身韵因果 + 相机状态机」。

### 2. 分段 AUDIO 1 是唯一时间轴，音乐只续不重启

- 每段用**段内局部时钟**（从 00:00 起），只在段头标注它覆盖的源曲全局区间；
- 第 2 段起明文 `The music CONTINUES … does NOT restart`；
- ★ / ◆ 两级卡点、唱词进出时间逐段列死，`Do not invent another rhythm`。

### 3. hand-off continuation state：段末状态 = 下段首帧

- 除末段外，每段结尾写一个可被下段第一拍接住的**物理状态**（重心偏向、哪只脚准备接重、躯干斜向、哪只袖还带残余动量、头跟随的方向）；
- 下段开场 `[00:00]` **复述同一状态**直接起新步，明文 `Do not reset / Do not return to neutral / Do not restart`；
- **第一段第一帧就已经在舞**（small gliding step into the phrase），杜绝开场站桩。

### 4. 身韵是一条因果链，水袖 / 裙摆只跟身、不自己动

```
feet → weight transfer → pelvis → waist → torso → shoulder → arm → wrist → fingertips → sleeve
```

下身（脚 / 重心 / 胯）是发动机，躯干延迟跟随，**头与眼神最后到**（「身去而目留」）；水袖与大摆裙严格按「腰头 → 内层 → 中层 → 外缘」的惯性滞后响应，`never move independently`。不字面演歌词、无 lip-sync，歌词只影响情绪重量与句读。

## The named language（命名）

> **Six-Segment Song-Style Shen-Yun with Hand-Off Hero Ladder（六段宋韵身韵 · 交接式英雄阶梯）**

「Hand-Off」指段与段之间用物理状态无缝交接；「Hero Ladder」指五个高光被排成一条**机制互不重复、能量有呼吸**的阶梯，而不是每段重复同一个大招。

## 六段总览（全局区间可复算，严丝合缝）

| 段 | 源曲全局区间 | 段内时长 | 角色 / 能量 | 唱词 |
|---|---|---|---|---|
| 01 | 00:00.000–00:12.410 | 12.41s | 首帧即舞，建立身韵本体；两个**脸部** Hero | 暗淡轻黄体性柔 / 情疏迹远只香留 |
| 02 | 00:12.410–00:25.610 | 13.20s | 自信增长；首个高难技术 Hero（刃裙） | 何须浅碧何须深红 / 第一流×2 |
| 03 | 00:25.610–00:38.070 | 12.46s | 器乐 release，最深身体释放；空中换腿 | （纯器乐，无唱词） |
| 04 | 00:38.070–00:50.533 | 12.463s | 爆发后的呼吸，难度转为单腿平衡控制 | 梅定妒（10.563 进） |
| 05 | 00:50.533–01:02.500 | 11.967s | 再度外扩；位移型腾空换腿 | 菊应羞（0.100 进） |
| 06 | 01:02.500–01:15.468 | 12.968s | 把速度转成贴地旋转，肩背轴风车，绝对静止收尾（16:9） | 何事当年不见收 / 不见收不见收 |

> 区间由各段时长依次相加得到，且 Seg04 结束 38.070+12.463 = **50.533** 与 Seg05 标注起点、Seg05 结束 50.533+11.967 = **62.500** 与 Seg06 标注起点完全吻合，可作为分段时间轴的自检。

## Hero 谱系（关键：每段机制刻意不重复）

| 段 | 时间 | Hero 机制 | 明确排除（NOT …） |
|---|---|---|---|
| 01 | 04.439 / 08.389 | 脸部 Hero：首次脸部定格；「身去而目留」turn-back | 不做高难技术 Hero、不做脸部大特写 |
| 02 | **03.880** | **BLADE-SKIRT**：压缩→空中变向 + 不对称腿线 + 大摆裙旋成圆盘如刀轮（腰为轴，内层先动外缘滞后） | 非简单旋转 / 抬腿 / 转裙 / 单个跳跃；非原地转 |
| 03 | **06.483** | **AIRBORNE CROSS-DIRECTION**：单脚蹬地斜向腾空，**空中换腿** mid-air leg switch，不对称剪影 | **不复刻 Seg02 旋裙圆盘**；非单手地板转、非简单跳 |
| 04 | **05.875** | **SINGLE-AXIS BODY SPIRAL**：单腿支撑低重心、自由腿贴地低划、躯干反向、单轴 180° 的平衡控制 | 非腾空 Hero、非地板转、非大型旋转特技 |
| 05 | **09.214** | **HIGH-SPEED TRAVELLING LEAP / AIRBORNE LEG EXCHANGE**：位移型腾空、空中剪刀式换腿、对侧脚落地缓冲 | 非 720° 单手地板转、非原地站立转、非悬浮 |
| 06 | **09.099** | **SHOULDER-BACK-AXIS WINDMILL → FISH-LIKE LOW SWEEP**：肩 / 上背贴地为轴的低位风车扫腿，剪刀腿大圆，裙摆贴地铺成圆形花 | 非单手 720° 地板转、非倒立、非腾空；全场仅一个地板 Hero |

每个技术 Hero 之后都接一个**右前低位仰拍三-quarter 脸部 Hero**（Seg02 07.831、Seg03 10.434、Seg04 09.825、Seg05 11.190、Seg06 11.650），且 Seg06 用 11.650–12.968 的 **ABSOLUTE STATIC FINAL FRAME** 收束全舞——「前 75 秒最终落在一个画面里」。

## 卡点总表（★ 强拍 / ◆ 次点，段内本地时间）

- **Seg01** ★ 0.488 / 4.439 / 8.389 / 12.340　◆ 2.463 / 6.414 / 10.365
- **Seg02** ★ 3.880 / 7.831 / 11.782　◆ 1.905 / 5.856 / 9.806
- **Seg03**（器乐）◆ 0.557　★ 2.532　◆ 4.508　★ 6.483　◆ 8.458　★ 10.434　◆ 12.409
- **Seg04** ★ 1.924　◆ 3.899　★ 5.875　◆ 7.850　★ 9.825　◆ 11.801（唱词「梅定妒」10.563）
- **Seg05** ★ 1.313 / 5.264 / 9.214　◆ 3.288 / 7.239 / 11.190（唱词「菊应羞」0.100）
- **Seg06** ★ 本地 1.198 / 5.148 / 9.099（全局 01:03.698 / 01:07.648 / 01:11.599）　◆ 3.173 / 7.124 / 11.074（全局 01:05.673 / 01:09.624 / 01:13.574）

> Seg04 原始 beat map 存在时间乱序，提示词内已自行纠正为时间顺序（1.924→3.899→5.875→7.850→9.825→11.801）再执行。

## 相机语言（服务动作，不抢戏）

- 统一**角色主导中近景**：下小腿 / 脚踝上方 → 头顶略上，人物占画面主体；不依赖远景、不为展示环境把人缩小。
- 招牌脸部角度 = **右前方、明显低于眼平、向上仰拍的三-quarter 脸**，只在身段自然给出 payoff 时用，不每个强拍都插。
- 相机只有两种被允许的行为：**被重心转移 / 变向 / 腾空 / 落地 / 袖展开 / 脸部揭示触发的极短 editorial burst**，与到达构图后的 **ZERO / ABSOLUTE HARD LOCK**；不全程环绕、不连续电影感 zoom、不无故漂移。
- Seg06 地板 Hero 的 09.099–09.700 是全段最重要的一次硬锁——**让舞者自己制造冲击，相机不许动**。

## 48-hour takeaway（可复用清单）

1. **长片分段先算全局区间**：各段时长相加必须首尾相接（本案例 6 段正好 0 → 75.468，Seg04/05、Seg05/06 边界互证），切点放在气口、不切在爆发中间。
2. **每段都重复图权威 / 构图 / 相机纪律 / negative**：各段独立生成，不能依赖「上段说过」；外貌与颜色交给图，正文一个字不重写。
3. **音乐只续不重启、时钟每段归零**：段内时间从 00:00，段头标全局区间；固定节拍才写 BPM，自由散板只写 `on the strong accent`。
4. **hand-off 是物理状态不是镜头语言**：写清重心 / 脚 / 躯干 / 袖尾 / 头方向，下段开场逐字复述；第一段第一帧就要在运动里。
5. **Hero 排成阶梯且互相排除**：每段一个新机制，并明文 `NOT <上一段机制>`，技术 Hero 后配一个脸部 Hero，能量有爆发也有控制段。
6. **身韵因果链不可倒置**：脚→重心→胯→腰→躯干→肩→腕→指尖→袖，头眼最后；面料永远滞后身体、绝不自主运动。

本案例是配套反推技能 [`video-to-h3-prompt`](https://github.com/LoveRain1997/video-to-h3-prompt) 中「多图 all-purpose 锁定 + 长片分段 SOP（references/08）」「片型分流与相机状态机（references/06）」「音头卡点（references/07）」的正向创作产物。

## The final prompt

六段完整可粘贴提示词见 [`prompt.md`](./prompt.md)，每段一个代码块、可逐段复制。统一骨架：

1. **INPUT AUTHORITY**：PICTURE 1 唯一视觉权威 / PICTURE 2 三视图澄清 / AUDIO 1 唯一时间权威
2. **CORE DIRECTING LANGUAGE**：身韵动力链、快慢双网格、不字面演歌词、本段情绪
3. **AUDIO 1 TIMING**：BPM、★ / ◆、唱词区间（分段续接声明）
4. **CONTINUATION STATE / FRAMING / CAMERA LANGUAGE**：hand-off、中近景、右前低位仰拍
5. **逐时间块**：每个 ★ / ◆ 的身段因果 + 相机 burst / lock
6. **SIGNATURE HERO**：压缩→发力→空中 / 地面机制→落地 / 回收（含裙 / 袖物理）
7. **BODY MECHANICS / EDITORIAL / PERFORMANCE / VISUAL CONSISTENCY / FINAL READ / NEGATIVE**

## Result（成片要点）

- 6 段拼接为一条连续身韵句，无站桩开场、无段间重置、音乐一气呵成；
- 人物身份 / 服装 / 环境跨 6 段稳定，极端身段不漂移；
- 五个高光机制依次为：刃裙旋盘 → 空中换腿 → 单轴控制 → 位移腾跃 → 贴地风车，无一重复；
- 每个技术高光后落到右前低位仰拍脸部 Hero，终段以肩背轴风车 + 裙摆铺地圆花 + 绝对静止终帧收束。

## Assets

| 文件 | 作用 |
|---|---|
| [`prompt.md`](./prompt.md) | 六段最终有效提示词全文（每段可单独粘贴） |

> 生成时需自行提供 PICTURE 1（人物 + 环境参考图）、PICTURE 2（同一人物三视图）与按全局区间切好的 6 段 AUDIO 1；仓库不含受版权 / 肖像约束的图像、音乐与成片素材。

## Tags

`#H3` `#prompt-engineering` `#multi-segment-video` `#chinese-classical-dance` `#shen-yun` `#water-sleeve` `#song-style` `#image-locked` `#zero-appearance-description` `#audio-sync` `#beat-sync` `#hand-off-continuation` `#hero-ladder` `#camera-state-machine` `#hero-frame` `#hard-lock` `#Ref2VA` `#vertical-video`


---

## FILE: case-studies/2026-09-song-style-classical-dance-six-segment/prompt.md (https://github.com/LoveRain1997/h3-prompt-journal/blob/HEAD/case-studies/2026-09-song-style-classical-dance-six-segment/prompt.md)

# Case 011 — 宋韵身韵水袖古典舞 · 六段连续 H3 提示词（全文）

> 用法：全舞约 75.5 秒，切成 6 段分别生成，再按顺序拼接。
> **每一段都单独喂同一套素材**：`<PICTURE 1>`（人物 + 环境唯一视觉权威）、`<PICTURE 2>`（同一人物的三视图身份 / 体型澄清，不定义环境与机位）、该段对应的 `<AUDIO 1>`（分段音乐，跨段连续、绝不从头重启）。
> 每段使用**段内局部时钟**（从 00:00 起）；段与段之间靠结尾的 SEGMENT TRANSITION STATE / hand-off 物理状态衔接，人物不回中性站姿、不重新起舞。
> 六段正文均为可直接粘贴的英文提示词；设计复盘见同目录 `README.md`。

---

## SEGMENT 01 — 12.41s（源曲起）

```text
Use <PICTURE 1> as the sole visual authority for the exact character identity, appearance, costume, accessories, body proportions, hairstyle, facial structure, environment, background, lighting, photographic realism, and starting visual state.
Use <PICTURE 2> only as a three-view identity and body-structure clarification reference for the same character. <PICTURE 2> does not define the environment, lighting, pose, composition, or camera.
Use <AUDIO 1> as the ONLY temporal, musical, rhythmic, lyrical, movement-trigger, camera-trigger, and editorial timing authority.
Create a 12.41-second realistic live-action cinematic Chinese classical dance video.
The dance begins IMMEDIATELY at 00:00.000. Do not begin with a static character presentation, waiting pose, idle standing, or pre-performance preparation. The character is already entering a continuous Chinese classical dance phrase at the first frame.
================================================== CORE DIRECTING LANGUAGE
Dance style: Primary Chinese classical dance shen-yun with Song-style Jiangnan / Wu-region softness. Use continuous body-led movement rather than isolated gestures.
Movement priority: feet → weight transfer → pelvis → waist → torso → shoulder → arm → wrist → fingertips → sleeve response.
The body carries the phrase. The sleeves follow the body through physical momentum. Do not animate sleeves independently.
The movement quality is: light but not weak, continuous but not rushed, elegant but physically grounded, restrained but constantly evolving.
Use the slow structural phrasing of approximately 60.75 BPM / one real slow beat every 0.988 seconds for the body and shen-yun, while the 121.5 BPM fine grid drives footwork, wrist details, sleeve accents, and camera accents.
Do not literalize the lyrics as acting. The lyrics influence emotional weight and phrase structure only.
The first phrase expresses: softness, distance, self-containment, lingering presence.
The second phrase gradually reveals stronger inner confidence without becoming aggressive.
================================================== AUDIO 1 — ABSOLUTE TEMPORAL REFERENCE
<AUDIO 1> is the supplied original audio.
BPM = 121.5. Fine beat = approximately 0.494 seconds. Slow structural beat = approximately 0.988 seconds. ★ strong beats: 00:00.488 00:04.439 00:08.389 00:12.340
◆ secondary accents: 00:02.463 00:06.414 00:10.365
Lyric timing: 00:00.666–00:04.566: “暗淡轻黄体性柔”
00:04.666–00:09.650: “情疏迹远只香留”
Do not invent another rhythm. Do not shift major movement events away from the supplied AUDIO 1 timing. All choreography and camera events below are synchronized to these exact local times.
================================================== FRAMING PRINCIPLE
Keep the character visually dominant for the entire segment.
Primary framing: medium-close cinematic coverage from approximately the lower calf / lower leg area to just above the head.
The character should occupy most of the frame. Preserve enough vertical space for leg mechanics, torso movement, face, and sleeve trajectories without wasting pixels on empty surroundings.
Do not rely on distant wide shots. Do not make the character small in frame.
The preferred facial hero angle is: camera positioned at the character's front-right and clearly below eye level, looking upward toward the face.
Use this angle only when the choreography naturally creates a facial payoff. Do not insert it mechanically at every strong beat.
================================================== SHOT 01 — IMMEDIATE DANCE ENTRY 00:00.000–00:00.488
Begin directly in motion.
The character takes a small gliding step into the phrase. The supporting foot carries the weight first. The pelvis follows. The torso remains quiet for a fraction of a beat. The shoulder and existing sleeve follow afterward.
The first arm phrase is restrained: one arm begins near the body's line, the wrist gradually turns, the fingertips release last, and the sleeve follows with slight inertial delay.
The opposite side of the body begins preparing the next weight transfer before the first gesture fully finishes.
CAMERA: Start in a medium-close three-quarter frontal composition. Camera is slightly below the character's natural chest level, positioned toward the front-right side. Very subtle lateral tracking follows the first gliding step.
Do not push into the face yet. The camera should feel as though it has entered an already-living dance phrase.
At 00:00.488 the first strong beat arrives.
================================================== 00:00.488 ★ FIRST STRONG BEAT
The body completes its first full shen-yun opening exactly on the ★.
ACTION: The supporting leg receives the weight. The opposite foot glides outward. Pelvis turns slightly. Waist follows. Chest opens only after the lower body has begun the turn. The leading arm opens through the shoulder. The wrist turns last.
The movement creates one continuous curved line from supporting foot through waist, shoulder, wrist, and fingertips.
The sleeve follows the arm a fraction behind the hand, creating a soft trailing arc.
CAMERA: At the exact ★ accent, make a short editorial camera burst slightly upward and forward while retaining the front-right three-quarter relationship.
Finish the reframing immediately after the accent. The camera then returns to controlled movement following the body.
================================================== 00:00.666–00:02.463 LYRIC PHRASE 01 — “暗淡轻黄体性柔”
Do not act out the literal words.
Use a restrained traveling phrase.
00:00.666: The character begins a diagonal gliding travel.
Footwork: small, precise, quiet, continuous.
The lower body moves faster than the upper body. The torso appears to float because the weight transfer is smooth rather than because the body is physically static.
The character makes a shallow inward spiral through the waist.
The leading hand passes across the body's front line. The wrist redirects outward. The sleeve follows.
00:01.20–00:01.90: The character continues traveling without resetting the previous gesture.
The trailing foot closes only partially, immediately becoming the next supporting foot.
At approximately 00:02.10: the torso begins a gentle directional change.
================================================== 00:02.463 ◆
Secondary phrase point.
ACTION: Complete the directional change through: foot → knee → hip → waist → shoulder.
The head does NOT turn first.
The head follows the torso slightly late.
One sleeve folds inward as the other continues outward, producing an asymmetrical flowing shape.
CAMERA: Perform a small lateral tracking adjustment synchronized to the ◆. Maintain medium-close framing. Keep both the lower legs and face readable.
Do not turn this into a dramatic zoom.
================================================== 00:02.463–00:04.439
The dance gains a little more amplitude.
The character travels through two consecutive weight transfers without stopping.
First transfer: outer foot receives weight, waist rotates, leading sleeve sweeps across the body.
Second transfer: the opposite foot receives weight, torso opens, the first sleeve decelerates while the second sleeve catches up.
The result must read as ONE continuous phrase rather than two separate poses.
00:03.40–00:04.20: The character gradually redirects toward the camera's front-right side.
The torso arrives first. The head arrives later.
This prepares the first facial payoff.
================================================== 00:04.439 ★ FIRST FACIAL HERO ACCENT
This is the first important character-image moment of the segment.
It is NOT a high-difficulty Hero Moment. It is a facial / compositional Hero Moment.
ACTION: The character completes the weight transfer. The torso opens from a gentle inward curve into a longer upright line. The nearer shoulder settles. The wrist finishes its arc. The sleeve reaches its widest natural extension.
Only after the body settles: the face turns approximately three-quarters toward the camera.
The eyes settle toward the lens. The expression remains serene and self-possessed. No exaggerated performance expression.
CAMERA: Use the preferred fixed character angle: FRONT-RIGHT LOW CAMERA → UPWARD THREE-QUARTER FACIAL VIEW.
The camera is clearly below the face. The upward perspective should make the face feel sculptural and elegant.
Maintain the medium-close body framing: lower legs remain inside the frame, head remains close to the upper frame boundary, the character fills the image.
Make a short, decisive editorial reposition into this composition immediately around the ★. After reaching the composition, hold the camera almost completely still for a brief Hero Frame.
The character continues only with minimal natural body and sleeve settling. Do not turn this into a face close-up.
================================================== 00:04.666–00:06.414 LYRIC PHRASE 02 — “情疏迹远只香留”
The emotional quality shifts from softness toward distance.
ACTION: Do not walk directly toward the lens.
Instead, the character begins moving laterally across the visual field.
The feet remain light and continuous. The hips and shoulders travel in slightly different timing.
The torso leads away from the camera. The head remains behind for a moment. The eyes leave the lens last.
This creates: body departing → shoulder departing → head delaying → gaze lingering.
The sleeve follows the departure with visible physical lag.
At approximately 00:05.40: the character makes a small change of direction through the supporting foot rather than through a sudden upper-body turn.
At approximately 00:05.90: the opposite arm opens.
At approximately 00:06.20: the body prepares another diagonal change.
================================================== 00:06.414 ◆
Secondary accent.
ACTION: Complete the diagonal change.
The supporting foot receives the body. The opposite leg extends lightly. The pelvis turns first. The torso folds slightly over the supporting side. The shoulder follows. The head remains calm.
The sleeve creates a small circular flourish at the wrist.
This circular sleeve motion must originate from the wrist and shoulder, not from independent fabric animation.
CAMERA: Track the character laterally through the accent. Keep the camera slightly low and front-right relative to the body. Keep the lower legs and face visible together.
The camera movement must resolve immediately after the accent instead of drifting.
================================================== 00:06.414–00:08.389
Increase the dance's spatial confidence.
The character executes a continuous sequence:
gliding step → immediate weight transfer → shallow turn → opposite-side recovery step → torso spiral → sleeve release.
Do not reset between actions.
The lower body remains continuously active.
The torso should appear calm because the movement is coordinated, not because it is frozen.
From 00:07.40 onward: the character gradually redirects the torso back toward the camera.
The feet are already moving in the new direction before the head turns.
================================================== 00:08.389 ★ SECOND STRONG BEAT
This is the second major visual accent.
The phrase “只香留” should be represented through lingering movement rather than literal acting.
ACTION: The character executes a controlled turn-away and delayed return.
Sequence: supporting foot plants → pelvis redirects → torso follows → shoulders rotate → trailing sleeve opens → head remains turned away → head returns last → eyes reconnect toward the camera direction.
The final moment of the turn should feel as though the body has already moved on while the gaze briefly remains behind.
CAMERA: Follow the turn through a controlled short tracking arc. Do not make a full orbit.
During the final head return, shift naturally into the preferred front-right low-angle relationship.
At the instant the face reconnects: camera movement slows sharply and settles.
This becomes the second facial payoff of the segment.
================================================== 00:08.389–00:10.365
After the ★, do not stop.
The character immediately continues through another traveling phrase.
Use: small rapid foot repositioning → weight transfer → waist redirection → shoulder release → sleeve continuation.
The energy is slightly stronger than the opening but remains elegant.
The upper body never becomes disconnected from the feet.
At approximately 00:09.20: the character takes a longer diagonal step.
At approximately 00:09.60: the body passes through a shallow side inclination.
At approximately 00:10.00: the supporting leg receives the weight and the torso begins returning upright.
================================================== 00:10.365 ◆
Final secondary accent.
ACTION: The character closes the current movement sentence through: foot placement → knee absorption → pelvis settling → torso lengthening → shoulder release → wrist completion.
One sleeve descends first. The other sleeve follows slightly later.
The face remains calm.
CAMERA: Return to a slightly wider version of the same medium-close framing while keeping the lower legs and head fully readable.
Do not pull into a distant shot.
================================================== 00:10.365–00:12.340
The final phrase of Segment 01 must feel like continuation, not a conclusion.
The character moves through one final traveling curve.
The feet continue moving. The body gradually changes direction toward the movement path required by Segment 02.
Use: gliding step → subtle cross-step → weight transfer → waist turn → chest release → sleeve trailing behind → partial recovery.
At approximately 00:11.40: the character passes through a three-quarter profile.
At approximately 00:11.80: the head begins turning toward the next movement direction.
Do not stop the body first and then turn the head. The head follows the already-moving body.
================================================== 00:12.340 ★ FINAL STRONG BEAT
This is the Segment 01 transition accent.
Do not create a large final pose.
ACTION: Complete the current directional change exactly on the ★.
The supporting leg takes the weight. The opposite leg is already preparing the next step. The torso settles into a slight diagonal. One sleeve reaches the end of its arc. The other sleeve is beginning the next movement.
The character's gaze shifts toward the next travel direction.
CAMERA: Use a brief front-right low-angle relationship one final time, but do NOT turn this into another face close-up.
Keep the same medium-close body coverage.
The camera should briefly stabilize on the completed body line, then preserve the direction of travel into the next segment.
================================================== 00:12.340–00:12.410 SEGMENT TRANSITION STATE
Do NOT reset.
Do NOT return to neutral standing.
Do NOT introduce a new starting pose.
The character remains in active dance continuity.
At exactly 00:12.410 the physical state should be:
weight already transferred toward the next travel direction, torso slightly diagonally oriented, one foot ready to continue, the sleeve tails still carrying faint residual motion, head beginning to follow the next movement direction.
Segment 02 must be able to begin directly from this physical state.
================================================== CAMERA DIRECTING RULE
Camera movement must always be motivated by the choreography.
Primary camera language: medium-close, front-right three-quarter, slightly low perspective, short lateral tracking, small controlled tracking arcs, brief decisive editorial reframing at meaningful musical accents.
Preferred facial angle: front-right low camera looking upward.
Use the low-angle facial relationship especially around: 00:04.439 ★ 00:08.389 ★
Do not repeat it at every beat.
The camera must never become more important than the dance.
Every camera change should correspond to: a weight transfer, a directional change, a completed turn, a sleeve expansion, a facial reveal, or a strong musical accent.
================================================== BODY MECHANICS
Every movement follows a physically believable kinetic chain.
Footwork creates the movement. Weight transfer creates direction. Pelvis initiates rotational change. Waist carries the shen-yun. Torso follows with slight delay. Shoulders release after the torso. Arms follow the shoulder. Wrists complete the gesture. Fingertips finish the line. Sleeves react to the completed movement with natural inertial lag.
The upper body must never appear disconnected from the legs.
Do not use floating or zero-weight movement.
Do not reset between individual gestures. Each action must provide momentum for the next action.
================================================== EDITING
The dance remains one continuous choreographic sentence.
Editorial reframings occur only at musically meaningful timestamps.
Preferred editorial accents: 00:00.488 00:04.439 00:08.389 00:12.340
Cuts or camera bursts must preserve movement direction and body continuity.
Do not cut during an unresolved weight transfer. Complete the visual action on the beat before changing composition.
================================================== PERFORMANCE
Emotional progression:
00:00–04.439 quiet elegance and restrained softness.
04.439–08.389 slightly stronger self-possession.
08.389–10.365 increasing visual confidence.
10.365–12.410 graceful withdrawal into the next phrase.
The face remains calm throughout. No lip-sync. No singing. No literal acting of lyric words.
================================================== VISUAL CONSISTENCY
Preserve the exact character and the exact visual world established by <PICTURE 1>.
Do not redesign the character. Do not reinterpret the costume. Do not replace the background. Do not change the visual identity of the character.
<PICTURE 2> is only an identity clarification reference.
All environmental and photographic information comes from <PICTURE 1>.
The result must read as: IMMEDIATE DANCE ENTRY → CONTINUOUS SONG-STYLE SHEN-YUN → 00:04.439 FACIAL HERO ACCENT → LATERAL DISTANCE / LINGERING GAZE → 00:08.389 TURN-BACK HERO ACCENT → CONTINUOUS TRAVEL → 00:12.340 TRANSITION ACCENT → DIRECT PHYSICAL CONTINUATION INTO SEGMENT 02.
```

> 第一段导演备注（不进提示词）：第一段直接从舞蹈开始，核心不是“古风动作拼接”，而是 **脚步—重心—身韵—袖—眼神** 连续推进；镜头始终服务动作，只在 4.439s、8.389s 等真正有视觉意义的位置进入仰拍 / 英雄构图。4.439s 负责第一次脸部 Hero，8.389s 负责“身去而目留”的宋式韵味，12.340s 把身体状态送进第二段。

---

## SEGMENT 02 — 13.20s

```text
Use <PICTURE 1> as the sole visual authority for the exact character identity, appearance, costume, accessories, body proportions, hairstyle, facial structure, environment, background, lighting, photographic realism, and the visual continuation state inherited from Segment 01.
Use <PICTURE 2> only as a three-view identity and body-structure clarification reference for the same character. <PICTURE 2> does not define environment, lighting, pose, composition, or camera.
Use <AUDIO 1> as the ONLY temporal, musical, rhythmic, lyrical, movement-trigger, camera-trigger, editorial timing authority.
Create a 13.20-second realistic live-action cinematic Chinese classical dance video.
This segment MUST begin as the physical continuation of Segment 01.
Do not reset the character to a neutral stance.
Do not create a new starting pose.
Do not pause before dancing.
The body must already be carrying the momentum inherited from Segment 01 at 00:00.000.
==================================================
CORE DIRECTING LANGUAGE
=======================
Primary dance language:
Chinese classical dance shen-yun with Song-style Jiangnan / Wu-region elegance.
The movement must evolve from Segment 01 rather than restart.
Segment 01 established:
softness,
distance,
lingering gaze,
controlled traveling movement.
Segment 02 develops this into:
self-possession,
greater spatial confidence,
faster footwork,
larger body arcs,
stronger weight transfer,
and a visible sense of inner superiority without aggression.
The choreography is one connected kinetic sentence.
Movement chain:
feet
→ weight transfer
→ pelvis
→ waist
→ torso
→ shoulder
→ arm
→ wrist
→ fingertips
→ sleeve response
→ continuation into the next foot placement.
The lower body must remain continuously active.
Do not perform isolated arm gestures on a stationary body.
Do not perform isolated spins.
Do not use simple leg lifts as major movement events.
Every action must generate the momentum required by the next action.
Use approximately 60.75 BPM for large body phrasing and approximately 121.5 BPM fine-grid accents for footwork, wrist changes, sleeve responses, camera accents, and editorial events.
==================================================
AUDIO 1 — ABSOLUTE TEMPORAL REFERENCE
=====================================
<AUDIO 1> is the supplied original audio.
BPM = 121.5.
Fine beat = approximately 0.494 seconds.
Slow structural beat = approximately 0.988 seconds.
Segment 02 local timing:
★ strong beats:
00:03.880
00:07.831
00:11.782
◆ secondary accents:
00:01.905
00:05.856
00:09.806
Lyric timing:
00:00.256–00:03.606
“何须浅碧何须深红”
00:04.256–00:08.123
“第一流”
00:08.223–00:13.040
“第一流” repeated
All movement and camera events MUST synchronize to these exact local timestamps.
Do not invent independent musical timing.
==================================================
CONTINUATION STATE FROM SEGMENT 01
00:00.000
=========
At the exact beginning of Segment 02:
The character is already moving from the state established at the end of Segment 01.
Weight is already biased toward the next travel direction.
The torso remains slightly diagonal.
One foot is prepared to receive or redirect weight.
The previous sleeve phrase still carries faint residual motion.
The first action of Segment 02 begins from this exact moving state.
Do not reset the hips.
Do not square the shoulders.
Do not return both feet to parallel.
Do not freeze the sleeves.
==================================================
FRAMING PRINCIPLE
=================
Keep the character visually dominant.
Primary framing throughout the segment:
medium-close cinematic coverage,
approximately from lower calf / lower-leg region to just above the head.
The character should occupy most of the frame.
Always preserve:
lower-leg readability,
full torso readability,
face readability,
and enough space for sleeve and skirt trajectories.
Do not use distant establishing shots.
Do not shrink the character to display empty surroundings.
The camera may change angle and direction aggressively while remaining within this medium-close visual scale.
Preferred facial Hero angle:
camera positioned toward the character's front-right,
clearly below eye level,
looking upward toward a three-quarter facial angle.
Use this only when the choreography creates a natural facial payoff.
==================================================
00:00.000–00:01.905
IMMEDIATE CONTINUATION + BUILD
==============================
Begin directly in motion.
The character continues the directional flow inherited from Segment 01.
00:00.000–00:00.494:
The prepared foot receives weight.
The opposite foot glides through.
The pelvis follows the travel direction.
The torso remains slightly delayed.
The nearer shoulder begins opening.
The previous sleeve finishes its residual arc while the opposite sleeve begins the next phrase.
00:00.494–00:00.988:
Increase footwork speed.
Use two light consecutive gliding steps.
The second step must begin before the first gesture has visually reset.
The hips change direction first.
The torso follows.
The head follows later.
00:00.988–00:01.482:
Introduce a small diagonal crossing step.
The crossing foot does not simply close.
It redirects the body toward a new diagonal.
The waist begins a restrained spiral.
One sleeve traces a narrow inward curve.
The opposite sleeve opens outward.
00:01.482–00:01.905:
Prepare a stronger change of level.
The supporting knee absorbs downward.
The pelvis lowers slightly.
The chest remains lifted.
The opposite foot becomes the next directional trigger.
The movement must look prepared by the body rather than suddenly inserted.
CAMERA:
Use controlled lateral tracking in the character's travel direction.
Stay medium-close.
Keep the camera slightly below the torso.
Do not push into the face.
==================================================
00:01.905 ◆
===========
Secondary musical accent.
ACTION:
Complete the level change exactly on the ◆.
Supporting foot receives the body.
Pelvis drops then redirects.
The torso follows with a delayed curve.
The leading arm crosses the body's center line.
The wrist turns outward at the exact end of the beat.
The sleeve follows the wrist with natural inertial lag.
The opposite foot immediately begins the next step.
No pause after the accent.
CAMERA:
Perform a short, decisive lateral camera burst synchronized to the ◆.
Reframe slightly toward the character's front-right side.
Do not zoom aggressively.
At the end of the burst, camera motion immediately settles.
==================================================
00:01.905–00:03.606
LYRIC PHRASE — “何须浅碧何须深红”
=========================
Do not literalize the lyric.
The phrase expresses:
confidence without competition,
beauty without deliberate display,
controlled superiority.
ACTION:
The character now travels with noticeably greater spatial confidence.
Footwork:
rapid short glide
→ immediate weight transfer
→ crossing step
→ directional release
→ another supporting step.
The body stays vertically composed while the feet move faster.
The pelvis initiates each directional change.
The torso should remain fluid rather than rigid.
00:02.20–00:02.80:
Introduce an asymmetric sleeve phrase.
The nearer sleeve closes toward the body while the far sleeve opens.
The hands do not mirror each other.
The body turns subtly through the waist.
00:02.80–00:03.40:
Increase stride length.
One step travels diagonally forward-right.
The next redirects slightly across the previous pathway.
The torso should visibly lag the feet for a fraction of a beat.
At approximately 00:03.30:
the body begins sinking into the preparation for the first major Hero.
The supporting knee bends.
The pelvis lowers.
The chest remains open.
Both arms begin gathering the sleeve mass closer to the waist.
This is a physical compression phase.
==================================================
00:03.880 ★
SIGNATURE HERO MOMENT 01 — BLADE-SKIRT DANCE
============================================
This is the first true high-difficulty Hero Moment of Segment 02.
The action must NOT read as:
simple spinning,
simple leg lifting,
simple skirt twirling,
or a single isolated jump.
The full sequence is:
RAPID TRAVEL
→ LOW COMPRESSION
→ CROSS-STEP REDIRECTION
→ BODY FOLD
→ EXPLOSIVE TAKEOFF
→ AIRBORNE DIRECTION CHANGE
→ ASYMMETRIC LEG LINE
→ BLADE-SKIRT DEPLOYMENT
→ CONTROLLED LANDING
→ IMMEDIATE CONTINUATION.
---
## 00:03.25–00:03.72 PRE-HERO COMPRESSION
The character takes the final two rapid steps into the Hero.
The first foot cuts diagonally across the travel path.
Immediately transfer weight.
The opposite foot arrives underneath the center of mass.
The pelvis drops.
The torso folds slightly forward and diagonally.
Both arms compact toward the body's center.
The skirt mass remains attached naturally at the waist and begins receiving rotational momentum from the body's turn.
The final preparation must visibly load the legs and core.
---
## 00:03.72–00:03.880 FINAL TAKEOFF PREPARATION
The supporting foot drives force into the floor.
The knee straightens explosively.
The pelvis rises and rotates.
The torso does not jump independently.
The chest follows the lower body's rotational impulse.
One shoulder leads.
The opposite shoulder remains delayed.
One arm begins opening.
The opposite arm remains compact for a fraction of a beat.
---
## 00:03.880 ★ HERO STRIKE
At the exact ★:
The character launches into a short airborne directional exchange.
The body does not simply spin in place.
The movement travels diagonally.
During the airborne phase:
the hips change direction first,
the torso folds and opens through the middle of the motion,
one leg extends along a strong diagonal,
the other leg remains bent and draws through,
then the extended leg changes its line before landing.
The body therefore creates two different silhouettes within one continuous airborne phrase.
The visual center of the Hero is the combination of:
airborne directional change
+
asymmetric leg geometry
+
large waist-driven skirt deployment.
---
## BLADE-SKIRT GEOMETRY
The full-circle skirt remains physically fixed at the waist.
The dancer's rotational and lateral momentum deploys the skirt into a broad circular plane.
During the airborne directional exchange, the skirt opens into a large disc-like ring.
The ring should visually resemble a fast-moving blade or cutting wheel sweeping through space.
The fabric is not independently animated.
Its motion originates from:
pelvic rotation
→ waistband rotation
→ inner skirt layer
→ middle layers
→ outer hem.
The inner layers respond first.
The outer hem follows with greater inertial delay.
The skirt should form a clean, continuous radial geometry around the waist.
The dancer's body remains in the central opening of the ring.
No limb should intersect the outer fabric ring.
The extended leg moves through the hollow center.
The skirt is the visual amplifier of the body's movement, not a separate performer.
---
## HERO CAMERA
Immediately before the ★:
CAMERA = controlled tracking in the existing travel direction.
At the exact 00:03.880 ★:
perform a short high-velocity editorial camera burst following the airborne directional exchange.
The burst is lateral-forward and slightly upward.
Do NOT perform a full orbit.
Do NOT rotate the camera continuously around the character.
The camera's purpose is to reveal the blade-like skirt geometry and the asymmetric airborne body line within one medium-close composition.
The character must remain large in frame.
At the end of the burst:
CAMERA = ZERO MOTION for a brief Hero lock.
The Hero frame must clearly display:
lower legs
torso
face
and the deployed circular skirt geometry.
The camera angle should now naturally sit slightly below the character's right-front side.
The upward perspective begins to reveal the face.
---
## HERO LANDING
The Hero does NOT end in a frozen pose.
The extended leg retracts from its airborne line.
The bent leg moves underneath the center of mass.
The first landing foot receives weight.
The knee absorbs impact.
The pelvis continues the direction of travel.
The torso rises after the lower body stabilizes.
One sleeve follows the landing slightly later.
The outer skirt hem continues moving for a moment after the body has redirected.
This residual skirt motion is the visual afterimage of the Hero.
Immediately continue into the next traveling phrase.
==================================================
00:04.256–00:05.856
LYRIC PHRASE — FIRST “第一流”
==========================
The character lands from the Hero and continues.
Do NOT reset.
The body should feel as though it has gained confidence from the previous movement.
00:04.256:
The torso rises from the landing.
The feet continue with two quick gliding steps.
The pelvis redirects toward camera-right.
One arm opens across the body.
The other follows later.
00:04.70–00:05.30:
Introduce a controlled traveling body spiral.
The lower body changes direction twice.
The torso remains elegant and stable.
The sleeves create a long asymmetric curve.
At approximately 00:05.50:
The body begins turning naturally toward the camera.
The feet continue traveling.
The head follows slightly later.
==================================================
00:05.856 ◆
===========
Secondary accent.
ACTION:
At the exact ◆:
The supporting foot lands.
Pelvis changes direction.
Torso follows.
Near shoulder opens.
Wrist finishes with a small circular turn.
The near sleeve expands outward.
The opposite sleeve remains slightly delayed.
This is a compact continuation accent, not a Hero pose.
CAMERA:
Short lateral tracking adjustment synchronized to the ◆.
Keep the medium-close scale.
Do not move closer than the framing needed to keep the face and lower legs visible.
==================================================
00:05.856–00:07.831
FIRST “第一流” DEVELOPMENT
=======================
Build confidence through continuous motion.
The phrase should feel more expansive than Segment 01.
Use:
cross-step
→ weight transfer
→ shallow turn
→ immediate recovery step
→ torso spiral
→ sleeve release
→ another directional change.
Do not perform a complete standalone spin.
The body travels through space.
At approximately 00:06.70:
the character passes through a low diagonal line.
The supporting leg bends.
The torso inclines.
The opposite leg extends naturally through the direction of travel.
At approximately 00:07.20:
recover upward through the supporting foot.
The pelvis rises first.
The torso follows.
The head arrives last.
The camera begins moving toward the character's preferred facial side.
==================================================
00:07.831 ★
SECOND STRONG BEAT — FACIAL / BODY HERO
=======================================
This is NOT another major stunt.
The first Hero already provided the technical spectacle.
This point provides the character's visual authority.
ACTION:
The character completes the traveling phrase with one final strong weight transfer.
The supporting foot plants.
The opposite foot glides through.
The pelvis rotates.
The waist opens.
The chest rises.
The leading arm extends outward.
The sleeve reaches a large diagonal arc.
Only after the torso completes the movement:
the head turns toward the camera.
The face settles into a calm three-quarter view.
The expression remains composed.
No exaggerated smile.
No lyric acting.
CAMERA:
Move into the signature:
FRONT-RIGHT LOW-ANGLE THREE-QUARTER FACIAL VIEW.
Camera is clearly below eye level and looking upward.
Use a short decisive camera burst into this composition.
Keep medium-close framing.
The character should fill most of the frame.
The head should approach the upper edge without cutting the hair.
The lower-leg region must remain visible.
Immediately after reaching the composition:
CAMERA = STATIC HERO LOCK for a brief interval.
Do not zoom.
Do not orbit.
Do not drift.
The body may continue with subtle natural settling.
The sleeve remains in motion longer than the body.
This creates:
body settles
→ sleeve continues
→ gaze settles last.
==================================================
00:08.223–00:09.806
SECOND “第一流” REPEAT — INCREASED CONFIDENCE
==========================================
Do not repeat the exact 07.831 Hero choreography.
The repeated lyric needs a new physical interpretation.
Instead of opening upward:
the body now travels laterally.
At 00:08.223:
immediately leave the Hero composition through a small weight shift.
The character moves diagonally away from the previous camera axis.
Use:
small rapid footwork
→ cross-step
→ immediate weight exchange
→ torso side spiral
→ opposite-side recovery step.
The sleeves trail behind the changing direction.
At approximately 00:08.90:
one sleeve sweeps behind the body.
At approximately 00:09.20:
the opposite sleeve crosses forward.
At approximately 00:09.50:
the feet accelerate toward the final strong beat.
==================================================
00:09.806 ◆
===========
Secondary accent.
ACTION:
Complete the traveling phrase.
The supporting foot lands.
The pelvis redirects sharply but cleanly.
The torso follows.
The head remains slightly delayed.
One sleeve finishes its lateral sweep.
The opposite sleeve begins to gather toward the body.
This prepares the final strong beat.
CAMERA:
Track laterally.
Slightly lower the camera position.
Maintain front-right relationship.
No long push-in.
==================================================
00:09.806–00:11.782
FINAL BUILD
===========
The second lyric repeat becomes increasingly assured.
The movement should now feel more mature and decisive than earlier.
Use a continuous acceleration phrase:
quick footwork
→ cross-step
→ immediate weight transfer
→ low level change
→ rise
→ directional redirection
→ sleeve release.
At approximately 00:10.50:
the body moves across the frame.
The camera tracks with it.
At approximately 00:10.90:
the pelvis changes direction.
The torso is delayed.
At approximately 00:11.20:
the shoulders open.
At approximately 00:11.50:
the face starts moving toward the camera's right-front direction.
The character is preparing for the final visual accent.
==================================================
00:11.782 ★
FINAL STRONG BEAT
=================
This is not a second full technical Hero.
It is the resolution of the Blade-Skirt phrase into a strong protagonist image.
ACTION:
The final traveling step receives weight exactly on the ★.
The body rises from the lower body upward.
Foot
→ knee
→ pelvis
→ waist
→ chest
→ shoulder
→ wrist
→ fingertips.
One sleeve releases outward.
The opposite sleeve remains closer to the body.
The body forms a strong diagonal line.
The face turns toward the camera only after the torso reaches its final alignment.
The gaze settles calmly.
CAMERA:
Use the preferred right-front low-angle perspective again, but from a slightly different lateral position than 07.831.
Do not repeat the exact same framing.
The camera performs a very short editorial reposition on the ★.
Then locks.
The character remains large in frame.
The camera sees:
lower legs
→ torso
→ face
in one coherent medium-close composition.
The sleeve continues to move after the body settles.
==================================================
00:11.782–00:13.040
TRANSITION FROM HERO INTO CONTINUATION
======================================
The body does not freeze after the ★.
Immediately release the final pose into another traveling phrase.
The supporting foot begins moving again.
The opposite foot prepares the next direction.
The pelvis redirects.
The torso follows.
One sleeve descends.
The other sleeve begins the next movement.
The final “第一流” repetition should therefore feel complete but physically unfinished.
This is important because Segment 03 begins with the instrumental release.
==================================================
00:13.040–00:13.200
SEGMENT TRANSITION STATE
========================
At the end of Segment 02:
the character remains in motion.
Do not return to neutral standing.
Do not reset the feet.
Do not return the torso to a flat frontal position.
Do not stop the sleeves.
The body should finish with:
weight already traveling into the next direction,
one foot prepared to receive the next phrase,
torso slightly rotated,
head following the body,
one sleeve still carrying residual motion,
the blade-skirt fabric fully recovered from the Hero and hanging naturally from the waist with a faint remaining sway.
Segment 03 must be able to begin immediately from this exact physical state.
==================================================
CAMERA DIRECTING
================
The camera is subordinate to the choreography.
Primary camera language:
medium-close,
front-right three-quarter,
slightly low perspective,
short tracking,
controlled lateral arcs,
brief editorial bursts,
short Hero locks.
Never use a distant wide shot to explain the Hero.
The camera must reveal the Hero inside the character-dominant framing.
Camera changes are motivated by:
weight transfer,
direction change,
skirt deployment,
airborne direction change,
facial reveal,
strong beat completion.
The right-front low-angle facial perspective is used strategically around:
00:07.831 ★
00:11.782 ★
and as a brief payoff after the Hero around the 00:03.880 event.
Do not use the same facial angle mechanically at every beat.
==================================================
BLADE-SKIRT PHYSICS
===================
The skirt remains fixed at the waist.
Its shape responds to actual body acceleration, rotational force, gravity, and directional change.
During ordinary movement:
subtle lateral sway.
During rapid travel:
the outer layers lag behind the waistband.
During the 00:03.880 Hero:
pelvic acceleration creates the dominant deployment,
inner layers respond first,
outer layers respond last,
the entire circle expands into a blade-like radial disc.
The skirt must remain a coherent waist-centered ring.
The dancer occupies the hollow center.
No body part should visually intersect the outer fabric ring.
During landing:
the skirt loses centrifugal expansion,
then falls naturally around the legs through gravity.
Do not independently animate the skirt.
Every skirt motion must be caused by body movement.
==================================================
BODY MECHANICS
==============
The Hero is driven by the legs and core.
Do not pull the body through the arms.
The major momentum chain is:
rapid footwork
→ weight transfer
→ low compression
→ supporting-foot drive
→ pelvic rotation
→ torso redirection
→ airborne directional exchange
→ asymmetric leg line
→ skirt deployment
→ landing absorption
→ immediate continuation.
The airborne phase must have a real center of mass.
The legs must not float independently.
The landing must visibly absorb weight through:
foot
→ knee
→ hip
→ spine.
The next step begins before the previous action visually resets.
==================================================
PERFORMANCE
===========
Emotional progression:
00:00–03.880
controlled confidence building.
03.880
technical Hero release.
04.256–07.831
greater self-possession.
07.831
facial character Hero.
08.223–11.782
stronger, more assured traveling phrase.
11.782–13.200
visual authority dissolving back into continuous motion.
The face remains serene and self-possessed.
Do not sing.
Do not lip-sync.
Do not literally act the lyrics.
==================================================
EDITING
=======
The segment remains one continuous choreographic sentence.
Editorial camera bursts occur only at meaningful musical events.
Primary visual accents:
00:01.905 ◆
00:03.880 ★
00:05.856 ◆
00:07.831 ★
00:09.806 ◆
00:11.782 ★
The largest editorial event is 00:03.880.
The most important facial composition event is 00:07.831.
The final visual authority event is 00:11.782.
Do not insert arbitrary cuts between these events.
Any reframing must complete inside its specified burst interval and immediately return to stable controlled camera behavior.
==================================================
VISUAL CONSISTENCY
==================
Preserve the exact character and exact visual world established by <PICTURE 1>.
<PICTURE 1> remains the sole visual authority.
<PICTURE 2> is identity clarification only.
Do not replace the background.
Do not invent a separate environment.
Do not redesign the character.
Do not alter the costume.
Do not create additional people.
The final result must read as:
SEGMENT 01 CONTINUATION
→ FASTER FOOTWORK
→ WEIGHT-DRIVEN SHEN-YUN
→ 00:03.880 BLADE-SKIRT SIGNATURE HERO
→ IMMEDIATE PHYSICAL LANDING
→ CONTINUOUS “第一流” PHRASE
→ 00:07.831 RIGHT-FRONT LOW-ANGLE FACIAL HERO
→ SECOND TRAVELING PHRASE
→ 00:11.782 FINAL HERO ACCENT
→ CONTINUOUS TRANSITION INTO SEGMENT 03.
```

---

## SEGMENT 03 — 12.46s（器乐 release，无唱词）

```text
Use <PICTURE 1> as the sole visual authority for the exact character identity, appearance, costume, accessories, body proportions, hairstyle, facial structure, environment, background, lighting, photographic realism, and the exact visual continuation state inherited from Segment 02.
Use <PICTURE 2> only as a three-view identity and body-structure clarification reference for the same character. <PICTURE 2> does not define the environment, background, lighting, pose, composition, or camera.
Use <AUDIO 1> as the ONLY temporal, musical, rhythmic, movement-trigger, camera-trigger, editorial timing, and transition authority.
Create a 12.46-second realistic live-action cinematic Chinese classical dance video.
This segment is a DIRECT PHYSICAL CONTINUATION of Segment 02. Do not reset the character. Do not create a new starting pose. Do not return to neutral standing. Do not restart the choreography. The character is already carrying directional momentum at 00:00.000.
================================================== SEGMENT ROLE
Segment 03 is the first major instrumental release of the full dance.
There are no sung lyrics in this segment.
Do not invent lyrical acting.
The choreography is driven entirely by the supplied AUDIO 1 rhythmic structure.
The movement language is:
Chinese classical dance shen-yun + Song-style Jiangnan / Wu-region elegance + continuous traveling footwork + water-sleeve body mechanics + one high-difficulty airborne Signature Hero.
The segment must feel like the dancer has stopped "presenting" herself and has entered the deepest physical release of the dance.
Energy progression:
CONTINUATION → ACCELERATION → COMPRESSION → SIGNATURE HERO → IMPACT ABSORPTION → CONTINUOUS RECOVERY → SECONDARY EXPANSION → FACIAL HERO → CONTROLLED RESOLUTION.
Do not make the entire segment equally explosive.
================================================== AUDIO 1 — ABSOLUTE TIMING AUTHORITY
<AUDIO 1> is the actual supplied audio track.
BPM = 121.5. Fine beat = approximately 0.494 seconds. Large body phrasing = approximately every 0.988 seconds.
Segment 03 local timeline:
◆ 00:00.557 ★ 00:02.532 ◆ 00:04.508 ★ 00:06.483 ◆ 00:08.458 ★ 00:10.434 ◆ 00:12.409
All major choreography, camera transitions, hero accents, and body timing must synchronize to these exact local timestamps.
Do not create another rhythmic system.
Do not invent independent cinematic timing.
================================================== CONTINUATION STATE 00:00.000
Begin from the exact physical continuation inherited from Segment 02.
The character is already moving.
At 00:00.000:
weight is traveling, one foot is already preparing the next transfer, the torso is slightly rotated, the head follows the body's direction with a fractional delay, one sleeve still carries residual motion from the final phrase of Segment 02.
Do not cancel that momentum.
The first new step must emerge directly from the inherited motion.
The character must never appear to "start dancing again."
================================================== FRAMING PRINCIPLE
Maintain character-dominant medium-close cinematic framing throughout the entire segment.
Primary visual coverage:
lower calf / lower leg → full torso → head and face.
The character should occupy most of the frame.
Do not use a distant establishing shot.
Do not shrink the dancer in order to display more environment.
The framing must preserve:
lower-leg footwork, weight changes, torso mechanics, face readability, and enough spatial room for sleeve trajectories.
Camera distance may change only within this medium-close range.
================================================== CAMERA LANGUAGE
The camera follows the choreography rather than decorating it.
Primary camera grammar:
controlled lateral tracking, short directional camera bursts, low three-quarter perspective, brief Hero locks, motivated angle changes, movement-driven reframing.
The preferred facial Hero perspective is:
camera positioned toward the character's front-right, clearly below eye level, looking upward toward a natural three-quarter facial angle.
This angle is NOT used constantly.
It appears only when the choreography naturally reaches a facial payoff.
The camera does not perform a full continuous orbit around the dancer.
Camera movement must occur only when motivated by:
directional weight transfer, accelerating footwork, airborne trajectory, landing, sleeve expansion, or facial resolution.
================================================== 00:00.000–00:00.557 INHERITED MOMENTUM → ACCELERATION
Continue immediately from Segment 02.
00:00.000–00:00.247:
The supporting foot receives the inherited weight.
The opposite foot glides forward.
The pelvis immediately redirects the body.
The torso remains fractionally delayed.
The nearer shoulder opens.
The previous sleeve finishes its residual movement while the other sleeve begins a new inward curve.
00:00.247–00:00.494:
Increase footwork speed.
Use two compact traveling steps.
The second step begins before the first movement visually resets.
The feet remain light and precise.
The pelvis changes direction first.
The waist follows.
The chest follows later.
The head arrives last.
00:00.494–00:00.557:
Begin a subtle lowering of the center of mass.
The supporting knee bends.
The pelvis moves downward.
The chest remains lifted.
The arms begin gathering inward.
The body is visibly preparing a stronger phrase.
CAMERA:
Begin with controlled lateral tracking.
Stay medium-close.
Remain slightly below torso level.
Do not push into a facial close-up yet.
================================================== 00:00.557 ◆
SECONDARY RHYTHMIC ACCENT.
At the exact ◆:
The supporting foot completes the weight transfer.
The pelvis drops and immediately redirects.
The torso coils subtly through the waist.
The nearer wrist turns outward.
The sleeve responds one fraction later.
The opposite foot is already moving toward the next placement.
Do not pause.
CAMERA:
Perform a short directional burst in the same travel direction.
The camera moves only enough to reveal the new diagonal of the dancer's body.
At the exact end of the short burst:
camera stabilizes.
================================================== 00:00.557–00:02.532 RISING KINETIC SENTENCE
The choreography now becomes visibly faster.
Use:
rapid footwork → cross-step → immediate weight transfer → shallow level change → directional release → torso spiral → sleeve response → immediate next step.
Do not use isolated arm gestures.
Every arm phrase originates from the lower body.
00:01.000 approximately:
The character crosses one foot in front of the other.
The crossing is not ornamental.
It redirects the center of mass.
The pelvis follows.
The torso delays.
One sleeve begins a long diagonal sweep.
00:01.40–00:01.90:
Increase traveling speed.
Take two short consecutive steps.
The supporting leg absorbs the acceleration.
The opposite leg passes through rather than stopping.
The upper body remains controlled against the faster lower body.
00:02.20–00:02.532:
Begin the first large compression.
The dancer lowers through the supporting leg.
The pelvis drops.
The torso folds diagonally.
Both arms briefly collect inward.
The skirt mass begins to react to the increasing acceleration.
This is the preparation for the first strong beat.
================================================== 00:02.532 ★ SECONDARY LARGE EXPANSION
At the exact ★:
Release the compressed body into a broad traveling turn.
This is NOT the major Signature Hero.
The dancer:
pushes from the supporting foot, opens the pelvis, rotates through the waist, lets the chest follow, extends one leg through the travel direction, and opens one sleeve on the delayed side.
The movement should cover visible horizontal distance.
The body creates a large diagonal silhouette.
The opposite sleeve follows later.
The dancer does not complete a full isolated spin.
The body exits the turn already traveling.
CAMERA:
At the exact ★:
perform a short tracking burst following the diagonal travel.
The camera shifts slightly toward the character's right-front side.
Stay medium-close.
Do not pull away.
At the end of the burst:
camera stabilizes.
================================================== 00:02.532–00:04.508 ACCELERATION INTO HERO PREPARATION
After the 00:02.532 phrase, do not relax.
Continue the momentum.
The dancer uses increasingly decisive footwork.
00:02.532–00:03.30:
rapid step → immediate transfer → crossing step → torso spiral.
The pelvis remains the true source of direction.
00:03.30–00:04.00:
Take a longer diagonal traveling step.
The supporting knee bends deeper.
The opposite foot passes through quickly.
The torso begins counter-rotation.
One sleeve gathers toward the waist.
The opposite sleeve trails behind.
This creates asymmetry.
00:04.00–00:04.508:
Compress further.
The center of mass lowers.
The supporting foot loads against the floor.
The pelvis rotates against the shoulders.
The chest folds slightly.
The head remains calm and slightly delayed.
This is not a pose.
This is a real mechanical loading phase for the 00:06.483 Hero.
CAMERA:
Gradually lower the camera position while tracking.
Do not zoom dramatically.
Keep lower legs and face in the same overall medium-close visual relationship.
================================================== 00:04.508 ◆
At the exact ◆:
The loaded supporting foot lands decisively.
The pelvis completes the final pre-Hero directional change.
The torso follows.
The leading shoulder opens.
The rear shoulder remains delayed.
The near sleeve snaps outward briefly.
The dancer immediately begins a second compression.
The next foot placement is already prepared.
CAMERA:
Short low lateral burst.
The camera moves toward the character's front-right diagonal.
This is the beginning of the Hero camera setup.
At the end:
camera stabilizes briefly.
================================================== 00:04.508–00:06.483 FINAL HERO BUILD
This is the most important physical buildup in the first half of the segment.
Do not insert any unrelated gesture.
The dancer performs a continuous sequence:
rapid step → cross-step → immediate weight transfer → deep plié → counter-rotation → sleeve gather → supporting-foot drive.
00:05.00 approximately:
The supporting leg becomes visibly loaded.
The pelvis begins to rotate against the upper body.
The torso remains compact.
The head stays quiet.
00:05.40 approximately:
The opposite foot passes behind the center line.
The body redirects diagonally.
The arms remain asymmetrical.
One sleeve is compact.
The other hangs slightly behind the moving shoulder.
00:05.90 approximately:
The supporting leg drives upward.
The pelvis begins to rise.
The chest begins to unfold.
The head is still not leading.
00:06.20–00:06.483:
The dancer reaches the final launch preparation.
Supporting foot fully loaded.
Knee extending.
Pelvis rotating.
Chest opening.
One arm releasing upward.
The opposite arm remaining lower.
The body is visibly storing the force required for the Hero.
================================================== 00:06.483 ★ SIGNATURE HERO MOMENT 01 AIRBORNE CROSS-DIRECTION HERO
At the exact ★:
EXECUTE THE FULL SIGNATURE HERO.
This is the main technical Hero of Segment 03.
It must be clearly different from the rotational blade-skirt Hero established in Segment 02.
Do NOT reproduce the previous skirt-disc spin.
Do NOT perform a one-hand floor spin.
Do NOT use a ground-supported hand rotation.
Do NOT make this a simple jump.
The Hero sequence is:
EXPLOSIVE ONE-FOOT DRIVE → AIRBORNE DIAGONAL RISE → BODY COUNTER-ROTATION → MID-AIR LEG SWITCH → ASYMMETRIC EXTENDED LINE → CONTROLLED HALF-TO-FULL DIRECTION CHANGE → DESCENDING BODY FOLD → OPPOSITE-FOOT LANDING → IMMEDIATE TRAVEL.
HERO BODY MECHANICS
The supporting foot provides the launch.
The force travels:
foot → ankle → knee → hip → pelvis → waist → torso → shoulder → arm.
The arms do NOT create the jump.
The head does NOT initiate the rotation.
At takeoff:
the pelvis rises first.
The torso follows.
The chest opens after the lower body has already committed.
During the airborne phase:
one leg initially extends along the travel direction.
The opposite leg remains bent.
Then the extended leg retracts and the bent leg switches outward.
This produces a visible mid-air change of leg geometry.
The body simultaneously changes its direction through the waist and ribcage.
The final airborne silhouette is asymmetric:
one leg extended diagonally, one leg bent and drawn inward, torso leaning on a different diagonal, one arm high, one arm lower.
Do not create a symmetrical split.
Do not freeze the body at the top.
The body must pass continuously through the geometry.
AIRBORNE MOVEMENT
The Hero travels diagonally rather than rotating in place.
The dancer rises.
The body changes direction while airborne.
The legs exchange their visual roles during flight.
The torso folds slightly through the center.
Then the chest opens again just before descent.
This creates a clear:
compressed → expanded → re-compressed airborne phrase.
SLEEVE PHYSICS
The sleeve movement originates from the body.
At takeoff:
one sleeve releases upward.
The opposite sleeve remains slightly behind.
At mid-air expansion:
the upper sleeve forms a long curved trajectory.
The opposite sleeve creates a counter-arc.
The sleeves do not move independently.
They trail the shoulder and wrist motions with natural inertial lag.
At descent:
the upper sleeve begins falling before the body fully lands.
Its hem continues traveling after the hand has already redirected.
SKIRT PHYSICS
The skirt reacts to the airborne acceleration but does NOT become a second Hero.
It remains attached at the waist.
The centrifugal and vertical forces create a broad asymmetric lift.
The skirt follows:
pelvic acceleration → waistband → inner layers → middle layers → outer hem.
Do not form the same upright disc geometry used in Segment 02.
Instead:
the skirt lifts into an oblique flowing arc, with the outer hem trailing behind the direction of travel.
The visual emphasis is:
BODY TRAJECTORY + LEG SWITCH + SLEEVE ARC
not skirt spectacle.
HERO CAMERA
At 00:06.483 ★:
Perform a short, high-velocity editorial camera burst.
The camera follows the airborne trajectory diagonally forward and slightly upward.
The camera does NOT orbit fully around the dancer.
The camera does NOT rotate around the body.
The purpose is to reveal the airborne leg switch and diagonal body line while preserving the dancer's large on-screen presence.
Maintain medium-close coverage.
Keep:
lower legs, torso, and face
within the readable composition.
During the apex of the Hero:
the camera reaches the decisive Hero framing.
Immediately after the burst:
CAMERA = ZERO MOTION.
Hold the Hero geometry briefly.
The character's body completes the airborne motion.
The camera does not chase every millimeter of the flight.
LANDING
The Hero resolves through the lower body.
The extended leg retracts.
The opposite leg moves underneath the center of mass.
The landing foot receives weight.
The knee absorbs impact.
The pelvis continues traveling.
The torso rises after the lower body has absorbed the landing.
The head arrives last.
Do NOT land and freeze.
Do NOT land and pose.
The landing must directly generate the next traveling action.
================================================== 00:06.483–00:08.458 HERO AFTERMATH / CONTINUOUS RECOVERY
The dancer is now immediately moving out of the Hero.
No reset.
00:06.70 approximately:
The landing knee bends.
The opposite foot passes through.
The pelvis continues in the same direction.
00:07.10 approximately:
The torso unfolds.
The chest opens.
The lower sleeve descends.
The upper sleeve continues its inertial arc.
00:07.50 approximately:
The feet begin a rapid two-step recovery.
The character uses the landing momentum to redirect diagonally.
The body becomes vertically composed again.
The sleeves remain behind the body for a fraction of a beat.
This creates:
body recovered → fabric still traveling.
CAMERA:
After the Hero lock, resume controlled lateral tracking.
Do not immediately jump to the face.
Keep the whole body movement readable.
================================================== 00:08.458 ◆
At the exact ◆:
Complete the recovery phrase.
One foot lands.
The pelvis redirects sharply.
The torso follows.
The near shoulder opens.
The wrist completes a small circular release.
The far sleeve catches up.
Immediately begin the next movement.
No pause.
CAMERA:
Use a short, controlled reframing burst.
Shift from the Hero's diagonal tracking axis toward a more frontal-right relationship.
This is preparation for the facial Hero at 00:10.434.
================================================== 00:08.458–00:10.434 POST-HERO EXPANSION
The dance now changes character.
The difficult Hero has already happened.
Do not attempt to exceed it with another stunt.
Instead, show the dancer's control.
Use:
rapid footwork → long diagonal step → waist spiral → controlled level change → rising torso → sleeve opening → traveling redirection.
00:08.90 approximately:
The character passes laterally through the frame.
The feet remain quick.
The torso stays elegant and calm.
00:09.30 approximately:
The pelvis initiates a turn.
The chest follows.
The head remains delayed.
One sleeve opens backward.
00:09.70 approximately:
The character takes a deeper traveling step.
The body forms a long diagonal line.
00:10.00 approximately:
The supporting foot receives weight.
The torso begins to rise.
The head starts turning toward the camera.
00:10.20–00:10.434:
The body completes the directional change.
The chest opens.
One sleeve forms a broad asymmetric arc.
The face approaches its natural right-front three-quarter orientation.
This is the setup for the second major visual payoff.
================================================== 00:10.434 ★ FACIAL HERO / SONG-STYLE HERO IMAGE
This ★ is NOT another technical stunt.
It is the visual and emotional resolution of the physical Hero.
At the exact beat:
The supporting foot completes a decisive weight transfer.
The pelvis settles.
The waist opens.
The chest rises.
One sleeve reaches full extension.
The opposite sleeve remains slightly lower and delayed.
The body forms a controlled diagonal line.
Only after the torso completes:
the head turns.
The eyes arrive last.
The expression remains calm, self-possessed, and serene.
No exaggerated smile.
No literal lyric acting.
No lip-sync.
CAMERA:
This is the strongest use of the preferred facial angle in Segment 03.
Move into:
FRONT-RIGHT + LOW CAMERA POSITION + UPWARD THREE-QUARTER FACIAL VIEW.
The camera is clearly below eye level.
The camera looks upward toward the character's face.
The face must read strongly without becoming an isolated head close-up.
Maintain the medium-close framing:
lower-leg region → torso → head.
The character occupies most of the frame.
Perform a short decisive camera burst into this composition.
Immediately after reaching the composition:
CAMERA = ABSOLUTE HARD LOCK.
No orbit. No push. No pull. No drift. No shake.
Let the body settle naturally.
Let the sleeve continue moving slightly longer than the body.
This should create:
BODY STOPS → SLEEVE CONTINUES → EYES SETTLE LAST.
================================================== 00:10.434–00:12.409 HERO RESOLUTION → FINAL EXPANSION
Do not remain frozen for the entire remainder.
The facial Hero is brief.
Immediately begin releasing the pose.
00:10.60 approximately:
The gaze remains calm.
The supporting leg begins moving again.
The torso starts to redirect.
The near sleeve lowers.
The opposite sleeve begins a new arc.
00:11.00 approximately:
Use a compact traveling step.
The pelvis redirects.
The torso follows.
The head follows slightly later.
00:11.40 approximately:
The character expands the final movement phrase.
One arm opens.
The other remains closer to the body.
The feet continue traveling.
The dancer is again moving away from the Hero image.
The movement must feel effortless after the difficult Hero.
================================================== 00:12.409 ◆
At the exact final ◆:
Complete the final phrase of Segment 03.
The supporting foot receives weight.
The pelvis settles into the next diagonal.
The torso lowers slightly.
The leading sleeve drops through a long controlled arc.
The opposite sleeve remains in slight motion.
The head turns subtly toward the direction of Segment 04.
This is not a final pose.
It is a transition state.
CAMERA:
Use a very small lateral adjustment synchronized to the ◆.
Remain medium-close.
Do not return to a flat frontal composition.
Do not zoom away.
================================================== 00:12.409–00:12.460 SEGMENT TRANSITION STATE
At the end of Segment 03:
The character is still physically alive and in motion.
Weight is already directed into the next phrase.
One foot is preparing the next transfer.
The torso remains slightly diagonal.
The head follows the body.
One sleeve has not yet completely settled.
The skirt remains naturally attached at the waist with a faint residual sway.
Do not freeze.
Do not reset.
Segment 04 must begin directly from this physical state.
================================================== MOTION QUALITY
The dancer must feel professionally trained.
Every movement follows real weight logic.
The lower body is the driver.
The upper body follows.
The sleeves follow the wrists.
The fabric follows the body.
The head follows the torso.
Never reverse this causality unless a deliberate delayed gesture is explicitly described.
The dominant kinetic chain is:
foot → weight → pelvis → waist → torso → shoulder → arm → wrist → fingertips → sleeve.
The Hero chain is:
rapid footwork → compression → supporting-foot drive → pelvic rotation → airborne rise → mid-air leg switch → counter-rotation → controlled descent → landing absorption → immediate continuation.
================================================== EDITORIAL STRUCTURE
This segment is one continuous choreographic sentence.
Camera changes are discrete and motivated.
Major visual events occur at:
00:00.557 ◆ 00:02.532 ★ 00:04.508 ◆ 00:06.483 ★ HERO 00:08.458 ◆ 00:10.434 ★ FACIAL HERO 00:12.409 ◆
Do not insert arbitrary cuts between them.
Do not make every accent a huge camera move.
The visual hierarchy is:
00:02.532 = large dance expansion
00:06.483 = main technical Signature Hero
00:10.434 = facial / character Hero
The segment must visually breathe between these three levels.
================================================== PERFORMANCE
No singing.
No lip-sync.
No literal interpretation of lyrics.
No theatrical overacting.
The emotional quality is:
freedom → release → confidence → control.
The dancer becomes increasingly expansive through movement rather than facial expression.
The face remains serene and composed even during difficult physical actions.
================================================== VISUAL CONSISTENCY
PICTURE 1 remains the sole complete visual authority for the character and environment.
Keep the exact character.
Keep the exact costume.
Keep the exact appearance.
Keep the exact background and environmental continuity from PICTURE 1.
Do not invent a replacement environment.
Do not introduce new scenery.
PICTURE 2 is identity clarification only.
Do not use PICTURE 2 as a scene source.
================================================== FINAL READ
The result must clearly read as:
SEGMENT 02 CONTINUATION → RAPID SONG-STYLE FOOTWORK → 00:02.532 LARGE EXPANSION → 00:04.508 HERO LOADING → 00:06.483 AIRBORNE CROSS-DIRECTION SIGNATURE HERO → CONTROLLED LANDING WITHOUT RESET → 00:08.458 CONTINUOUS RECOVERY → 00:10.434 RIGHT-FRONT LOW-ANGLE FACIAL HERO → CONTROLLED RELEASE → 00:12.409 TRANSITION INTO SEGMENT 04.
The visual identity of this segment is:
CONTINUOUS TRAVEL + LOWER-BODY-DRIVEN SHEN-YUN + ASYMMETRIC AIRBORNE LEG EXCHANGE + WATER-SLEEVE INERTIAL FOLLOW-THROUGH + ONE DISTINCT TECHNICAL HERO + ONE DISTINCT FACIAL HERO + MEDIUM-CLOSE CHARACTER-DOMINANT CINEMATIC FRAMING.
```

---

## SEGMENT 04 — 12.463s（爆发后的呼吸 / 控制段）

```text
Use <PICTURE 1> as the sole visual authority for the exact character identity, appearance, costume, accessories, body proportions, hairstyle, facial structure, environment, background, lighting, photographic realism, and the exact visual continuation state inherited from Segment 03.
Use <PICTURE 2> only as a three-view identity and body-structure clarification reference for the same character. <PICTURE 2> does not define the environment, background, lighting, pose, composition, or camera.
Use <AUDIO 1> as the ONLY temporal, musical, rhythmic, movement-trigger, camera-trigger, editorial timing, and transition authority.
Create a 12.463-second realistic live-action cinematic Chinese classical dance video.
This segment is a DIRECT PHYSICAL CONTINUATION of Segment 03. The dancer does not restart. The dancer does not reset into a neutral standing pose. The dancer remains inside the same ongoing choreographic sentence.
================================================== SEGMENT ROLE
Segment 04 is the deliberate breath after the first major instrumental Hero of the full dance.
Segment 03 was expansive, fast, airborne, and technically explosive.
Segment 04 changes the difficulty rather than simply reducing it:
EXPLOSIVE AFTERMATH → CONTROLLED DECELERATION → LOW CENTER OF MASS → QUIET SINGLE-LEG AXIS → RESTRAINED SHEN-YUN → GRADUAL UPPER-BODY OPENING → FACIAL HERO → LYRICAL RESOLUTION → CONTINUATION.
The movement language remains:
Chinese classical dance shen-yun + Song-style Jiangnan / Wu-region elegance + yuan-chang traveling steps + subtle water-sleeve mechanics + low-level controlled body spirals + delayed head and eye timing.
The segment must feel like a highly trained dancer who has complete control over the violent momentum of the previous section.
Do not make the movement lifeless.
The lower body continues moving continuously, but the visible effort becomes quieter.
The difficulty comes from balance, weight control, delayed coordination, and precision rather than from another large stunt.
================================================== AUDIO 1 — ABSOLUTE TIMING AUTHORITY
<AUDIO 1> is the actual supplied audio track.
BPM = 121.5. Fine beat = approximately 0.494 seconds. Large body phrasing = approximately 0.988 seconds.
Segment 04 local timeline:
◆ 00:03.899 ★ 00:01.924 ◆ 00:07.850 ★ 00:05.875 ◆ 00:03.899 ★ 00:09.825 ◆ 00:11.801
Primary chronological beat map:
◆ 00:03.899 ★ 00:01.924 ◆ 00:07.850 ★ 00:05.875 ◆ 00:11.801 ★ 00:09.825
For execution clarity, the actual chronological order of the main accents is:
★ 00:01.924 ◆ 00:03.899 ★ 00:05.875 ◆ 00:07.850 ★ 00:09.825 ◆ 00:11.801
Lyric entry:
00:10.563 — “梅定妒”
The lyric enters after the 00:09.825 ★ visual preparation.
All choreography, camera changes, facial timing, Hero framing, and editorial transitions must synchronize directly to <AUDIO 1>.
Do not create an independent cinematic rhythm.
================================================== CONTINUATION STATE 00:00.000
Begin from the exact physical state established at the end of Segment 03.
At 00:00.000:
the dancer is already moving out of the previous phrase, the body remains slightly diagonal, the weight is already traveling into the next supporting foot, one sleeve is still descending from the preceding movement, the opposite sleeve retains a faint residual arc, the pelvis is already redirecting, the head remains fractionally delayed behind the torso.
The first motion of Segment 04 must emerge directly from this inherited momentum.
Do not cancel the previous movement before beginning the new phrase.
================================================== FRAMING PRINCIPLE
Maintain character-dominant medium-close cinematic framing throughout the segment.
Primary coverage:
lower calf / lower legs → hips → torso → head and face.
The dancer occupies most of the frame.
The camera remains close enough for facial detail to retain strong pixel density while preserving enough of the lower legs to reveal real weight transfer.
Maintain readable:
foot placement, supporting leg, pelvis direction, torso mechanics, sleeve trajectories, and facial angle.
Do not pull into a distant establishing shot.
Do not shrink the character to reveal additional environment.
The visual emphasis is always the dancer.
================================================== CAMERA LANGUAGE
The camera behaves as a responsive dance camera.
It remains close to the dancer and changes angle only when motivated by:
decelerating weight, single-leg balance, torso redirection, sleeve release, head delay, facial arrival, or lyrical resolution.
Primary camera grammar:
restrained lateral tracking, short editorial camera bursts, slight low three-quarter perspective, brief hard locks, right-front low-angle facial Hero framing.
The preferred facial Hero perspective is:
camera positioned toward the character's front-right, camera clearly below eye level, camera looking upward, natural three-quarter facial angle.
This angle is reserved for the 00:09.825 Hero preparation and its immediate lyric resolution.
Do not mechanically repeat this angle throughout the whole segment.
================================================== 00:00.000–00:00.557 AFTERMATH DECELERATION
Continue immediately from Segment 03.
00:00.000–00:00.247:
The landing/recovery momentum from Segment 03 continues into a softer directional travel.
The supporting foot receives weight.
The opposite foot passes forward without stopping.
The pelvis begins to lower.
The torso follows with a slight delay.
The sleeve that was still traveling from Segment 03 completes its downward arc.
The opposite wrist begins a quiet inward curve.
00:00.247–00:00.494:
Reduce the travel speed without becoming static.
Use two small, precise steps.
The second step begins before the first step has completely visually reset.
The lower body remains active.
The upper body becomes calmer.
The chest remains gently lifted.
The head stays behind the torso by a small fraction.
00:00.494–00:00.557:
The supporting knee softens.
The center of mass lowers.
The dancer gathers the arms closer to the body's center without collapsing.
This creates the preparation for the first quiet large phrase.
CAMERA:
Use controlled lateral tracking.
Stay medium-close.
Remain slightly below torso level.
Do not move toward the face yet.
================================================== 00:00.557–00:01.924 QUIET TRAVELING PHRASE
Continue the dance as a continuous Song-style traveling phrase.
Use:
compact yuan-chang step → immediate weight transfer → subtle cross-step → waist spiral → delayed shoulder release → controlled sleeve response.
The dancer should appear to glide rather than march.
The feet are still active.
The visual energy has changed from explosive to refined.
The body's center of mass stays relatively low while the upper body remains elegant.
One sleeve traces a narrow horizontal-to-diagonal path.
The other sleeve remains closer to the torso and catches up later.
Do not synchronize both sleeves into the same motion at the same instant.
================================================== 00:01.924 ★ LARGE BODY PHRASE 01
At the exact ★:
The dancer completes a controlled lateral opening.
The supporting foot receives the full transfer.
The pelvis rotates first.
The waist follows.
The chest opens after the pelvis has already committed.
One leg extends through the direction of travel.
The opposite leg remains soft and ready for the next placement.
One sleeve opens outward on the same side as the torso rotation.
The opposite sleeve follows slightly later.
The movement forms a long, elegant diagonal rather than a vertical jump.
This is a large visual phrase but not a technical Hero.
Its purpose is to establish:
quiet amplitude + controlled asymmetry + continuous travel.
CAMERA:
Perform a short directional burst following the opening diagonal.
Shift slightly toward the dancer's right-front side.
Do not zoom dramatically.
Remain within medium-close character-dominant framing.
At the exact end of the burst:
CAMERA = HARD LOCK.
Hold only long enough for the completed body line to read.
Then continue moving.
================================================== 00:01.924–00:03.899 CONTROLLED DESCENT
Immediately release the large phrase.
Do not pose.
The extended leg returns through the next step.
The supporting knee softens.
The pelvis descends.
The torso inclines slightly forward and sideways.
The head remains calm.
The wrists become smaller and more articulate.
The dancer now travels with compressed amplitude.
00:02.40 approximately:
Use a compact crossing step.
The pelvis redirects.
The upper body briefly resists that redirection.
This produces a subtle torsional stretch.
00:03.00 approximately:
The dancer passes through a low center-of-mass position.
One sleeve curves around the outside of the body.
The other sleeve remains delayed.
00:03.50 approximately:
The body begins to prepare a controlled single-leg balance phrase.
One foot becomes the future support.
The other foot passes through rather than being lifted prematurely.
================================================== 00:03.899 ◆
At the exact ◆:
The supporting foot plants cleanly.
The pelvis completes a small redirection.
The torso folds slightly through the waist.
The near shoulder drops.
The far shoulder remains open.
The wrist releases outward.
The sleeve reacts after the wrist.
The opposite foot is already preparing the next step.
No pause.
CAMERA:
Use a very short low lateral reframing burst.
Lower the camera slightly while keeping the dancer's head and lower legs readable.
At the end of the burst:
camera stabilizes.
This prepares the low-axis phrase.
================================================== 00:03.899–00:05.875 LOW AXIS PREPARATION
This phrase becomes more technically demanding while remaining visually restrained.
The dancer moves:
cross-step → supporting-foot load → low plié → pelvis rotation → torso counter-tilt → free-leg sweep → sleeve counter-arc.
The difficulty comes from maintaining balance while the body and lower body move in different directions.
00:04.30 approximately:
The supporting knee bends deeper.
The free leg passes close to the floor.
The pelvis rotates.
The torso delays.
00:04.70 approximately:
The free leg begins a low circular sweep around the supporting axis.
The torso moves in the opposite direction.
One arm lengthens outward.
The other remains folded closer to the waist.
00:05.20 approximately:
The dancer reaches the most compressed point.
The supporting leg carries almost the entire visible weight.
The torso is diagonally inclined.
The head remains calm and does not anticipate the upcoming opening.
00:05.60–00:05.875:
The supporting foot drives upward without jumping.
The pelvis rises first.
The torso unfolds.
The free leg completes its sweep and passes through into the next direction.
================================================== 00:05.875 ★ CONTROL HERO — SINGLE-AXIS BODY SPIRAL
At the exact ★:
Perform the most technically demanding controlled movement of this segment.
This is NOT another airborne Hero.
This is NOT a floor spin.
This is NOT a large rotational stunt.
The difficulty is the dancer's ability to control a long, asymmetric body line around a single supporting foot.
Sequence:
DEEP SUPPORTING KNEE → LOW FREE-LEG SWEEP → PELVIC ROTATION → TORSO COUNTER-TILT → SINGLE-AXIS BODY SPIRAL → FREE-LEG EXTENSION → 180° REDIRECTION → CONTROLLED RISE → IMMEDIATE STEP-OUT.
The supporting leg remains active and grounded.
The torso leans away from the supporting leg.
The free leg extends diagonally outward during the middle of the spiral.
The shoulders remain intentionally asymmetric.
One arm reaches along the direction of the free leg.
The opposite arm traces a lower counter-curve.
The head does not lead the rotation.
The pelvis changes direction first.
The ribcage follows.
The head arrives last.
The dancer passes through the extreme line continuously.
Do not freeze at the deepest angle.
Do not make the movement look weightless.
The body must visibly carry its own weight.
The supporting ankle and knee make tiny natural corrections.
The free leg passes through a controlled arc and then retracts into the next step.
This is a balance-and-control Hero, not a spectacle Hero.
================================================== HERO FABRIC RESPONSE
During the 00:05.875 controlled spiral:
the sleeve follows the wrist with slight delay, the outer sleeve continues traveling after the wrist changes direction, the skirt follows the pelvis and supporting leg, the fabric never moves independently.
The visual hierarchy is:
body axis → free leg line → sleeve counter-arc → face.
Do not let the fabric become the main subject.
================================================== HERO CAMERA
At 00:05.875:
Use a short, precise camera burst following the body axis.
The camera moves slightly around the dancer's front-right diagonal while remaining close.
The movement is quick and motivated by the supporting-foot rotation.
Do not perform a full orbit.
Do not circle continuously around the dancer.
At the point where the dancer reaches the longest asymmetric body line:
CAMERA = HARD LOCK.
Hold the composition briefly.
The visual intention is:
DIFFICULT BALANCE → PERFECT CONTROL → STILL CAMERA → BODY COMPLETES THE SPIRAL.
Immediately after the Hero geometry resolves, resume controlled tracking.
================================================== 00:05.875–00:07.850 HERO RECOVERY
The dancer exits immediately.
No neutral standing reset.
The free leg returns underneath the center of mass.
The supporting knee absorbs the transition.
The pelvis redirects.
The torso rises after the pelvis.
The shoulders soften.
The head follows last.
00:06.30 approximately:
One compact traveling step.
00:06.80 approximately:
A second step begins immediately.
The dancer is now moving in a new diagonal.
00:07.20 approximately:
One sleeve opens gently.
The opposite sleeve remains low and catches the motion later.
The body has returned upright, but the lower body remains active.
The difficult movement is absorbed into graceful travel.
CAMERA:
Controlled lateral tracking.
Stay medium-close.
Do not immediately move to the face.
================================================== 00:07.850 ◆
At the exact ◆:
Complete the recovery phrase.
One foot receives weight.
The pelvis subtly turns.
The torso remains almost still for a fractional moment.
The near wrist traces a small circle.
The far sleeve follows.
The dancer then immediately enters the next phrase.
This is the beginning of the emotional preparation for “梅定妒”.
CAMERA:
Use a very short directional burst toward the dancer's front-right side.
The camera becomes slightly lower.
The framing remains:
lower legs → torso → head.
Do not yet reveal the full facial Hero angle.
================================================== 00:07.850–00:09.825 EMOTIONAL BUILD
The dance becomes quieter but more authoritative.
The dancer takes:
two controlled traveling steps → slight body inclination → waist-led return → sleeve opening → chest lift → head delay.
The body is no longer trying to prove technical ability.
The dancer appears completely in control of the movement.
00:08.30 approximately:
The pelvis redirects first.
The torso follows.
The near shoulder opens.
The opposite shoulder remains slightly behind.
00:08.80 approximately:
One foot crosses through.
The free leg extends briefly before returning to the ground.
00:09.20 approximately:
The chest begins to rise.
The arms open asymmetrically.
One sleeve creates a long diagonal line.
The other sleeve hangs lower.
00:09.55 approximately:
The body completes the final redirection.
The head begins turning.
The eyes have not yet fully arrived.
00:09.825:
Everything reaches the final pre-lyric Hero geometry.
================================================== 00:09.825 ★ FACIAL HERO — RIGHT-FRONT LOW-ANGLE RESOLUTION
At the exact ★:
Complete the most important visual Hero of Segment 04.
This is the emotional and character Hero rather than a stunt.
The supporting foot finishes its weight transfer.
The pelvis settles.
The waist lengthens.
The chest opens.
One sleeve reaches the end of its broad arc.
The opposite sleeve remains slightly delayed.
The head turns only after the torso has completed its movement.
The eyes arrive last.
The face is calm.
The expression is self-possessed, serene, and quietly proud.
No exaggerated smile.
No theatrical emotion.
No literal acting of the lyric.
The visual feeling is:
the dancer does not need to prove anything.
CAMERA:
Perform a short decisive camera burst into:
FRONT-RIGHT + LOW POSITION + UPWARD THREE-QUARTER FACE.
The camera is clearly below eye level.
The camera looks upward toward the natural three-quarter facial angle.
The face becomes the strongest visual point.
However, maintain medium-close coverage.
The frame must still preserve:
lower-leg region → torso → head.
Do not turn this into an isolated floating face close-up.
At the exact moment the Hero framing is established:
CAMERA = ABSOLUTE HARD LOCK.
No drift.
No continued push.
No orbit.
No residual travel.
The camera remains still while the dancer completes the final fraction of the movement.
The sleeve may continue to move subtly after the body settles.
================================================== 00:09.825–00:10.563 HERO HOLD → LYRIC ENTRY
Maintain the Hero composition briefly.
The body remains almost still.
Natural breathing only.
The head remains in the established three-quarter angle.
The gaze is calm.
The nearest sleeve continues a tiny residual descent.
The opposite sleeve settles more slowly.
The camera stays completely locked.
At 00:10.563 the lyric enters:
“梅定妒”
The lyric should trigger a subtle emotional change in the body, not theatrical mime.
================================================== 00:10.563–00:11.801 “梅定妒” — RESTRAINED LYRIC PHRASE
At 00:10.563:
The torso begins a very small redirection.
The supporting foot initiates a new weight transfer.
The shoulders follow later.
The head does not immediately turn.
The face remains serene.
The lyric is expressed through:
posture + timing + gaze + controlled sleeve movement,
not through literal character acting.
00:10.90 approximately:
The near wrist lowers.
The sleeve descends from the Hero position.
The free foot passes into the next placement.
00:11.20 approximately:
The pelvis redirects.
The torso follows.
The head remains delayed.
00:11.50 approximately:
The dancer begins to leave the facial Hero composition physically while remaining visually close to camera.
The movement remains elegant and restrained.
CAMERA:
Remain almost completely stable through this lyrical phrase.
Only a tiny lateral correction is allowed to preserve the dancer's medium-close framing.
Do not abandon the established low-angle language abruptly.
================================================== 00:11.801 ◆
At the exact ◆:
Complete the lyric-resolution phrase.
The supporting foot receives weight.
The torso lowers by a small controlled amount.
The near sleeve reaches its lowest point.
The far sleeve remains fractionally elevated and then follows.
The head turns toward the direction of the next segment.
This is not a final pose.
It is a continuation state.
CAMERA:
Use a short, restrained lateral burst.
Reframe just enough to reveal the dancer's body already preparing the next direction.
Remain medium-close.
Do not zoom out.
Do not create a new environment.
================================================== 00:11.801–00:12.463 DIRECT CONTINUATION STATE
The movement continues naturally.
The dancer takes one final compact step.
The pelvis carries the directional change.
The torso follows.
The head remains slightly behind.
One sleeve is still descending.
The opposite sleeve begins another small arc.
The dancer ends this segment with the body already committed to Segment 05.
At 00:12.463:
the weight is slightly forward, one foot is prepared for the next transfer, the torso remains subtly diagonal, the head has begun turning toward the next direction, the sleeves retain faint residual movement.
Do not stop the dance.
Do not return to neutral.
Do not create a concluding pose.
Segment 05 must be able to start directly from this physical state.
================================================== BODY MECHANICS
Every movement follows the same causal chain:
foot → weight transfer → supporting leg → pelvis → waist → torso → shoulder → arm → wrist → fingertips → sleeve.
For this segment, the primary visible principle is:
LOWER-BODY CONTROL + UPPER-BODY DELAY.
The supporting foot must visibly carry the dancer.
The knee absorbs and redirects weight.
The pelvis changes direction before the chest.
The chest changes direction before the head.
The head arrives last.
The sleeves follow the wrists with inertial delay.
The difficult 00:05.875 Hero must feel physically supported by the grounded leg rather than weightless.
================================================== SONG-STYLE WU-RHYTHM
The body language should retain the elegance of Song-style Jiangnan / Wu-region movement:
light traveling steps, contained amplitude, rounded transitions, delayed head timing, subtle torso spirals, water-sleeve responsiveness, quiet gaze changes, and movement that feels continuous rather than posed.
Avoid making the dancer look like she is performing a generic modern ballet routine.
The visual character should feel distinctly Chinese classical and Song-like:
the lower body glides, the torso carries delayed motion, the arms describe restrained arcs, and the sleeves preserve the memory of each movement for a fraction of a beat.
================================================== EDITORIAL STRUCTURE
The segment remains one continuous dance sentence.
Major rhythmic events:
00:01.924 ★ — large body opening
00:03.899 ◆ — low-axis preparation
00:05.875 ★ — CONTROL HERO / SINGLE-AXIS BODY SPIRAL
00:07.850 ◆ — recovery and emotional preparation
00:09.825 ★ — FACIAL HERO / RIGHT-FRONT LOW-ANGLE
00:10.563 — lyric entry “梅定妒”
00:11.801 ◆ — lyrical resolution and transition.
Do not make every timestamp an equal visual explosion.
The hierarchy is deliberate:
00:01.924 = body expansion
00:05.875 = technical control
00:09.825 = character / facial payoff
This creates the intended visual breathing:
EXPANSION → RESTRAINT → TECHNICAL CONTROL → RECOVERY → CHARACTER REVEAL → LYRICAL RESOLUTION.
================================================== PERFORMANCE
No singing.
No lip-sync.
No exaggerated theatrical acting.
Do not mime the literal meaning of “梅定妒”.
The face remains composed.
The emotional progression is:
AFTERGLOW → RESTRAINT → CONTROL → SELF-POSSESSION → QUIET PRIDE → CONTINUATION.
The character's emotional strength should come primarily from timing, posture, gaze, and control of movement.
================================================== CAMERA TEMPORAL RULE
Every camera movement must be confined to an explicitly defined interval.
Static moments are visibly static.
Burst moments are short and decisive.
Do not allow camera movement to leak into a locked composition.
The exact camera logic is:
00:00.000–00:00.557 controlled continuation tracking
00:00.557–00:01.924 close traveling observation
00:01.924 short expansion burst → lock
00:03.899 short low-axis reframing burst → stabilize
00:05.875 Hero camera burst → Hero lock
00:05.875–00:07.850 controlled tracking recovery
00:07.850 short front-right adjustment
00:09.825 decisive right-front low-angle facial burst → absolute lock
00:10.563–00:11.801 mostly static lyrical observation with only tiny framing correction
00:11.801 restrained transition burst
00:12.463 continuous transition state.
================================================== VISUAL CONTINUITY
Maintain exact visual continuity with Segment 03 through <PICTURE 1>.
The environment remains the same visual world already established.
Do not introduce a replacement setting.
Do not redesign the character.
Do not reinterpret the costume.
Do not alter identity.
<PICTURE 2> remains identity clarification only.
The entire segment must feel like the same continuous dancer performing the next eight-count rather than a newly generated scene.
================================================== REALISM
Realistic live-action photography.
Real human anatomy.
Natural balance corrections.
Natural weight transfer.
Natural fabric inertia.
Natural optical perspective.
Natural skin and face rendering.
Realistic depth of field.
Realistic motion blur only during short camera bursts.
No artificial floating motion.
No weightless body behavior.
No synthetic mechanical gestures.
The supporting leg must visibly carry the body during the 00:05.875 control Hero.
================================================== FINAL READ
The final result must clearly read as:
SEGMENT 03 AFTERMATH → QUIET TRAVELING RECOVERY → 00:01.924 BODY EXPANSION → LOW CENTER-OF-MASS PHRASE → 00:05.875 SINGLE-AXIS CONTROL HERO → IMMEDIATE RECOVERY → 00:09.825 RIGHT-FRONT LOW-ANGLE FACIAL HERO → 00:10.563 “梅定妒” LYRIC ENTRY → RESTRAINED LYRICAL MOVEMENT → 00:11.801 TRANSITION → 00:12.463 DIRECT PHYSICAL CONTINUATION INTO SEGMENT 05.
The signature identity of Segment 04 is:
AFTER-EXPLOSION CALM + LOW-BODY CONTROL + ASYMMETRIC SINGLE-AXIS BALANCE + SONG-STYLE SHEN-YUN + WATER-SLEEVE DELAY + RIGHT-FRONT LOW-ANGLE FACIAL HERO + MEDIUM-CLOSE CHARACTER-DOMINANT CINEMATOGRAPHY + CONTINUOUS PHYSICAL TRANSITION.
```

---

## SEGMENT 05 — 11.967s（源曲全局 00:50.533–01:02.500）

```text
Use <PICTURE 1> as the ONLY complete visual authority for the exact character identity, facial identity, hairstyle, body proportions, costume, accessories, footwear, environment, lighting, photographic realism, and all visual color relationships.
Use <PICTURE 2> ONLY as a three-view identity and body-structure clarification reference for the same character. PICTURE 2 does not define the environment, lighting, color, first-frame composition, camera angle, or pose.
Use <AUDIO 1> as the ONLY temporal and rhythmic authority.
<AUDIO 1> corresponds to original track 00:50.533–01:02.500.
The music CONTINUES DIRECTLY from Segment 04.
Do NOT restart the music.
Do NOT reset the character to a neutral starting pose.
Inherit the exact physical continuation state from the final frame of Segment 04 and continue the dance from that existing momentum.
Do not describe or redesign the environment. The environment remains exactly inherited from PICTURE 1 and Segment 01.
Do not name any specific color anywhere in this prompt; all color is carried by PICTURE 1.
==================================================
CORE DANCE DIRECTOR LOGIC
Dance language:
Chinese classical dance shen-yun, refined Song-style Jiangnan movement quality, water-sleeve technique, continuous travelling footwork, circular torso pathways, delayed shoulder-to-wrist phrasing, restrained elegance, and physically grounded weight transfer.
Primary kinetic chain:
feet → supporting leg → weight shift → pelvis → torso → shoulder → arm → wrist → fingertips → sleeve.
Every action must generate the following action.
Do not reset between moves.
No idle preparation.
No isolated gesture list.
Movement quality:
continuous, elegant, precise, fast when the music requires speed, but never frantic.
The lower body carries the acceleration.
The torso remains controlled while the feet become faster.
Sleeves always inherit the momentum of the body and never move independently.
The phrase structure is:
quick travelling steps → increasing directional pressure → larger body line → airborne transition → controlled landing → immediate continuation.
This segment must feel more confident and spatially expansive than Segment 04 without becoming aggressive or acrobatic.
==================================================
GLOBAL FRAMING RULE
Keep the dancer predominantly in MEDIUM-CLOSE / CLOSE-MEDIUM framing.
The normal framing target is:
FROM LOWER CALVES OR SHINS
TO JUST ABOVE THE TOP OF THE HEAD.
Keep the character large enough in frame that facial detail remains readable.
Do NOT use distant full-body wide shots.
Do NOT sacrifice facial pixels merely to show the dance.
The camera may change position, angle, and composition, but overall viewing distance remains close.
The preferred character angle, when narratively justified, is:
RIGHT-FRONT LOW CAMERA + SLIGHT UPWARD ANGLE + THREE-QUARTER FACIAL VIEW.
Use this angle selectively at musical or emotional payoff points.
Do NOT insert it mechanically at every beat.
==================================================
00:00.000–00:00.100
CONTINUATION INTO SEGMENT 05
Continue directly from Segment 04.
Do not reset the stance.
The existing torso direction, supporting leg, sleeve momentum, and gaze direction from Segment 04 remain physically continuous.
The first lyric enters at 00:00.100.
The dancer immediately converts the inherited weight into a small forward-diagonal traveling step.
No introductory pause.
CAMERA:
Remain in the existing medium-close continuation framing.
Small natural tracking movement only if necessary to preserve the established body composition.
==================================================
00:00.100–00:01.313
LYRIC PHRASE — “菊应羞”
The movement is understated but self-possessed.
The dancer takes two short gliding steps on a diagonal.
The first step carries the weight forward.
The second step slightly crosses the supporting pathway without crossing the legs unnaturally.
The pelvis initiates the change of direction.
The torso follows one fraction later.
The shoulder remains calm.
The leading wrist draws a restrained inward sleeve arc near the torso.
The head remains slightly delayed behind the travelling body.
At the final instant before 01.313:
the dancer subtly raises the chest and allows the gaze to arrive last.
CAMERA:
Begin slightly lateral-front.
Track the travelling body rather than rotating around it.
00:01.313 = ★ STRONG BEAT.
==================================================
00:01.313 ★
CHARACTER ACCENT / FIRST VISUAL LIFT
On the exact strong beat:
the supporting foot plants decisively.
The opposite foot glides outward.
The torso opens through the waist and ribcage.
The leading arm rises in a diagonal curve.
The opposite sleeve remains lower, creating an asymmetric S-shaped body line.
Do NOT mirror the arms.
Do NOT create a symmetrical V pose.
The head lifts last.
CAMERA:
Perform a short editorial camera burst into the preferred RIGHT-FRONT LOW-ANGLE THREE-QUARTER VIEW.
The camera moves only enough to improve the facial angle while preserving the medium-close body coverage from calves to head.
At the end of the burst:
CAMERA = ZERO MOTION.
Hold the resulting hero composition briefly.
==================================================
00:01.500–00:03.288
TRAVELLING SENTENCE
Immediately continue from the 01.313 accent.
Do not hold the previous pose.
The dancer shifts into faster travelling footwork.
Sequence:
right-side weight transfer
→ short diagonal step
→ immediate left-side recovery
→ crossing pathway
→ half-step backward redirection
→ forward glide.
The torso remains calm while the feet become increasingly active.
Water sleeves remain close to the body at first, then gradually widen as the momentum accumulates.
The movement should resemble flowing movement across a narrow spatial corridor rather than walking in place.
CAMERA:
Tracking movement follows the body laterally.
Maintain medium-close distance.
Do not pull away.
==================================================
00:03.288 ◆
SECONDARY ACCENT
At the exact ◆ beat:
The dancer abruptly changes weight from the travelling foot to the opposite supporting leg.
The pelvis turns first.
The torso follows.
The rear shoulder opens.
One sleeve traces a low crescent.
The opposite sleeve passes near the waist and then releases outward.
This produces a delayed sleeve wave.
Do not stop the feet after the accent.
The next travelling step begins immediately.
CAMERA:
Very small positional correction toward the front-right side.
No dramatic motion.
==================================================
00:03.288–00:05.264
ASCENDING KINETIC SENTENCE
The dancer now travels diagonally through the frame.
Footwork becomes faster than Segment 04.
Use rapid but elegant Chinese classical travelling steps.
The supporting foot changes every phrase.
Never drag the body without real weight transfer.
The torso creates a gentle counter-direction against the legs:
legs travel one way,
chest slightly resists,
then the waist releases the stored tension into the next step.
Sleeves begin trailing behind the body more visibly.
At approximately 04.6–05.0:
the dancer prepares for a larger directional displacement by lowering the center of gravity very briefly.
This lowering is not a squat.
It is a controlled compression through ankles, knees, and hips.
==================================================
00:05.264 ★
SECOND MAJOR VISUAL EXPANSION
At the exact ★ beat:
Explode upward from the compressed support.
The dancer takes a long diagonal step.
The rear leg extends fully through the ankle and knee.
The torso lengthens.
One arm rises behind the body while the opposite sleeve sweeps forward.
The body forms a long diagonal from supporting foot through waist to fingertips.
The line remains asymmetrical.
CAMERA:
Short directional burst from lateral tracking into a slightly frontal-right position.
Immediately lock.
This is NOT the Hero Moment.
It is the launch of the Hero Moment.
==================================================
00:05.500–00:07.239
HERO BUILD
The dancer does NOT spin.
The dancer instead builds translational momentum.
Three increasingly large travelling steps:
STEP 1:
short and controlled.
STEP 2:
longer, with a sharper weight transfer.
STEP 3:
the final step strongly pushes the floor backward.
The final push originates from:
supporting foot → ankle → knee → hip → lower abdomen.
The arms remain behind the body's momentum for a fraction of a beat, then release forward.
This delay is critical.
The body moves first.
The sleeves chase the body.
CAMERA:
Track laterally.
Do not orbit around the dancer.
Do not zoom out.
==================================================
00:07.239 ◆
PRE-HERO ACCENT
On the exact ◆ beat:
The dancer performs a sudden low-to-high weight transfer.
Supporting knee bends.
Opposite foot skims forward.
Torso folds slightly through the waist.
Then the spine rebounds upward.
The head remains down during the compression.
The eyes do not look toward camera yet.
The forward arm gathers the sleeve close to the torso.
This creates the final compression before release.
CAMERA:
Move slightly lower and closer while maintaining the calves-to-head framing.
No dramatic camera spin.
==================================================
00:07.700–00:09.214
FINAL HERO PREPARATION
The dancer accelerates.
Rapid footwork:
short right step
→ immediate left weight transfer
→ diagonal crossing step
→ strong right-foot plant.
No pause.
At the final plant:
the supporting leg bends.
The hips rotate only enough to redirect the body toward a new diagonal pathway.
The torso rises.
The opposite leg is prepared for a powerful airborne exchange.
The arms open unevenly:
one arm leads,
the other remains behind,
creating delayed sleeve tension.
The entire body is visibly loading force into the supporting leg.
==================================================
00:09.214 ★
SIGNATURE HERO MOMENT 02
HIGH-SPEED TRAVELLING LEAP / AIRBORNE LEG EXCHANGE
This is the second major Hero Moment of the entire dance.
It MUST NOT resemble the previous 720-degree one-hand floor rotation.
It MUST NOT be a floor spin.
It MUST NOT be a repeated standing spin.
The Hero is defined by:
TRAVEL + AIRBORNE CHANGE OF LEG LINE + DIRECTIONAL REATTACHMENT TO THE FLOOR.
At the exact ★ beat:
The supporting foot drives hard into the floor.
The body travels diagonally across the frame.
The leading leg rises forward and outward.
The trailing leg folds under the body.
At the airborne peak:
the legs exchange their relative positions in a clean scissor-like change,
with one leg extending while the other folds,
creating a strong asymmetric airborne silhouette.
The torso remains elongated and slightly inclined rather than curled.
The head stays aligned with the travelling direction.
No backward head throw.
No uncontrolled twist.
The leading sleeve stretches diagonally behind the motion.
The trailing sleeve follows with visible inertial delay.
The body completes the airborne transition and lands on the opposite supporting foot.
LANDING ORDER:
opposite foot contacts first
→ ankle absorbs
→ knee accepts weight
→ hip receives the downward force
→ torso follows
→ arms recover last.
Immediately after landing:
the dancer continues travelling.
No landing freeze.
This is a physically grounded leap, not levitation.
CAMERA:
During 09.214–09.500:
perform a short aggressive lateral camera burst matching the travelling direction.
Do NOT orbit around the dancer.
Do NOT pull into a wide shot.
At the airborne apex:
the camera reaches the optimal medium-close composition.
At approximately 09.500:
CAMERA = ZERO MOTION.
The dancer continues the landing inside the locked composition.
Use the temporary lock to make the airborne silhouette read clearly.
==================================================
00:09.500–00:11.190
HERO RESOLUTION
After landing, the dancer does not repeat another leap or spin.
The landing energy continues into two controlled gliding steps.
The torso folds slightly forward from the hips.
The rear shoulder releases.
The leading sleeve travels downward in a long arc.
The body becomes calmer while the feet remain mobile.
The dancer then changes direction once:
feet redirect first,
waist follows,
shoulders follow,
head follows last.
The final turn is approximately one quarter-turn only.
Do NOT make it a full rotation.
At the end of the phrase:
the dancer's body settles into a diagonal forward-facing continuation state.
CAMERA:
Transition gradually through a short lateral correction into the RIGHT-FRONT LOW-ANGLE VIEW.
Keep the dancer large in frame.
The face becomes more visible as the head follows the final directional change.
==================================================
00:11.190 ◆
FINAL ACCENT OF SEGMENT 05
At the exact ◆ beat:
The front foot plants.
The back heel releases.
The torso rises from the lower body.
The leading sleeve finishes its descent.
The trailing sleeve arrives a fraction later.
The head turns toward the right-front camera axis.
Do not stop the entire body.
The foot remains ready for the next segment.
CAMERA:
Hard lock on the right-front low-angle three-quarter view.
The face is readable.
The body remains readable from lower calves to head.
No close-up face crop.
==================================================
00:11.500–00:11.967
CONTINUATION STATE
The dancer sustains the final weight distribution and continues a barely perceptible forward body drift.
Sleeve tails retain residual movement.
The gaze remains stable.
Do not create a new pose.
At 01:02.500:
the exact physical continuation state becomes the starting state for Segment 06.
==================================================
PERFORMANCE
Emotional progression:
quiet confidence → increasing assurance → spatial expansion → explosive travelling Hero → controlled composure.
Do not perform the lyrics literally.
Do not mouth lyrics.
Do not lip-sync.
Do not exaggerate facial emotion.
The face should remain elegant and composed.
The Hero moment is carried by body mechanics, timing, space, and camera rather than exaggerated expression.
==================================================
BODY MECHANICS
The entire segment obeys:
force originates in the feet
→ transfers through the supporting leg
→ pelvis redirects it
→ core stabilizes
→ torso lengthens
→ shoulder releases
→ arm follows
→ wrist finishes
→ sleeve responds last.
During the leap:
floor reaction force begins at the supporting foot.
The body cannot rise before the supporting foot creates the push.
During landing:
foot first,
then ankle,
then knee,
then hip,
then spine.
No floating.
No zero-gravity.
No body deformation.
Sleeves respond to body acceleration.
They never move independently.
==================================================
CAMERA LANGUAGE
Camera is an observer participating in the dance.
Use only purposeful camera changes:
continuation tracking
→ short reposition burst
→ brief hero lock
→ lateral tracking
→ lower-angle approach
→ final right-front low-angle lock.
Never create a continuous cinematic zoom throughout the segment.
Never pull away into a distant wide shot.
Keep the character large.
The most important facial composition is the RIGHT-FRONT LOW-ANGLE THREE-QUARTER VIEW, especially after the Hero resolves.
==================================================
OVERALL SOUNDSCAPE
Subtle diegetic physical dance sounds only:
soft foot contact,
fabric movement,
light sleeve displacement,
natural landing sound.
No added dialogue.
No added singing.
No artificial cinematic impacts.
==================================================
NON-DIEGETIC MUSIC
Use the actual supplied <Audio 1> track segment corresponding to 00:50.533–01:02.500.
The music continues directly from Segment 04.
BPM remains 121.5.
Follow the supplied timing exactly.
Critical synchronization points:
00:00.100 lyric entrance
00:01.313 ★ first major lift
00:03.288 ◆ secondary accent
00:05.264 ★ second major expansion
00:07.239 ◆ pre-Hero compression
00:09.214 ★ SIGNATURE HERO MOMENT
00:11.190 ◆ final accent
00:11.967 segment-end continuation.
No independently invented rhythmic structure.
Do not replace Audio 1 timing with generic cinematic timing.
==================================================
FINAL CONTINUITY RULE
At 01:02.500 the dancer MUST NOT be standing in a neutral reset.
The final state must retain:
forward-diagonal momentum,
one side carrying slightly more weight,
feet ready to travel,
sleeve tails still moving,
gaze already oriented for the next phrase.
Segment 06 begins from this exact physical continuity.
```

---

## SEGMENT 06 — SIGNATURE FINAL HERO · 12.968s（源曲全局 01:02.500–01:15.468，16:9）

```text
SEGMENT 06 — SIGNATURE FINAL HERO
GLOBAL AUDIO RANGE: 01:02.500–01:15.468
LOCAL VIDEO TIME: 00:00.000–00:12.968
FORMAT: 16:9 realistic live-action cinematic dance video
# ================================================== INPUT AUTHORITY
<Picture 1> is the PRIMARY and SOLE visual authority for the character and the complete existing visual world.
Use <Picture 1> for:
facial identity,
hair,
body proportions,
costume construction,
accessories,
footwear,
existing lighting,
existing photographic realism,
and the complete environment/background.
Do not redesign the character.
Do not reinterpret any visible appearance characteristic.
The environment is inherited directly from <Picture 1> and remains unchanged.
Do not describe or invent a new scene.
Do not introduce a second environment.
Do not import the background from any other image.
<Picture 2> is a THREE-VIEW IDENTITY CLARIFICATION reference only.
Use it only to resolve character structure and identity consistency when needed.
<Picture 2> does NOT provide:
environment,
background,
lighting,
starting frame,
camera composition,
or a new costume design.
<Audio 1> is the ONLY temporal and rhythmic authority.
The dance, action accents, camera events, Hero Moment, facial timing, editorial transitions, and ending hold must all synchronize to <Audio 1>.
Do not invent an independent rhythm.
# ================================================== CORE DIRECTORIAL CONCEPT
This is the FINAL SEGMENT of one continuous Chinese classical dance performance.
The subject must continue directly from Segment 05.
This is NOT a reset.
Do not return to a neutral standing pose.
The opening physical state inherits the exact travel momentum and body orientation established at the end of Segment 05.
Dance language:
PRIMARY Chinese classical dance shen-yun,
with refined water-sleeve vocabulary,
continuous weight transfer,
rapid footwork,
controlled low-level floor technique,
and one advanced shoulder/back-axis rotational Hero sequence.
Movement chain:
travel
→ acceleration
→ directional change
→ delayed head turn
→ deceleration
→ controlled low descent
→ shoulder/back-axis windmill sweep
→ scissor-leg circular momentum
→ skirt blooms across the floor
→ reverse momentum recovery
→ low kneeling / seated spiral
→ final upper-body opening
→ right-front low-angle facial Hero
→ absolute stillness.
The final Hero must feel earned through accumulated momentum.
Do NOT make the Hero appear as a random acrobatic insertion.
# ================================================== MUSICAL STRUCTURE
AUDIO 1 corresponds to original track:
01:02.500–01:15.468
AUDIO 1 continues directly from Segment 05.
The music does NOT restart.
BPM:
121.5 BPM
Fine beat:
approximately 0.494 seconds
Slow-body pulse:
approximately 0.988 seconds
★ major structural accents:
01:03.698 global / local 00:01.198
01:07.648 global / local 00:05.148
01:11.599 global / local 00:09.099
◆ secondary accents:
01:05.673 global / local 00:03.173
01:09.624 global / local 00:07.124
01:13.574 global / local 00:11.074
Lyrics:
00:00.100–00:04.150
“何事当年不见收”
00:04.150–00:08.433
“不见收不见收”
After 00:08.433:
instrumental tail / emotional resolution.
The lyrics are not acted out literally.
The dance expresses their emotional progression through weight, direction, hesitation, return, and controlled release.
# ================================================== FRAMING PRINCIPLE
Maintain a close-medium cinematic scale throughout the entire segment.
The subject should occupy most of the frame.
Primary framing target:
lower legs / calves through the top of the head.
Do not rely on distant wide shots.
Preserve maximum facial pixel density while still showing enough of the lower body to read footwork and low-level movement.
The camera may become extremely low during the Hero, but the subject must remain large and readable.
The preferred facial composition is:
CAMERA POSITION:
front-right of the subject,
below the face,
slightly forward,
low-angle upward perspective,
three-quarter facial view.
This RIGHT-FRONT LOW-ANGLE FACIAL LANGUAGE is a recurring visual signature of the dance, but it must only appear when motivated by the choreography.
Never insert it mechanically.
# ================================================== 0.000–1.198 CONTINUATION / CONTROLLED TRAVEL
Begin directly from the exact physical continuation state of Segment 05.
The subject is still moving.
Carry the previous segment's travel momentum forward with no reset.
The first action phrase is:
rapid small-foot travel
→ immediate weight transfer
→ crossing step
→ slight directional redirection
→ upper body remains controlled
→ sleeves trail behind the body
→ head follows late.
The lower body is active while the upper torso remains calm.
The dancer takes two compact, precise traveling steps.
The second step does not stop.
The body already loads the following direction change.
At 00:01.198 ★:
Complete a strong directional redirection.
The supporting foot catches the floor.
The pelvis rotates first.
The torso follows one fraction later.
The shoulders remain open.
The head turns last.
This is the first important visual accent.
It is NOT yet a Hero.
CAMERA:
medium-close tracking,
slightly below normal eye level,
front-right three-quarter perspective.
The camera follows the body's lateral travel rather than floating independently.
At the exact ★ accent, make a small editorial camera compression toward the subject.
No large zoom.
Immediately after the accent, return to normal close-medium tracking.
# ================================================== 1.198–3.173 “何事当年不见收” FIRST INTERNAL TURN
The subject continues traveling after the 01:03.698 accent.
Do not stop after the directional redirection.
The next phrase is:
cross-step
→ weight passes through the supporting leg
→ pelvis rotates
→ ribcage delays
→ sleeve follows
→ head turns last.
At 00:02.2:
the torso begins a soft inward contraction.
At 00:02.6:
the pelvis has already redirected while the shoulders remain slightly behind.
At 00:03.173 ◆:
Finish the phrase with a compact internal turn.
One sleeve passes across the body's side.
The other sleeve trails behind.
The action should read as a single continuous Chinese classical dance sentence, not separate gestures.
CAMERA:
track laterally at the same distance.
Do not pull away.
As the body turns, the camera shifts slightly toward the subject's right-front side so that the next facial angle becomes available.
# ================================================== 3.173–5.148 CONTROLLED EXPANSION / FIRST MAJOR FACIAL ANGLE
Continue directly from the turn.
The body opens outward through the legs first.
The supporting foot presses into the floor.
The opposite foot travels forward on a shallow diagonal.
The pelvis opens.
The spine lengthens.
The chest releases.
The sleeves follow through a delayed outward arc.
At 00:04.150 the lyric changes into the repeated phrase:
“不见收不见收”
The choreography becomes more internally restrained while the lower body continues moving.
Do NOT mime the lyric.
At 00:04.55:
the dancer takes one longer diagonal step.
At 00:04.85:
the supporting leg bends slightly and receives the body weight.
At 00:05.148 ★:
Perform the second major body accent.
The dancer executes a full-body back-turn with delayed head timing:
supporting foot
→ pelvis
→ torso
→ shoulder
→ neck
→ head
→ eyes.
The head arrives last.
This creates the first clearly readable right-front low-angle facial moment.
CAMERA:
from 00:04.90–00:05.148 perform a short directional camera burst toward the front-right lower position.
The camera drops slightly below the dancer's face and rotates only enough to establish the preferred upward perspective.
At 00:05.148:
CAMERA = HARD LOCK for the short Hero portrait beat.
The face appears in a natural three-quarter angle.
The dancer does not pose artificially.
The body is still carrying residual travel.
# ================================================== 5.148–7.124 MOMENTUM REBUILD
Immediately after the 00:05.148 accent:
Do not stop.
The dancer releases the previous turn into a faster travel phrase.
ACTION CHAIN:
right-foot catch
→ immediate opposite-foot transfer
→ compact cross-step
→ low rebound through the supporting knee
→ torso inclines slightly
→ sleeve follows on a delayed wave
→ direction changes again.
Use rapid footwork.
The upper body remains elegant and stable.
The dancer travels on a shallow diagonal, covering space without leaving the close-medium camera scale.
At 00:06.1:
the body begins lowering its center of gravity.
At 00:06.5:
the knees bend deeper.
At 00:06.8:
the pelvis moves closer to the floor.
This is the beginning of the Hero preparation.
Do not abruptly kneel.
The descent must be generated by accumulated horizontal momentum.
At 00:07.124 ◆:
Execute a quick low cross-step and let the trailing leg sweep behind the body.
The torso turns partially away from the camera.
The head remains forward for a brief fraction of a beat.
The delayed head return establishes the final emotional phase.
CAMERA:
tracking continues but gradually lowers with the dancer.
Still keep the frame within close-medium scale.
Do NOT widen into a distant floor shot.
# ================================================== 7.124–8.433 HERO PREPARATION / DISSIPATING SPEED
This is the critical preparation period.
The previous phrase is fast.
Now the dancer does NOT simply slow down.
Instead:
speed is converted into rotational momentum.
ACTION:
rapid final traveling step
→ supporting foot lands decisively
→ opposite foot sweeps around
→ hips lower
→ one knee approaches the floor
→ torso folds diagonally
→ shoulder prepares to receive weight.
At 00:07.65:
the dancer takes the final fast step.
At 00:07.95:
the front foot becomes the brake.
At 00:08.15:
the opposite knee begins a controlled floor entry.
At 00:08.35:
the torso rotates toward the shoulder/back axis.
At 00:08.433:
The lyric ends.
The body has already entered the physical setup for the Hero.
There must be no pause.
The end of the vocal phrase is converted directly into the Hero's kinetic pickup.
# ================================================== 8.433–9.099 SIGNATURE HERO ENTRY
This is the single most important action of the segment.
The Hero is:
# SHOULDER-BACK-AXIS WINDMILL → FISH-LIKE LOW SWEEP
This Hero is intentionally different from the previous Heroes.
It is NOT:
a one-hand 720-degree floor spin,
a vertical inversion,
or a jumping / airborne Hero.
The movement is grounded and circular.
It derives power from:
final foot drive
→ hip drop
→ torso rotation
→ shoulder-blade / upper-back contact
→ continuous leg sweep
→ rotational inertia.
The dancer enters a controlled low windmill-like rotation around the outer upper back / shoulder region.
One leg extends long through the rotation.
The other leg folds inward and then releases.
The legs create a large scissoring circular path.
The pelvis follows the sweeping legs.
The torso stays connected to the axis.
The head remains protected and naturally aligned with the spine.
No isolated head whipping.
The action must look physically supported by the floor.
### SKIRT MOTION
The skirt remains fixed at the waist.
As the body rotates close to the floor, the skirt spreads outward horizontally from the waist.
The fabric sweeps along the floor in a broad circular flower-like geometry.
The inner layers respond first.
The outer layers follow with visible rotational delay.
The hem traces the circular momentum.
The skirt remains low and floor-bound.
It does not rise over the face.
It does not become an airborne umbrella.
The human body remains the center of the circular fabric field.
### 00:08.75–00:09.099
The legs complete the strongest part of the first circular sweep.
At 00:09.099 ★:
# SIGNATURE HERO PEAK
The dancer reaches the maximum spatial extension of the windmill sweep.
One leg extends in a long diagonal line.
The opposite leg remains folded inward.
The torso is strongly diagonal.
The skirt reaches its widest horizontal floor spread.
The body and skirt form a single large radial composition.
The face rotates toward the camera.
The face is readable in a three-quarter angle from the floor.
This is the ONE unmistakable Hero image of the ending.
# ================================================== 9.099–10.250 HERO CONTINUATION / SECONDARY SWEEP
Do not stop immediately after the 00:09.099 ★.
The Hero must have after-momentum.
Continue the shoulder/back-axis rotation through a second partial sweep.
The extended leg folds inward.
The second leg releases.
The pelvis follows.
The torso rolls from the outer shoulder/back toward a low side-facing position.
The circular skirt remains spread and continues sliding across the floor.
The outer hem lags behind the body's rotation.
This second sweep is smaller than the first.
Its purpose is not to create another Hero.
Its purpose is to convert the Hero's energy into a controlled recovery.
CAMERA:
00:09.099–00:09.700
CAMERA = ABSOLUTE HARD LOCK.
The audience must see the body finish the Hero without camera interference.
No orbit.
No zoom.
No push.
No pull.
The dancer creates the motion.
00:09.700–00:10.250:
A very small floor-level camera reposition is permitted only to keep the recovering face and lower leg inside the close-medium composition.
The move is slow and secondary.
The subject remains dominant.
# ================================================== 10.250–11.074 HERO RESOLUTION / LOW SPIRAL
Use the remaining rotational momentum.
The dancer does not stand.
Instead:
sweeping leg retracts
→ pelvis rotates underneath
→ torso coils
→ supporting hand briefly assists the floor transition
→ body rises from shoulder/back support into a low kneeling spiral.
The assisting hand is only a transitional support point.
It does not become a second floor-spin mechanism.
The dancer arrives in a refined low seated / kneeling classical position.
One lower leg extends forward enough to remain visible.
The other leg folds beneath the body.
The torso rises from the pelvis.
The spine articulates upward from bottom to top.
The head is the final body segment to rise.
At 00:11.074 ◆:
Complete the low spiral.
The sleeves settle around the body.
The skirt finishes its circular slide and begins settling around the waist and folded legs.
The dancer's head turns toward the front-right.
This is the emotional release after the Hero.
# ================================================== 11.074–12.968 FINAL FACE HERO / ABSOLUTE END FRAME
This is the FINAL VISUAL PAYOFF.
The dancer is now almost completely still.
CAMERA:
front-right,
low camera position,
below face level,
gentle upward angle,
three-quarter face,
close-medium framing.
The framing remains large enough that:
lower leg
→ torso
→ face
all remain readable.
The face receives the greatest pixel density.
At 00:11.20:
The dancer's eyes are no longer moving rapidly.
The gaze slowly settles into the lens.
At approximately 00:11.45:
The chin rises by a very small amount.
Not a dramatic head lift.
Just enough to open the face toward the low camera.
At 00:11.65:
The final facial composition is established.
CAMERA = HARD LOCK.
From 00:11.65–00:12.968:
NO CAMERA MOTION.
The body remains almost completely still.
Only natural residual motion remains:
very subtle breathing,
tiny sleeve settling,
minor hair movement,
small fabric after-motion.
The emotional tone is controlled and serene.
The subject does not smile dramatically.
The final feeling is:
the dance has finished,
but its momentum is still faintly alive in the fabric.
# ================================================== CAMERA / EDITORIAL MAP
00:00.000–00:03.173
RIGHT-FRONT CLOSE-MEDIUM TRACKING
Follow the lower-body travel while preserving the face.
00:03.173–00:04.900
DIAGONAL FOLLOWING SHOT
Slightly adjust the camera toward the dancer's front-right side.
Keep the horizon natural.
Do not widen.
00:04.900–00:05.148
SHORT CAMERA BURST
Move into the preferred right-front low-angle facial perspective.
Very short, decisive reframing.
At 00:05.148:
CAMERA LOCK.
00:05.148–00:06.900
LOCKED / VERY SMALL FOLLOW
Let the body move through the composition.
The camera does not chase every sleeve motion.
00:06.900–00:08.433
LOWER TRACKING
Camera gradually lowers as the center of gravity lowers.
Maintain close-medium scale.
00:08.433–00:09.099
FLOOR-LEVEL HERO APPROACH
A short, decisive camera burst toward the right-front lower position.
The final Hero composition becomes readable before the ★ peak.
00:09.099–00:09.700
ABSOLUTE HARD LOCK
This is the most important camera lock of the entire segment.
The dancer's Hero action must create the visual impact.
00:09.700–00:10.250
SUBTLE RECOVERY REFRAME
Only enough movement to preserve the subject within the close-medium composition.
10.250–11.074
LOW-LEVEL FOLLOW
The camera follows the recovery spiral at very low speed.
11.074–11.650
SHORT FINAL FACIAL REFRAME
Settle precisely into the right-front low-angle facial Hero composition.
11.650–12.968
ABSOLUTE STATIC FINAL FRAME
No movement.
No zoom.
No drift.
No hidden motion.
# ================================================== BODY MECHANICS
All movement follows the causal chain:
feet
→ supporting leg
→ weight transfer
→ pelvis
→ lower spine
→ ribcage
→ shoulder
→ arm
→ wrist
→ fingertips
→ sleeve.
The Hero is driven from the lower body.
Never pull the body around by the head or by the arms.
The windmill movement is physically supported by the shoulder-blade / upper-back contact with the floor.
The legs create the primary circular momentum.
The pelvis follows the legs.
The torso stays mechanically connected to the axis.
The final recovery uses the remaining leg momentum rather than suddenly standing.
The final facial movement comes after the torso has already settled.
# ================================================== SLEEVE AND FABRIC PHYSICS
Sleeves respond to body movement.
They do not move independently.
During rapid travel:
sleeves trail behind the arms.
During the 00:05.148 back-turn:
the sleeve lags behind the torso rotation.
During the floor Hero:
the sleeves sweep outward from centrifugal motion and then settle after the body slows.
During the final static frame:
the sleeve tails continue only with tiny residual motion.
The full-circle skirt remains attached at the waist.
During the floor Hero:
waist rotation
→ inner skirt layer
→ middle layer
→ outer layer
→ hem.
The fabric spreads horizontally across the floor as rotational momentum increases.
The skirt forms a broad circular flower-like sweep around the dancer.
The fabric does not detach from the waist.
The body remains above the fabric opening.
During recovery, the skirt loses rotational spread and settles around the folded legs naturally.
Never animate the skirt as an independent object.
# ================================================== EDITORIAL RHYTHM
The segment must feel like one continuous dance sentence.
Primary editorial accents:
00:01.198 ★
directional transformation
00:03.173 ◆
internal turn
00:05.148 ★
large delayed back-turn and facial reveal
00:07.124 ◆
low-level directional compression
00:08.433
vocal phrase ends and kinetic Hero preparation is already underway
00:09.099 ★
SIGNATURE HERO PEAK
00:11.074 ◆
final low spiral resolution
00:11.650
FINAL FACIAL HERO FRAME ESTABLISHED
12.968
FINAL LOCK.
Do not cut on every beat.
Cuts occur only when the body changes direction, the camera changes visual role, or the Hero requires a new composition.
The final Hero must therefore feel like a culmination rather than another ordinary beat.
# ================================================== PERFORMANCE
Emotional progression:
restless forward movement
→ restrained return
→ internal turning
→ loss of speed
→ surrender of weight into the floor
→ powerful circular Hero
→ controlled recovery
→ serene final gaze.
The face remains calm and dignified.
Do not exaggerate emotion.
Do not perform the lyrics literally.
The emotional climax is expressed through the contrast between:
high-speed lower-body momentum
and
increasing upper-body stillness.
The final gaze is quiet and direct.
The final frame should feel like the entire previous 75 seconds have finally come to rest in one image.
# ================================================== AUDIO 1 SYNCHRONIZATION RULE
AUDIO 1 is the exact supplied music track.
Every specified event is synchronized to AUDIO 1.
00:01.198
major directional accent
00:03.173
secondary body accent
00:05.148
major turn / facial reveal
00:07.124
secondary compression
00:08.433
vocal phrase ending releases the Hero preparation
00:09.099
SIGNATURE HERO PEAK
00:11.074
secondary resolution accent
00:11.650
final facial lock
00:12.968
segment end.
No independent timing system.
No generic cinematic rhythm.
The supplied audio controls the event timing.
# ================================================== CONTINUATION STATE
At 00:12.968:
the dancer is in the final low seated / kneeling resolved position.
Weight is fully settled.
The torso is upright but relaxed.
The face remains turned toward the front-right low-angle camera.
The gaze is stable.
The sleeves have nearly finished settling.
The skirt has completed its floor sweep and rests naturally around the lower body.
This is a TRUE FINAL STATE.
Do not create a new movement after the final frame.
# ================================================== SELECTIVE NEGATIVE CONSTRAINTS
Only one major floor Hero in this segment.
Do not replace the Hero with a random standing spin.
Do not reuse the previous one-hand 720 floor-spin mechanism.
Do not reuse the previous airborne displacement Hero.
Do not introduce extra people.
Do not introduce a new scene or background.
Do not let the skirt become detached from the waist.
Do not let the skirt or sleeves cover the face during the Hero.
Do not let the camera become distant during the floor movement.
Do not turn the final facial payoff into a full-face flat frontal angle.
Do not add singing, lip-sync, dialogue, subtitles, text, or logos.
The final visual structure must clearly read as:
CONTINUED TRAVEL
→
DIRECTIONAL TURN
→
LOWERED CENTER OF GRAVITY
→
SHOULDER-BACK-AXIS WINDMILL ENTRY
→
LARGE FLOOR SKIRT SWEEP
→
00:09.099 SIGNATURE HERO PEAK
→
CONTROLLED LOW SPIRAL RECOVERY
→
RIGHT-FRONT LOW-ANGLE FACIAL HERO
→
ABSOLUTE FINAL LOCK.
```



---

## FILE: case-studies/2026-09-three-person-prank-freeze-gag-cos-swap/README.md (https://github.com/LoveRain1997/h3-prompt-journal/blob/HEAD/case-studies/2026-09-three-person-prank-freeze-gag-cos-swap/README.md)

# Case Study 009 — Reverse-Engineered Three-Person Prank with Freeze-Frame Gags + Skin-Only Cos Swap

## Inputs

- **No reference image** — pure **T2VA** rebuild
- Source: a 15-second real UGC vertical prank clip (9:16), reverse-engineered frame-by-frame
- Target: **~15 seconds**, vertical **9:16 / 24fps**, photorealistic live-action (real human cosplayers, NOT anime)
- Two characters reskinned to *Detective Conan* cosplay: Girl A = Hagiwara Chihaya (traffic-police cos), Girl B = Elena Miyano (scientist cos)
- Prompt language: **English** structured sections with two short spoken **Japanese** exclamations
- Result render: [`result-video.mp4`](./result-video.mp4) (0.8-resolution, with audio)

## The problem

This is the journal's first **reverse-engineering** entry — the prompt was not designed from scratch, it had to be recovered from an existing clip. Three failure modes appeared on the first naive pass:

1. **The invisible third actor.** The camera-holder is also the prankster, but only a green sleeve, a green scooter fairing and an occasional hand ever enter frame. A sparse frame walk reads those as clutter and treats the camera as a neutral observer — which inverts the whole plot.
2. **The one-to-many inverse problem.** Different scripts produce near-identical frames. At low sampling rate, "she sprays herself" and "an off-camera hand turns her head so she sprays her friend" look the same; the 3–5-frame helmet-grab that decides between them disappears entirely.
3. **Freeze frames mistaken for acting.** The three half-second shock holds are *post-production*, not people freezing on set. Writing them as action freezes the characters instead of the edit.

## The breakthrough

**Five-channel forensics + a frozen causal skeleton.**

Dense 4–8fps extraction over each ambiguous window adjudicated between competing plot hypotheses instead of confirming the most common one; a 0.5s RMS envelope located three energy peaks, and every peak was reconciled with a picture cause; a BPM false-detection guard (onset regularity + spectrogram kick grid) correctly ruled that the track has **no music — pure field audio**. What emerged is a strict THREE-BEAT loop:

```
rider provokes → girl charges a mouthful → rider physically sabotages →
backfire lands → FREEZE-FRAME shock gag   (×3, escalating)
```

The edit layer was then separated cleanly: one unbroken handheld take with exactly three ~0.5s freeze holds, each marked by a micro punch-in + one-frame white flash + a record-scratch→impact→vacuum post-SFX.

The **cos swap** was done last, under a "skin-only" contract: timestamps, actions, camera, freeze points and sound structure are frozen; only appearance, costume, and the wet-hair/fabric physics they touch are rewritten. A compatibility check handled the one real dependency — the flip-up visor the gags rely on — and character-specific props (lab gear, drawn duty equipment) were explicitly barred from frame so the setting never changes.

| Technique | Role |
|---|---|
| **Dense discriminative sampling** | 4–8fps on ambiguous windows to choose between candidate plots |
| **Audio-visual reconciliation** | each RMS peak must have a visible cause; BPM false-positive guard |
| **Off-camera actor tracking** | sleeve / hand / fairing merged into one real agent (Subject 3) |
| **Layer separation** | live action / camera / edit-effects / sound written into different fields |
| **N-Beat escalation** | the same sabotage loop repeated three times, escalating, in `editing` |
| **Freeze-frame four-tuple** | in-point, duration, overlay, snap-back — never "the actor stops" |
| **Skin-only reskin** | freeze the causal skeleton; rewrite appearance + touched physics only |

## The named language

> **Freeze-Gag Causal Loop with Skin-Only Reskin**

The deliverable is not a description of frames but an executable cause→effect chain a different cast can wear. Once the loop (provoke → charge → sabotage → backfire → freeze) is locked, swapping two original girls for two cosplayers touches `subject_definitions`, `costume_hair_physics`, and the handful of costume words an action physically reaches — and nothing else.

## 48-hour takeaway

When reversing footage, **sampling density is a discriminator, not a quality dial**: sparse frames are for building candidate hypotheses, dense frames are for killing the wrong ones. And treat the camera as a *potential actor* until proven otherwise — a purposeful moving object in frame (a hand, a sleeve, a fairing) always has a controller.

When reskinning, freeze the mechanism and rewrite the surface; the moment a costume change starts moving timestamps or beats, it has stopped being a reskin. A replacement table (appearance before/after, plus every carried-over change) is what keeps that boundary honest.

This entry is the worked example produced by the companion skill [`video-to-h3-prompt`](https://github.com/LoveRain1997/video-to-h3-prompt) — its five-channel pipeline, freeze-frame recipe and skin-only replacement SOP are documented there.

## The final prompt

See [`prompt.md`](./prompt.md) for the full paste-ready prompt: short Chinese usage notes followed by the complete English 14-field prompt.

Field order:

1. **subject_definitions** — three subjects; Subject 3 is the never-fully-shown rider/saboteur
2. **environment_definition** / **lighting_definition** — night residential alley, cool frontal phone fill
3. **integrated_multimodal_description** — `[Shot 1]` + strictly increasing timestamps, three-beat loop with embedded diegetic sound
4. **camera_direction** — handheld first-person POV, sabotage-motivated jolts, freeze punch-ins
5. **editing** — one take + exactly three white-flash freeze gags
6. **performance** / **body_mechanics** / **costume_hair_physics**
7. **negative_constraints** / **overall_soundscape** / **non_diegetic_music** (explicit silence)

## Result

A ~15-second single-take-feel vertical prank clip:

- Rider pours water on A's boot; A charges a mouthful, has her visor slapped down, is interrupted and sprays her own visor — **freeze 1**
- A recharges, has her helmet twisted toward B and accidentally sprays B's face/glasses — **freeze 2**
- B chugs to retaliate, is blocked mid-attack and chokes — **freeze 3**
- Both dissolve into laughter and link arms to camera; no music, only field audio and the three comedic SFX hits

## Assets

| File | Role |
|---|---|
| [`prompt.md`](./prompt.md) | Full final T2VA prompt (paste-ready) |
| [`result-video.mp4`](./result-video.mp4) | Generated result video, 9:16, ~15s, with audio |

## Tags

`#H3` `#prompt-engineering` `#reverse-engineering` `#video-to-prompt` `#first-person` `#off-camera-actor` `#three-beat` `#freeze-frame` `#comedy-timing` `#cos-reskin` `#field-audio` `#vertical-video` `#T2VA`


---

## FILE: case-studies/2026-09-three-person-prank-freeze-gag-cos-swap/prompt.md (https://github.com/LoveRain1997/h3-prompt-journal/blob/HEAD/case-studies/2026-09-three-person-prank-freeze-gag-cos-swap/prompt.md)

# 三人第一人称恶作剧 · 三 Beat 反击被打断 + 定格 Shock Gag（cos 换皮版）提示词

## 使用方法

纯文本 **T2VA** 提示词，无需参考图；竖屏 **9:16 / 约 15 秒 / 24fps**，真人实拍质感（real-human cosplayers，非动画、非赛璐璐）。

本提示词由一段真实 UGC 恶作剧样片**反推**得到因果动作链，再在「只换皮、不动骨」的前提下把两位女生替换为《名侦探柯南》萩原千速（Girl A，女警 cos）与宫野艾莲娜（Girl B，科学家 cos）：全部时间戳、动作、运镜、三次定格与声音结构保持不变，仅改外貌、服装与对应的湿身/二次运动物理。生成结果见同目录 `result-video.mp4`（0.8 分辨率带音频版）。

结构要点：不可见的拍摄者（Subject 3，只露绿色袖子与绿色车头）既是机位又是全程操控者；三次「含一口水准备反击 → 被拍摄者外力打断」构成 THREE-BEAT，每个 beat 落在一个约 0.5s 的白闪定格 shock gag 上。所有液体都是普通饮用水，不是血。

---

# subject_definitions

<Subject 1> (Girl A / 萩原千速) is a slim young East-Asian woman around 20, with long straight light brown / flaxen hair, the long hair flowing backward with visible natural movement, forehead and eyebrows clearly exposed, vivid blue eyes, mature, cool, composed and confident womanly presence. She wears a dark navy traffic-motorcycle police uniform jacket, a white high-neck inner top, matching dark navy trousers, a black wide-duty belt, realistic shoulder insignia, chest badge, sleeve patch and police badge details, thin black gloves, and black knee-high motorcycle leather boots. She wears a black motorcycle half-helmet with a flip-up clear visor for this prank scene, while retaining the recognizable Hagiwara Chihaya police-cos identity. She is the prank target who keeps trying to spray back. She holds the same matte-black tumbler used by Girl A in the original action design. No additional props are introduced.

<Subject 2> (Girl B / 宫野艾莲娜) is a young East-Asian woman around 20 with long straight pale-golden / light flaxen hair, softly side-parted with smooth strands falling naturally around her shoulders, deep green eyes, thin purple-framed oval glasses, fair natural skin, and a quiet, gentle, cool, intelligent presence with a subtle fragile quality. She wears a clean white long-sleeve laboratory coat with a simple chest pocket and no print, a pale plain crew-neck long-sleeve inner top, dark fitted trousers, dark short socks and black low-heeled leather shoes. She retains the recognizable Elena Miyano scientist-cos identity while participating in the exact same prank actions as Girl B. She holds the same small bottle early and the same blue plastic drink bottle later. No laboratory props or additional costume accessories are introduced.

<Subject 3> (the camera-holder/rider) is NEVER fully shown: seated on a green scooter whose green front fairing anchors the very bottom edge of frame, wearing a green sleeve; one arm repeatedly reaches into frame from camera position to interfere — first holding a clear plastic cup, later an empty green-sleeved hand and arm. This is a three-person prank video where Subject 3 films AND sabotages both girls. Vertical 9:16.

environment_definition: A narrow residential alley at night. Behind the girls: a pair of black ornate wrought-iron double gates with geometric lattice cutouts and small square plaques; to the right a grey brick wall with exposed conduit and a utility box. Damp concrete pavement that darkens where liquid is poured. The green scooter front fairing sits at the bottom foreground. Tight authentic night-street atmosphere, no shops, no signage, no bystanders.

lighting_definition: Low-light night scene lit mainly by a hard slightly cool-white frontal light from the camera position (phone-camera fill look): flat on faces, rapid falloff into deep shadow on the gate and alley, faint wet sheen on the ground. Visible low-light grain. No warm grade, no neon, no colored lighting; illumination stays constant.

integrated_multimodal_description: One continuous handheld prank clip built as a THREE-BEAT comedic structure: the unseen rider pours water on Girl A's foot, A twice tries to spray the rider back and is physically sabotaged each time, her second spray is redirected onto Girl B, then B's retaliation is also sabotaged; every beat ends on a freeze-frame shock gag. All liquid is plain pale drink water, never blood.

[Shot 1] Both girls stand side by side facing the camera in a medium-close slightly high-angle shot from upper thighs up; A is slightly forward at left, visor flipped up, black tumbler in hand; B half a step behind at right holding a small bottle with a deadpan sidelong glance; green scooter fairing at the bottom edge.

[00:00.5] The camera tilts down in an unsteady small dip toward A's feet as the green-sleeved arm enters from the lower right holding a clear cup.

[00:01.2] The cup tilts and pours water directly onto A's right foot and the pavement, audible pouring water, dark wet patch spreading around the toe of her black leather boot.

[00:01.8] The arm withdraws and the camera tilts back up, re-framing both girls.

[00:02.5] A opens her mouth wide toward the lens in an angry-playful shouted threat: 「ちょっと！」 B stays deadpan.

[00:03.5] A raises the black tumbler and takes a large sip, CHEEKS PUFFED as she holds the water in her mouth to charge a counter-spray. She keeps her clear visor FLIPPED UP while charging and faces the camera, preparing to spray.

[00:04.8] Just before A can spray, the green-sleeved hand of Subject 3 enters from the camera position and deliberately slaps/pushes A's clear visor DOWN over her face from the outside, with a small plastic click. A does NOT touch or operate the visor herself.

[00:05.4] A immediately leans forward and attempts to spray the camera while her visor is now unexpectedly lowered. At the same moment, the green-sleeved hand shoots in from the top edge and presses down on her helmet, interrupting her.

[00:05.8] Startled, A loses control of the mouthful and expels it straight into her own lowered visor; white spray bursts across the clear plastic with fine droplets ricocheting onto her hair and navy police uniform as her upper body jolts backward.

[00:06.2] FREEZE-FRAME GAG 1: the image locks for roughly half a second on A's visor plastered with white spray, a tiny punch-in zoom and a single white flash mark the frozen frame, comedic shock hold.

[00:06.8] Motion resumes: A leans back reacting, B glances at her with flat disapproval.

[00:07.2] A shoves the smeared visor BACK UP over the helmet, fully reopening the clear visor, wet strands on her cheeks, looking defiant and determined. The visor remains UP from this moment onward until the second spray is redirected toward B.

[00:08.3] A takes a second, bigger sip from the tumbler, cheeks puffed again, charging a second counter-spray with the visor still fully UP.

[00:08.9] A turns her head toward the camera/rider, lips pursed, visor still UP, about to fire.

[00:09.1] The green-sleeved arm lunges into frame from below.

[00:09.5] The rider's hand grabs the side of A's helmet and forcibly turns her head to the RIGHT, toward Girl B. The visor remains UP and is not touched or moved by the rider.

[00:09.8] A cannot hold the mouthful any longer and sprays it sideways directly into B's face at close range, a fan of water exploding across B's cheeks, closed eyes and the lenses of her purple-framed glasses, droplets suspended in the air between them.

[00:10.1] FREEZE-FRAME GAG 2: the image locks for roughly half a second on B's splash-covered face and A's head being pushed sideways, small punch-in zoom and white flash, comedic shock hold.

[00:10.6] Motion resumes: the green arm withdraws, B flinches and squeezes her eyes shut wiping water off her glasses and cheeks, A is left facing her guiltily. B reacts with a short angry line: 「もう！」.

[00:11.0] B grabs the blue plastic bottle.

[00:11.8] B tilts her head far back and chugs several large gulps to charge her own retaliation, throat pumping.

[00:12.7] Just as B lowers her head to spray, the green-sleeved arm shoots in from the right edge and blocks her — hand catching the bottle/her chin area.

[00:12.9] B is interrupted mid-attack and chokes, water spurting from her lips and bottle mouth in a small messy burst.

[00:13.1] FREEZE-FRAME GAG 3: the image locks for roughly half a second on B head-tilted with water dribbling down her chin and the green arm arresting her, small punch-in zoom and white flash, comedic shock hold.

[00:13.6] Motion resumes: B drops her head coughing and laughing, A breaks into laughter beside her.

[00:14.2] Soaked, the girls settle back side by side and A loops her arm through B's arm; B keeps the blue bottle, A keeps the black tumbler.

[00:15.3] They look into the camera, shoulders swaying in residual laughter, ending on the pair.

camera_direction: Continuous handheld first-person POV from the rider seated on the scooter, medium-close, slightly high angle. Subtle organic handheld shake throughout, never gimbal-smooth. [00:00.5–00:01.8] small unsteady tilt-down to the feet and tilt-back-up, motivated by the rider pouring water. At each sabotage moment [00:04.8], [00:05.4], [00:09.1–00:09.5], [00:12.7] the image jolts slightly from the rider's arm movement; each freeze-frame carries a tiny punch-in zoom. Otherwise hold near-static framing. No orbit, no 360, no rack focus, no slow motion outside the explicit freeze holds.

editing: One unbroken live-action take, but with three deliberate comedic FREEZE-FRAME holds layered on top at [00:06.2], [00:10.1], [00:13.1], each about 0.5 seconds, each accented by a micro punch-in zoom and a one-frame white flash, then cleanly resuming into normal-speed motion — a classic three-beat shock-gag rhythm that escalates self-backfire → accidental hit on the friend → choked retaliation. No ordinary hard cuts, no dissolves, no speed ramps; the only editorial devices are these three escalating freeze gags.

performance: Three-beat escalation. Girl A (Chihaya cos): playful anger at the poured water (wide-mouth threat), determined puffed-cheek concentration while charging water, startled jolt on the first sabotage, stubborn retry, guilty wide-eyed reaction when redirected onto B, ending in helpless laughter — her cool police-cos composure repeatedly cracking. Girl B (Elena cos): deadpan, gentle and unbothered at the start, sudden victim flinch and squeezed eyes behind wet glasses when sprayed, quiet vengeful determination while chugging, choked surprise on the third sabotage, dissolving into soft laughter. The rider never performs facially; only the green-sleeved arm acts, quick and mischievous. Reactions must feel spontaneous, no posed faces, no constant smiling, no lip-sync beyond the two short Japanese exclamations, no theatrical overacting beyond genuine prank surprise.

body_mechanics: Every spray is preceded by a visible charge: sip → cheeks puff → head orient toward target → jaw expel in a fan, never spray without the wind-up. The rider's sabotage arm moves first and drives each reaction: at [00:04.8] the outside hand slaps the raised visor DOWN before A can fire, the small impact closing it over her face; at [00:05.4] the hand pressing the helmet makes A's neck collapse backward into a flinch so the mouthful hits the lowered visor; at [00:09.5] the hand twisting the helmet sideways rotates A's head, neck and shoulders to the RIGHT before the water releases onto B; at [00:12.7] the hand blocking B's bottle stops her descending head mid-motion and triggers a choke with shoulders hunching. Drinking follows raise-arm → head-tilt-back → repeated throat swallows → lower. A raises her own visor back up only once, at [00:07.2]. The final arm-link starts from A's shoulder turning in, elbow bending, forearm hooking B. Feet stay grounded; no floating, no dance steps.

costume_hair_physics: Girl A's long straight light-brown hair becomes progressively wetter, separating into heavy strands that cling to face and neck and swing with weighted lag after each head jolt, settling slowly; her dark navy police jacket darkens and clings where soaked, the black knee-high boots gloss with the small puddle at the toe. Girl B's long pale-golden side-parted hair clings damp to her cheeks and lab coat collar; her thin purple-framed oval glasses catch a translucent water film, bead with droplets and slide a fraction down her nose when she is hit, and the white lab coat darkens and sticks where splashed. The helmet visor moves with slight mechanical resistance up and down; the first spray leaves a translucent white film with running droplets on it. Wet hair and fabric never move independently of body motion; the green sleeve enters and exits with natural arm arcs.

negative_constraints: No fourth person, no rider's face or body shown beyond the green-sleeved arm and scooter fairing, no passersby, no added props beyond the clear cup, black tumbler, blue bottle and helmet, no laboratory equipment, no drawn police weapons or extra duty gear brought into frame, no garments added, no face beautification or smoothing, photorealistic real human cosplayers only — no anime, no cartoon, no cel-shaded look, no daytime, no neon or stage lighting, no choreographed dancing, no gimbal-smooth camera, no orbit or zoom other than the freeze-frame punch-ins, no ordinary hard cuts, no subtitles or captions (the two Japanese exclamations are spoken diegetic lines, not on-screen text), no blood, no aerosol mist replacing the water splashes, no oversized cartoon water volume, no visor vanishing, no identity or hairstyle changes, no costume swap, no freeze frames outside the three specified beats.

overall_soundscape: Raw close-range phone field recording over a quiet night-alley bed (distant street hum, faint scooter idle vibration). Key diegetic cues in order: water pouring onto boot and pavement at 1.2s; A's shouted Japanese exclamation at 2.5s; swallowing while charging; the plastic visor slapped down at 4.8s; the first interrupted spray splattering onto the visor at 5.8s; the helmet-grab scuffle and the second spray hitting B's face and glasses at 9.8s; B's short Japanese exclamation at 10.6s; B's rapid chugging gulps at 11.8s and her choked spit at 12.9s; laughter swelling to the end. On EACH of the three freeze frames add a stylized comedic post-SFX: a short record-scratch/rewind squeal leading into a deep impact "boom" hit, then a roughly half-second near-silence vacuum during the frozen hold, normal ambient sound snapping back when motion resumes. Keep everything unpolished, no studio-clean foley.

non_diegetic_music: No non-diegetic music. Explicit silence of the music layer throughout; the soundtrack is diegetic field audio plus the three freeze-frame comedic SFX hits only.


---

## FILE: case-studies/2026-10-advanced-camera-combinations/README.md (https://github.com/LoveRain1997/h3-prompt-journal/blob/HEAD/case-studies/2026-10-advanced-camera-combinations/README.md)

# Advanced Camera Movement Combinations (物理运镜组合库 · 10 组合 · NO DEGREE VERSION)

- **Date:** 2026-10
- **Source:** 用户提供的 H3 ADVANCED CAMERA MOVEMENT COMBINATIONS (PHYSICAL CAMERA GEOMETRY VERSION)
- **Prompt:** see [prompt.md](./prompt.md) — 14 条 SEGMENT RULES + 13 词词库 + 10 组组合全文, 可直接复制进任何 H3 舞蹈 MV 提示词

## The problem

The most common camera language in H3 prompts — `ARC 40°`, `45° ORBIT`, `camera rotates 60°`, `low angle → high angle` — is a **term plus a number**. H3 cannot execute that unit. It produces either nothing, or a wild swinging orbit that nobody asked for.

## The breakthrough — replace the unit of description

> CAMERA TERM → DEGREES  ❌
> START POSITION → PHYSICAL PATH → SUBJECT TRIGGER → END POSITION → FINAL FRAMING  ✅

Wrong:

> `The camera arcs 45 degrees to her right.`

Right:

> `The camera starts directly in front of her, then the operator physically walks toward her right shoulder while keeping the lens aimed at her face; the camera finishes beside the right shoulder, slightly forward of the arm line.`

Every description now contains **where the operator starts, which side of her body they walk along, what the lens keeps aimed at, and where they finish**. H3 executes people walking in rooms; it does not execute protractor readings.

## The 14 segment rules (the constitution)

The rule set encodes the whole physics: real handheld operation in 3D space (PHYSICAL CAMERA LAW), real distance/height/side placement (POSITION LAW), zero degrees (NO DEGREE LANGUAGE), stated optical-axis targets (OPTICAL-AXIS LAW), distance and height only through real travel (DISTANCE/HEIGHT LAW), readable walking paths when crossing sides (SIDE-CHANGE LAW), and the master rule — **SUBJECT-TRIGGER LAW: the camera changes movement only because a written performer action causes the change**.

Plus the choreography contract: connected dance actions never replaced by walking; the dancer relocates with one or two purposeful steps while the camera does the spatial reframing; HERO LOCK when stopped; TRANSITION HOOK at the end.

The 15th rule binds it all together — **TIME-CODE SYNC LAW**: the performer's actions and the camera's movements share one precise absolute time-code, synchronized 1:1. The camera starts when the action starts, peaks when the accent peaks, recovers when the movement recovers. Every onset timestamp is simultaneously the body-peak timestamp and the camera-response timestamp. One timeline, two synchronized motion tracks, one shared set of action peaks.

## The thirteen-word lexicon

PUSH-IN · PULL-BACK · TRACK · PASS · WRAP · RISE · DROP · FLOOR SLIDE · OCCLUSION · REVEAL · WHIP PAN · RECOIL · HERO LOCK

Each word is defined as a **physical path performed by the operator**, not an effect. WRAP is not "orbit" — it is "walk around one side of the performer while continually redirecting the lens". OCCLUSION is not "a transition" — it is "her arm/sleeve/hair passes close to the lens and temporarily blocks the image while the operator keeps moving".

## The ten combinations

| # | Path | Signature move |
|---|---|---|
| 01 | Front follow → side travel → shoulder pass → front reacquire | the all-purpose single-take dance cover |
| 02 | Side follow → overtake → physical turn → reverse follow | walk-forward sections; camera ends ahead of her |
| 03 | Push-in → right-side reposition → three-quarter reframe | chest phrase into a finished 3/4 pose |
| 04 | Upper-body follow → drop → footwork insert → rise → face | the mirror law: 人蹲机蹲, 人起机起 |
| 05 | Low follow → rise with body → chest → face | grounded phrase rising through the frame |
| 06 | Push-in → arm occlusion → hidden side shift → reveal | her arm is the transition tool |
| 07 | Front follow → operator stop → she crosses → physical turn → follow behind | direction changes; camera stays put then re-acquires |
| 08 | Floor slide → body reveal → rise → upper-body follow | footwork from the floorboards up |
| 09 | Close front → pass one shoulder → body blocks lens → opposite-side reveal | intimate three-quarter finish |
| 10 | Hero lock → micro push → performance trigger → side response | the camera answers, never initiates |

## Takeaway

组合 10 是整库的伦理缩影：**相机先站定（HERO LOCK），只做一次微推进，然后完全由人物的强动作触发侧移响应**。所有十个组合共享同一条纪律——摄影师拿着真实摄影机在场景里怎么走、怎么摆、何时因为舞者的什么动作而改变路径；人物与镜头的关系永远写成空间与路径，从不写成术语与角度。Pairs with *System A: Physical Camera Position* (where the camera is) and *System B: Camera Movement Discipline* (how much it may move).


---

## FILE: case-studies/2026-10-advanced-camera-combinations/prompt.md (https://github.com/LoveRain1997/h3-prompt-journal/blob/HEAD/case-studies/2026-10-advanced-camera-combinations/prompt.md)

# H3 Advanced Camera Movement Combinations (物理运镜组合库 · 10 组合 · NO DEGREE VERSION)

> Source: 用户提供的 ADVANCED CAMERA MOVEMENT COMBINATIONS (PHYSICAL CAMERA GEOMETRY VERSION) 全文.
> 彻底取消 degrees/angle 数字体系 — 每个组合 = START POSITION → PHYSICAL PATH → SUBJECT TRIGGER → END POSITION → FINAL FRAMING.
> FILM FORMAT: Vertical 4:3 photorealistic cinematic live-action dance MV, 10-12.5s. Exactly one performer: the woman shown in <Picture 1>. <Picture 1> is the sole authority for identity, face, hairstyle, body proportions, costume, accessories and character appearance. <Picture 1> does not define generated pose, camera placement or choreography. This is a DANCE PERFORMANCE, not walking footage and not a static fashion portrait.

## SEGMENT RULES

* AUTONOMY: every segment below is complete on its own; paste the segment alone with <Picture 1> and <Audio 1>.
* PHYSICAL CAMERA LAW: all camera motion is performed by a real handheld or physically operated camera in real three-dimensional space.
* POSITION LAW: camera placement must be described using real distance, height, side, front/back relationship and travel direction. Never describe a camera move only with abstract cinematography terms.
* NO DEGREE LANGUAGE: do not use degree measurements to define camera position or camera travel. Define the start position and end position physically.
* OPTICAL-AXIS LAW: state what part of the performer the lens is physically aimed at during each major movement.
* DISTANCE LAW: camera distance changes only through actual physical forward or backward travel.
* HEIGHT LAW: camera height changes only through an actual physical rise or descent of the camera.
* SIDE-CHANGE LAW: when moving from one side of the performer to another, the operator must physically travel around the performer's body in a readable path.
* SUBJECT-TRIGGER LAW: the camera changes movement only because a written performer action causes the change.
* TIME-CODE SYNC LAW: the performer's actions and the camera's movements share one precise absolute time-code, synchronized 1:1 — the camera starts when the action starts, peaks when the accent peaks, recovers when the movement recovers; every onset timestamp is simultaneously the body-peak timestamp and the camera-response timestamp.
* UPPER-BODY PRIORITY: face, shoulders, chest, arms and torso remain the primary visual area; lower body is shown only when the choreography requires it.
* DANCE LAW: after the first establishing state, the performer continuously performs connected dance actions. Do not replace choreography with slow walking.
* POSITION CHANGE LAW: the performer uses no more than one or two purposeful steps to relocate within a segment. Camera movement does most of the spatial reframing.
* HERO LOCK: when the camera stops, it remains physically stable while the performer completes the final accent.
* TRANSITION HOOK: the last action must create a clear physical state that can motivate the next shot.

## CAMERA LEXICON

* PUSH-IN = the operator physically walks or glides toward the performer along the existing camera axis.
* PULL-BACK = the operator physically retreats from the performer while preserving the intended framing.
* TRACK = the operator physically moves sideways beside the performer while maintaining roughly the same forward-facing relationship.
* PASS = the camera physically travels beyond the performer's previous position, moving from one side relationship into another.
* WRAP = the operator physically walks around one side of the performer while continually redirecting the lens to keep the performer visible.
* RISE = the real camera physically moves upward through space while maintaining the written subject relationship.
* DROP = the real camera physically descends toward the lower body.
* FLOOR SLIDE = the camera physically travels laterally with the lens 5-15cm above the floor.
* OCCLUSION = a real nearby object, arm, sleeve, hair, shoulder or body edge temporarily blocks the lens.
* REVEAL = the operator continues moving until the performer becomes visible again from the newly established physical camera position.
* WHIP PAN = one fast physical pan caused by a performer action, immediately followed by stabilization.
* RECOIL = the operator physically backs away a short distance in response to a strong performer accent.
* HERO LOCK = the operator stops moving and holds the camera position physically.
* TIME-CODE SYNC = one absolute timeline shared by dance and camera: simultaneous start, synchronized travel, peak-to-peak alignment, synchronized recovery, continuous hand-off, shared onset timestamps.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## COMBINATION 01 — FRONT FOLLOW → SIDE TRAVEL → SHOULDER PASS → FRONT REACQUIRE

SHOT 1 — CLOSE FRONTAL FOLLOW.
Real handheld camera 0.8-1.1m directly in front of the performer, lens 95-110cm above floor, optical axis aimed at the upper chest and face.
The performer ALREADY begins the dance phrase.
0.000-2.500s: she performs connected upper-body choreography while the operator physically retreats backward to preserve the same head-to-waist framing.
The camera remains directly in front of her.

SHOT 2 — SIDE TRAVEL.
2.500-5.000s: the performer shifts her weight toward her right side and sends the right arm outward.
As the arm leaves the torso, the operator physically moves sideways toward the performer's right while continuing to keep the lens pointed toward her upper chest.
The camera travels beside her instead of orbiting around her.

SHOT 3 — SHOULDER PASS.
5.000-7.200s: the performer turns her shoulders while continuing the arm phrase.
The operator physically passes the performer's right shoulder, moving from the front-right side toward the space beside her right shoulder.
The lens continues tracking her face as the camera crosses that position.

SHOT 4 — FRONT REACQUIRE.
7.200-10.500s: the performer turns her torso back toward the lens.
The operator physically continues around the outside of her movement until the camera is again directly in front of her.
Camera height remains approximately 100cm.
10.500-11.500s: HERO LOCK on the finishing upper-body accent.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## COMBINATION 02 — SIDE FOLLOW → CAMERA OVERTAKES → TURN AROUND → REVERSE FOLLOW

SHOT 1 — SIDE FOLLOW.
Real handheld camera 1.0-1.3m to the performer's left side, lens 100-115cm high, optical axis aimed horizontally toward her ribcage and face.
0.000-2.800s: the performer performs two compact lateral dance steps toward camera-right.
The operator physically tracks sideways at matching speed.

SHOT 2 — OVERTAKE.
2.800-5.200s: the performer pushes one shoulder forward and extends the opposite arm.
The operator accelerates along the same travel direction and physically moves ahead of the performer.
The camera ends several steps in front of her rather than remaining beside her.

SHOT 3 — PHYSICAL TURN.
5.200-6.400s: the operator physically turns the camera around to face the performer from the front.
The performer continues the dance phrase without stopping.

SHOT 4 — REVERSE FOLLOW.
6.400-10.800s: the performer advances through one or two purposeful dance steps while the camera physically retreats backward in front of her.
The camera maintains upper-body framing.
10.800-11.500s: the performer hits the final accent and the camera settles.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## COMBINATION 03 — PUSH-IN → RIGHT-SIDE REPOSITION → CLOSE THREE-QUARTER REFRAME

SHOT 1 — PHYSICAL PUSH.
Real handheld camera 1.2-1.5m directly in front, lens 100-115cm high, optical axis aimed at the sternum.
0.000-3.000s: operator physically advances toward the performer while she performs a compact chest-and-arm phrase.
Camera closes the distance by approximately half a meter through real movement.
No digital zoom.

SHOT 2 — SIDE REPOSITION.
3.000-5.500s: the performer sharply turns her right shoulder toward camera while extending the near arm.
The operator stops advancing forward and physically moves toward the open space beside her right shoulder.
The lens continues pointing at her face.

SHOT 3 — THREE-QUARTER REFRAME.
5.500-8.000s: operator continues physically moving around the performer's right side until the camera is positioned beside and slightly forward of that shoulder.
The performer is now seen from the front-right three-quarter relationship without any camera spinning.

SHOT 4 — LOCK.
8.000-11.500s: operator stops.
The performer completes the choreography with one strong arm extension and settles into a finished three-quarter dance pose.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## COMBINATION 04 — UPPER-BODY FOLLOW → PHYSICAL DROP → FOOTWORK → PHYSICAL RISE → FACE

SHOT 1 — UPPER-BODY FOLLOW.
Real handheld camera 0.9-1.2m in front, lens 95-110cm high.
0.000-3.200s: performer dances continuously.
Camera remains around chest-and-face height.

SHOT 2 — PHYSICAL DROP.
3.200-4.300s: performer initiates a decisive weight transfer into one leg.
Only when the weight begins to settle, operator physically lowers the camera from approximately waist-height lens placement toward knee-height.
The optical axis tilts downward naturally with the physical descent.

SHOT 3 — FOOTWORK INSERT.
4.300-5.800s: camera is now approximately 40-60cm from the floor and close enough to show the feet and lower-leg mechanics.
One foot crosses, lands and transfers weight.
The supporting leg stays controlled.
Do not remain on the feet after the action is complete.

SHOT 4 — PHYSICAL RISE.
5.800-7.200s: immediately after the foot lands, the operator physically rises back upward while the performer continues the same dance phrase.
Camera returns to approximately chest height.

SHOT 5 — FACE REACQUIRE.
7.200-10.800s: operator physically corrects forward/back distance and re-establishes the face and upper torso.
10.800-11.500s: final upper-body accent and HERO LOCK.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## COMBINATION 05 — LOW FOLLOW → RISE WITH BODY → CHEST → FACE

SHOT 1 — LOW START.
Real physical camera 50-65cm above floor, 0.8-1.1m in front of the performer, optical axis aimed upward toward the lower torso.
0.000-2.500s: performer begins a grounded dance phrase with bent knees and compact hip movement.
Camera follows from the low position.

SHOT 2 — BODY RISE.
2.500-5.000s: performer extends the knees and lifts the torso.
At the beginning of that upward body action, operator physically raises the camera from the lower position toward the performer's waist and chest.
Camera remains close enough that the body appears to rise through the frame.

SHOT 3 — FACE ACQUISITION.
5.000-7.800s: performer lifts the chin and turns toward camera.
Operator continues physically raising the camera until the lens reaches approximately eye height.
No digital tilt or zoom substitute.

SHOT 4 — HERO.
7.800-11.500s: camera stops at eye level.
Performer finishes the phrase with a controlled arm or shoulder accent and holds the finished pose.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## COMBINATION 06 — PUSH-IN → ARM OCCLUSION → SIDE SHIFT WHILE HIDDEN → REVEAL

SHOT 1 — FRONTAL PUSH.
Real handheld camera 0.9-1.2m directly in front, lens 100-115cm high.
0.000-3.000s: physical push toward the performer while she performs a frontal dance phrase.

SHOT 2 — ARM OCCLUSION.
3.000-4.200s: performer sweeps one arm across the camera's field of view.
The hand, forearm or sleeve passes very close to the lens and temporarily blocks most of the image.
Operator keeps physically moving forward.

SHOT 3 — HIDDEN SIDE SHIFT.
4.200-5.100s: while the lens is blocked, operator physically steps sideways toward the performer's left.
No teleporting camera.
No digital repositioning.

SHOT 4 — REVEAL.
5.100-7.800s: as the arm leaves the lens, camera becomes visible again from the performer's front-left side.
The performer has already continued the dance phrase; no reset or pose substitution.

SHOT 5 — RELEASE.
7.800-11.500s: operator physically retreats slightly while following the performer's open-arm movement.
Final arm direction becomes the transition hook.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## COMBINATION 07 — FRONT FOLLOW → STOP → PERFORMER CROSSES → PHYSICAL TURN → FOLLOW FROM BEHIND

SHOT 1 — DIRECT FOLLOW.
Real handheld camera directly behind the performer, 1.0-1.4m away, lens 100-115cm high, optical axis aimed at shoulders and head.
0.000-2.800s: camera physically advances as the performer dances forward.

SHOT 2 — OPERATOR STOP.
2.800-4.200s: performer performs a sudden half-turn and changes direction.
Operator deliberately stops in place.
The performer now moves across the camera's previous line.

SHOT 3 — CROSSING.
4.200-5.600s: performer passes through the foreground with one decisive step and continues dancing.
Camera remains stationary until the crossing action finishes.

SHOT 4 — PHYSICAL TURN.
5.600-7.000s: operator physically turns around to reacquire her.
The new camera position is behind and slightly to the opposite side of the performer.

SHOT 5 — FOLLOW AGAIN.
7.000-10.800s: operator physically follows behind her again, matching the new direction.
10.800-11.500s: camera closes into the upper-body composition and locks.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## COMBINATION 08 — FLOOR SLIDE → SIDE BODY REVEAL → RISE → UPPER-BODY FOLLOW

SHOT 1 — FLOOR POSITION.
Real physical camera 5-15cm above the floor, approximately 0.8-1.0m from the performer's supporting foot.
Optical axis remains close to horizontal.
0.000-2.200s: performer performs one precise footwork phrase.
Camera physically slides laterally along the floor.

SHOT 2 — BODY REVEAL.
2.200-4.000s: performer transfers weight and turns the lower torso.
Camera continues sliding until the torso and costume naturally enter the frame above the feet.

SHOT 3 — RISE.
4.000-5.800s: performer rises through the torso.
Operator physically lifts the camera from floor level through knee and waist height toward the chest.
The entire movement is one continuous physical rise.

SHOT 4 — UPPER-BODY FOLLOW.
5.800-9.500s: camera reaches approximately 100cm lens height.
Operator resumes handheld follow of the performer's shoulders, arms and face.

SHOT 5 — FINISH.
9.500-11.500s: performer completes one strong upper-body accent.
Camera stops cleanly.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## COMBINATION 09 — CLOSE FRONT → PASS ONE SHOULDER → BODY BLOCKS LENS → OPPOSITE-SIDE REVEAL

SHOT 1 — CLOSE FRONT.
Real handheld camera 55-80cm in front of the performer's upper torso, lens 95-110cm high, optical axis aimed at the face.
0.000-2.600s: controlled close follow.

SHOT 2 — SHOULDER APPROACH.
2.600-4.300s: performer rotates one shoulder toward camera and moves the near arm across the torso.
Operator physically walks toward that shoulder rather than backing away.

SHOT 3 — BODY BLOCK.
4.300-5.300s: shoulder, upper arm and hair move close to the lens and partially block the image.
Operator continues physically passing that shoulder.

SHOT 4 — OPPOSITE-SIDE REVEAL.
5.300-7.300s: camera emerges beside the opposite side of the performer.
The performer remains in continuous dance motion.
The lens immediately reacquires the eyes and upper torso.

SHOT 5 — SETTLE.
7.300-11.200s: operator stops traveling laterally and follows only small dance movements.
Final accent ends in a close three-quarter Hero Lock.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## COMBINATION 10 — HERO LOCK → MICRO PUSH → PERFORMER TRIGGERS CAMERA → SIDE RESPONSE

SHOT 1 — TRUE HERO.
Real camera 1.2-1.5m directly in front, lens 105-120cm high, optical axis horizontal.
0.000-2.500s: performer ALREADY holds a completed Hero position.
Only breathing and extremely small eye movement.

SHOT 2 — MICRO PUSH.
2.500-4.800s: operator physically advances only 10-20cm along the existing camera axis.
Camera height and orientation do not change.
Performer remains almost motionless.

SHOT 3 — PERFORMANCE TRIGGER.
4.800-5.400s: performer suddenly initiates a strong shoulder-and-arm movement.
Camera does not move before the action starts.

SHOT 4 — PHYSICAL SIDE RESPONSE.
5.400-8.500s: operator immediately responds by physically stepping toward the same side as the performer's moving arm.
The lens remains aimed at the face and upper chest.
The camera does not circle the performer.

SHOT 5 — REFRAME AND HOLD.
8.500-11.500s: performer finishes the dance phrase.
Operator makes the final small physical position correction and stops.
Hero composition holds through the final beat.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 这里真正需要记住的改法

以后 H3 中，彻底淘汰：

> `30° ARC` / `45° ORBIT` / `camera rotates 60°` / `low angle → high angle`

全部换成**相对于人物的真实位置**。

错误：

> `The camera arcs 45 degrees to her right.`

正确：

> `The camera starts directly in front of her, then the operator physically walks toward her right shoulder while keeping the lens aimed at her face; the camera finishes beside the right shoulder, slightly forward of the arm line.`

错误：

> `Camera rotates 180 degrees around her.`

正确：

> `The camera begins directly in front of her. The operator physically passes along her right side, moving behind her shoulder line, then turns the camera to face her back as she continues the choreography.`

真正稳定的描述单位：

> **START POSITION → PHYSICAL PATH → SUBJECT TRIGGER → END POSITION → FINAL FRAMING**

而不是：

> **CAMERA TERM → DEGREES**


---

## FILE: case-studies/2026-10-baifeng-mengchen-classical-dance-six-segment/README.md (https://github.com/LoveRain1997/h3-prompt-journal/blob/HEAD/case-studies/2026-10-baifeng-mengchen-classical-dance-six-segment/README.md)

# 025 · H3 古风舞蹈进阶教程：动作链反推、局部特写转场，以及双提示词手动剪辑

![封面](cover.png)

> 很多人用 H3 生成古风舞蹈，最后都会遇到一个问题：人物很美、衣袖也会飘，但动作总是云手、转身、抬手、低头，每段都像同一支舞，镜头只是换了角度。真正的解决办法不是继续堆“优雅、灵动、仙气、唯美”，而是从**动作链、时间轴、摄影机、剪辑结构**四个层面重新设计。本篇以《白凤蒙尘》为例。

## 案例信息（Inputs）

| 项 | 内容 |
|---|---|
| 编号 | 025 |
| 类型 | 真人古风舞蹈音乐 MV（多段独立生成 + 第三段双提示词手动剪辑） |
| 输入 | 每段一张 `<Picture 1>`（人物+完整环境唯一视觉权威）+ 该段裁剪 `<Audio 1>` |
| 画幅 | 9:16 竖屏，真人摄影质感、冷中性低饱和 |
| 全曲 | 79.882 s · 105.25 BPM · 六段（第三段实际双生成、手动剪辑） |
| 完整提示词 | 通用模板 + 六段（SEG03 为完整原文一条）+ 手动剪辑时间点流程见 **[prompt.md](prompt.md)** |

## 音乐时间网格

| 项 | 值 |
|---|---|
| 全曲总长 | 79.882 s |
| BPM | 105.25 |
| 单拍 / 半拍 | 0.5701 s / 0.285 s |
| 单小节 / 乐句 | 2.2803 s / 4.5606 s |

## 六段总览（全曲窗口 / 段内时长 / 主动作机制）

| 段 | 全曲窗口 | 时长 | 主动作机制 |
|---|---|---:|---|
| SEG01 白凤凌霜 | 00:00.000–13.166 | 13.166 | 穿掌背转前 → 背身腰圈扫弧 → 半脚尖平衡 → 遮面露眼 → 碎步后踢 |
| SEG02 暗网坠谷 | 13.166–26.958 | 13.792 | 拧倾出胯 → 圆场碎步 → 低位扫弧 → 沉跪即起（转失败被截） |
| SEG03 仙姿蒙尘 | 26.958–39.413 | 12.455 | 交叠前臂螺旋下沉 → 掌心承重低位跪坐 → 甩发露眼 → 坐姿扭转 → 紧凑半转 → 交叉脚制动 → 半跪起身受阻（实际双生成手动剪辑，见第五节） |
| SEG04 强撑仙姿 | 39.413–52.166 | 12.753 | 横向八字臂线 → 半转甩发 → 半脚尖平衡 → 拧倾出胯（仪态重建） |
| SEG05 被迫校正 | 52.166–65.750 | 13.584 | 跪位撩臂起身 → 沉跪即起 → 关节链反向截断 → 起到一半被打断 |
| SEG06 傲骨未折 | 65.750–79.882 | 14.132 | 呼吸提沉 → 背身双臂渐升 → 立圆胸腰展开 → 空手致意 → 冷静终止 |

---

## 一、第一原则：图 1 锁定视觉，Audio 1 驱动运动

本项目只使用两个输入：

- **Picture 1：** 锁定人物身份、脸部、发型、现有服装、饰品、色彩、光线和原有环境。
- **Audio 1：** 提供原始音乐、节奏和动作时间依据。

不要在提示词中重新描写人物长相、服装和背景，否则相当于给模型第二套设计方案，容易让人物和环境漂移。提示词只负责导演三件事：人物如何运动、镜头如何拍摄、动作如何随音乐变化。

歌词不必写入提示词，保留原始音频即可；人物全程不唱歌、不说话、不做口型。舞蹈用身体表现情绪，而不是让模型照着歌词表演哑剧。

## 二、先建立音乐时间轴，再决定动作密度

《白凤蒙尘》参考参数：总长 79.882 s，BPM 105.25，单拍约 0.5701 s，单小节约 2.2803 s，乐句约 4.5606 s，全曲分六个主要段落、每段不超过 15 s。

这组参数不是要求人物每 0.57 s 机械换姿势，而是建立导演的时间网格：强拍适合身体方向转换、重心落点、动作峰值；弱拍可放头部、手指、手腕的微动作；乐句交界适合镜头揭示、遮挡转场和动作状态变化。

动作必须能跨越节拍自然发展。例如一个低位螺旋可以持续数拍，其中仍可出现脚步换重心、掌心落地、肩胯反向、头发延迟扫过等不同事件。节拍是组织运动的骨架，不是把舞蹈切碎的剪刀。

## 三、真正丰富的古风舞蹈，要从资料库中调用现成动链

《舞蹈动链 H3 库》把动作写成可组合的标准单元，每条链都有进入路径、受力过程、动作峰值、惯性延续、回收顺序和明确出态。例如：

- **C16 云手穿掌·背转前：** 背身双臂上扬 → 云手 → 穿掌翻腕 → 盖步 → 背转前 → 双臂交叉下沉收束。
- **C17 撩袖背身涮腰扫弧：** 撩袖转背 → 腰部沿完整圆弧运动 → 手臂沿低位轨迹划过 → 螺旋拧转回到正面。
- **C24 跪落、跪转、贴地扫弧：** 站姿沉气跪落 → 低位旋转 → 沿地面手臂扫弧 → 坐姿与后仰。
- **C28 侧卧到跪姿再起身：** 收腿 → 推地 → 跪撑 → 展胸 → 单膝蹬地，身体从地面逐层恢复站立。

这些不是必须全用，而是从库中挑选，再按人物、音乐、故事重新组合。反推时先问：这条链从什么姿态进入？哪个部位先启动？重心由哪里转到哪里？四肢如何依次传递动力？最高点在哪？如何利用惯性进入下一个动作？只写“旋转、下腰、展袖”，H3 要自行猜测过程；写出受力、路径与结果，模型才有机会生成更丰富的连续运动。

## 四、为什么我提供的宋玉第三段动作更丰富？

第三段对应全曲原始时间窗 **26.958–39.413 s，时长 12.455 s**，不是普通站立古风舞，而由五种身体运动机制构成：

1. 交叉脚步与前臂交叠，带动重心向下旋入；
2. 掌心承重，进入低位坐姿跪身；
3. 一次甩发完成遮挡与眼神揭示；
4. 坐姿中骨盆与胸廓向相反方向扭转；
5. 地面支撑起身，进入半跪、停滞与未完成的恢复。

重点不只是数量多，而是身体高度、支撑点、旋转轴、运动方向和动作质感都在改变。

更值得注意的是**甩发揭眼**：手经过脸部线条但不必真的碰头发；头部从左向右扫过，头发因惯性延迟跟随，最后露出眼睛。摄影机从肩后移到低位正面，让头发自然成为视觉遮挡。一次动作同时承担舞蹈、表情、镜头转场、叙事四种功能，比单纯插入一个眼睛特写更有效。

## 五、第三段：完整原文提示词奉上，实际用「双生成 + 手动剪辑」完成

提示词本体只有一条——就是那支完整的 **12.455 s 原文提示词**（[prompt.md](prompt.md) 原样奉上、直接复制），不要把它拆成两条。它的导演时间轴在动作上可分为两个章节：

- **前半（全曲 26.958–33.916，6.958 s，Shot 1–3）：低位进入与视觉揭示。** 双脚交叉 → 前臂交叠 → 螺旋下沉 → 掌心承重 → 低位跪坐 → 甩发露眼 → 坐姿侧向扭转；强调从站姿进入地面的完整路径，建立局部特写、头发遮挡与镜头揭示。
- **后半（全曲 33.916–39.413，5.497 s，Shot 4–6）：低位反转与恢复受阻。** 支撑脚与掌心重新承重 → 紧凑半转 → 交叉脚落地 → 脚尖制动 → 再次压低 → 半跪起身，身体试图恢复却在最后停住，留下未完成状态；强调旋转后的惯性、低位动作重组与恢复受阻的反差。

这两章不是两支互不相干的舞，而是同一条动作链的前后两部分。实际制作时，因平台单次生成时长受限、且动作机制复杂，以这条原文为唯一导演依据，分两次生成源片，再在剪辑软件里选动作最匹配的画格衔接；最终只截取需要的动作区间。

**手动剪辑点（精确）：全曲 33.916（SEG03 时间轴 6.958），即原文 Shot 3 结束 / Shot 4 开始。** 前半在坐姿扭转收尾（收手、收腿），后半从撑脚、掌心承重起身进入半转。在两源片该点找身体姿态、运动方向、手部轨迹、摄影机运动的共同相位，优先用手/袖/发丝掠过镜头短暂遮挡接缝；音乐不重启、接缝落在气口。完整五步流程见 [prompt.md](prompt.md)。

手动剪辑的关键不是把两段视频简单拼起来，而是找到两段素材在身体姿态、运动方向、手部轨迹和摄影机运动上的共同点。

## 六、段内转场与段间衔接，必须分开设计

**段内：利用局部特写引导视觉。** 同一段音乐内可用手腕、指尖、脚步、肩背、发丝、眼睛等局部特写：掌心触地时镜头短暂贴近手掌，再顺前臂与肩背受力方向抬起；甩发时镜头跟随发丝扫过，眼睛重新出现时完成揭示；转身时先见肩背，再追随旋转方向回到正面。局部特写不是随意插入的装饰，而是让观众知道接下来该看什么。

**段间：必须保持动作连续。** 段落之间不是每次重摆初始姿势——上一段结尾的手腕方向、脚下重心、身体朝向或眼神落点，应成为下一段的动作起点。把两次生成的源片手动拼接时，优先寻找这样的剪辑点：同一只手沿相同方向继续移动；身体转向保持相同趋势；头发或衣袖掠过镜头短暂遮挡接缝；重心正从一个支撑点转到另一个；摄影机正处于横移、下降或追随中。

段内局部特写负责改变观察位置，段间动作匹配负责维持同一舞者持续运动的感觉，两者不可混为一谈。

## 七、摄影机必须参与动作，而不是事后补拍

希区柯克式压迫变焦（Dolly Zoom）可用于坠落、受制、恢复失败等关键时刻，但不应每段都用。其他镜头按动作选择：

- **脚步与落地：** 保持全身构图，让重心变化可读。
- **贴地动作：** 摄影机随身体下降，保留支撑手与伸展腿的关系。
- **甩发揭眼：** 用头发自然遮挡，再落到眼睛。
- **坐姿扭转：** 沿骨盆与胸廓相反方向横移，强化扭转感。
- **旋转急停：** 先随身体转动，再稍迟到并重新找到人物，制造真实惯性。
- **受限起身：** 摄影机随身体升高，但在动作停滞时停止，让“起不来”被看见。

每个镜头都应回答：摄影机在哪里？为什么此刻运动？最后要揭示什么？只写“镜头环绕人物”，即使模型完成运镜，也未必让舞蹈更丰富。

## 八、最后的制作检查清单

提交 H3 提示词前，检查：

- [ ] 人物与环境是否完全由图 1 锁定？
- [ ] 音频是否保持原始状态、没有把歌词写进提示词？
- [ ] 是否明确禁止人物唱歌、说话、对口型？
- [ ] 每段是否拥有真正不同的脚步、重心和身体运动机制？
- [ ] 是否写清动作如何启动、发展、到达峰值、回收？
- [ ] 特写是否引导观众看到关键动作，而不是重复推脸？
- [ ] 每段结尾是否指定可供下一段接续的动作状态？
- [ ] 两次生成的源片是否通过动作匹配完成手动剪辑（剪辑点全曲 33.916）？
- [ ] 旋转、下沉、起身是否具有可信的支撑和惯性？
- [ ] 是否避免六段重复同一种云手、转身、抬手、定格组合？

## 小结

H3 古风舞蹈的质量，不取决于用了多少古风形容词，也不取决于加入多少摄影术语。真正有效的方法，是从成熟动作链中选择素材，按音乐时间轴重新编舞，让每个动作拥有清晰动力来源；再利用特写、遮挡、视差与剪辑，把多个生成片段组织成一条连续的舞蹈。

- **动作链**解决“怎么跳”；
- **时间轴**解决“何时跳”；
- **镜头**解决“观众看见什么”；
- **手动剪辑**解决“不同生成结果如何成为同一支舞”。

这四件事配合，才是可重复使用的 H3 古风 MV 制作流程。

## 完整提示词

- 通用可套用模板（self-contained 骨架）
- 六段完整提示词（人物称呼统一为 **Song Yu / 宋玉**；第三段为完整 **12.455 s 原文一条**，不拆分）
- 第三段双生成手动剪辑时间点流程（五步）
- 段间匹配剪辑逐边界表

**→ [prompt.md](prompt.md)**

---

← 返回 [H3 Prompt Journal 索引](../../README.md)


---

# ISSUES / PRs of LoveRain1997/h3-prompt-journal
