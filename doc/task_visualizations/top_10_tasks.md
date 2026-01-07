# Top 10 Successful Episodes for Video Recording
## Dataset: rerange+spatial_matched_subtasks.json.gz

| Rank | Episode ID | Index | Success Rate | Task Summary |
|------|------------|-------|--------------|--------------|
| 1 | 1312 | 64 | 8/11 | plate, bowl: kitchen counter → dining table |
| 2 | 1317 | 65 | 8/11 | picture frame, vase: living room table → dining room table |
| 3 | 1378 | 67 | 8/11 | laptop stand, monitor stand: bedroom table → living room shelves |
| 4 | 1327 | 66 | 7/11 | laptop, laptop stand: living room table → bedroom chair |
| 5 | 1425 | 69 | 7/11 | toy airplane, helmet: living room table → bedroom shelves |
| 6 | 212 | 15 | 6/11 | tomato, sushi mat: kitchen counter → dining room table |
| 7 | 214 | 17 | 6/11 | sushi mat, spatula, bowl, tomato: kitchen → living room table |
| 8 | 866 | 38 | 6/11 | plate, bowl: dining table → kitchen counter |
| 9 | 1471 | 71 | 5/11 | vase, candle: living room shelves → dining table |
| 10 | 153 | 3 | 5/11 | toy bee, toy construction set: living room table → closet shelves |

---

## Detailed Task Descriptions

### 1. Episode 1312 (Index: 64)
- **Success Rate:** 8/11 runs
- **Scene ID:** 102816009
- **Instruction:** Move the plate and bowl from the kitchen counter to the dining table. Place them next to each other on the table.
- **Objects:** plate, bowl
- **Source Location:** kitchen counter
- **Target Location:** dining table
- **Constraint Type:** Spatial (next to each other)

### 2. Episode 1317 (Index: 65)
- **Success Rate:** 8/11 runs
- **Scene ID:** 102816009
- **Instruction:** Move the picture frame and vase from the living room table to the other dining room table. Place them next to each other on the table.
- **Objects:** picture frame, vase
- **Source Location:** living room table
- **Target Location:** dining room table
- **Constraint Type:** Spatial (next to each other)

### 3. Episode 1378 (Index: 67)
- **Success Rate:** 8/11 runs
- **Scene ID:** 104348082_171512994
- **Instruction:** Move the laptop stand and monitor stand from the bedroom table to the living room shelves. Place them next to each other on the shelves.
- **Objects:** laptop stand, monitor stand
- **Source Location:** bedroom table
- **Target Location:** living room shelves
- **Constraint Type:** Spatial (next to each other)

### 4. Episode 1327 (Index: 66)
- **Success Rate:** 7/11 runs
- **Scene ID:** 102816216
- **Instruction:** Move the laptop and laptop stand from the living room table to the bedroom chair. Place them next to each other on the chair.
- **Objects:** laptop, laptop stand
- **Source Location:** living room table
- **Target Location:** bedroom chair
- **Constraint Type:** Spatial (next to each other)

### 5. Episode 1425 (Index: 69)
- **Success Rate:** 7/11 runs
- **Scene ID:** 104862660_172226844
- **Instruction:** Move the toy airplane and helmet from the table in the living room to the shelves in the bedroom. Place them next to each other on the shelves.
- **Objects:** toy airplane, helmet
- **Source Location:** living room table
- **Target Location:** bedroom shelves
- **Constraint Type:** Spatial (next to each other)

### 6. Episode 212 (Index: 15)
- **Success Rate:** 6/11 runs
- **Scene ID:** 108736824_177263559
- **Instruction:** Move the tomato and then the sushi mat from the counter in the kitchen to the table in the dining room and place them next to each other.
- **Objects:** tomato, sushi mat
- **Source Location:** kitchen counter
- **Target Location:** dining room table
- **Constraint Type:** Temporal + Spatial

### 7. Episode 214 (Index: 17)
- **Success Rate:** 6/11 runs
- **Scene ID:** 108736824_177263559
- **Instruction:** Move the sushi mat, spatula, bowl, and tomato from the kitchen to the table in the living room. Place them next to each other on the table.
- **Objects:** sushi mat, spatula, bowl, tomato
- **Source Location:** kitchen
- **Target Location:** living room table
- **Constraint Type:** Spatial (4 objects next to each other)

### 8. Episode 866 (Index: 38)
- **Success Rate:** 6/11 runs
- **Scene ID:** 102344250
- **Instruction:** Help me move the plate and bowl from the dining table to the kitchen counter. Place them next to each other.
- **Objects:** plate, bowl
- **Source Location:** dining table
- **Target Location:** kitchen counter
- **Constraint Type:** Spatial (next to each other)

### 9. Episode 1471 (Index: 71)
- **Success Rate:** 5/11 runs
- **Scene ID:** 106366173_174226431
- **Instruction:** Move the vase and candle from the shelves in the living room to the dining table. Place them next to each other on the table.
- **Objects:** vase, candle
- **Source Location:** living room shelves
- **Target Location:** dining table
- **Constraint Type:** Spatial (next to each other)

### 10. Episode 153 (Index: 3)
- **Success Rate:** 5/11 runs
- **Scene ID:** 102344049
- **Instruction:** Move the toy bee and toy construction set from the living room table to the closet shelves. Place them next to each other on the shelves.
- **Objects:** toy bee, toy construction set
- **Source Location:** living room table
- **Target Location:** closet shelves
- **Constraint Type:** Spatial (next to each other)

---

## Video Recording Command

```bash
# Run video recording for these episodes
export EPISODE_INDICES="[64,65,66,67,69,15,17,38,71,3]"
sbatch all_scripts/slurm_files/run_ours_rs_mem_r_video.slurm
```

## Task Visualization

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     REARRANGE + SPATIAL TASKS                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Episode 1312: plate, bowl                                                  │
│  ┌──────────────┐                      ┌──────────────┐                     │
│  │   KITCHEN    │  ───────────────►    │ DINING ROOM  │                     │
│  │   counter    │      move            │    table     │                     │
│  │  🍽️  🥣      │                      │   🍽️ 🥣     │                     │
│  └──────────────┘                      └──────────────┘                     │
│                                           next to                           │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Episode 1317: picture frame, vase                                          │
│  ┌──────────────┐                      ┌──────────────┐                     │
│  │ LIVING ROOM  │  ───────────────►    │ DINING ROOM  │                     │
│  │    table     │      move            │    table     │                     │
│  │  🖼️  🏺      │                      │   🖼️ 🏺     │                     │
│  └──────────────┘                      └──────────────┘                     │
│                                           next to                           │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Episode 1327: laptop, laptop stand                                         │
│  ┌──────────────┐                      ┌──────────────┐                     │
│  │ LIVING ROOM  │  ───────────────►    │   BEDROOM    │                     │
│  │    table     │      move            │    chair     │                     │
│  │  💻  📱      │                      │   💻 📱     │                     │
│  └──────────────┘                      └──────────────┘                     │
│                                           next to                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
