# Motion data

Only this note is kept in git. Everything else here is ignored: it's large, licensed separately,
and some of it may not be shared.

| Dataset | Used for | Here | Source | Licence |
|---|---|---|---|---|
| **LAFAN1** (Ubisoft La Forge) | M1 training: 77 takes by 5 performers, about 4.5 hours at 30 fps (walking, running, sprinting, falling and getting up, obstacles, ground work, dancing, fighting, aiming, pushes) | `lafan1/lafan1.zip`, `lafan1/license.txt` (downloaded 2026-10-23) | [GitHub, ubisoft-laforge-animation-dataset](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) (`lafan1/lafan1.zip`, via Git LFS) | CC BY-NC-ND 4.0: private training is fine; **retargeted copies must never be shared** |
| **100STYLE** (Mason et al. 2022) | M1 training: 810 takes by one performer, 22.1 hours at 60 fps, 100 styles × walking and running (forwards, backwards, sideways), transitions and idling | `100style/100STYLE.zip` (downloaded 2026-10-23) | [Zenodo record 8127870](https://zenodo.org/records/8127870) | CC BY 4.0 (on the record) |
| **CMU Motion Capture Database** | M1 training: 2,514 motions by 112 performers (2,503 converted, 10.8 hours) | `cmu/allasfamc.zip` (ASF/AMC, downloaded 2026-10-23) | [mocap.cs.cmu.edu](http://mocap.cs.cmu.edu/) | "Free for all uses"; acknowledgement requested |
| **BABEL** | Action labels for CMU takes (1,902 of them), frame segments in seconds; for intents later | `babel/babel_v1-0_release.zip` (downloaded by the owner, 2026-10-23) | [babel.is.tue.mpg.de](https://babel.is.tue.mpg.de/) | Owner registered and accepted it |
| **Bandai Namco Research Motion Dataset** | Planned: 3,077 takes, about 3.9 hours at 30 fps, conversational gestures and locomotion in emotional styles. Needs its own calibration (no T-pose anywhere) | `bandai_namco/bandai-namco-motiondataset.zip` (downloaded 2026-10-23) | [GitHub](https://github.com/BandaiNamcoResearchInc/Bandai-Namco-Research-Motiondataset) | CC BY-NC-ND 4.0: private training only, never share copies |
| **AMASS** | Not used: its server was too slow to download from. CMU's part comes from CMU directly; the converter (`datasets/amass.py`) stays ready | only AMASS's official sample, `amass/sample/amass_sample.npz` | [amass.is.tue.mpg.de](https://amass.is.tue.mpg.de/) | Needs registration |

This folder holds the **raw downloads** only, shared by every model version. Each version
converts them with its own scripts into its own folder: `versions/vN/data/processed/<dataset>/`
(each take retargeted onto the canonical skeleton at 60 Hz, with foot contacts and labels). Those
derived copies are git-ignored too, and never shared.
