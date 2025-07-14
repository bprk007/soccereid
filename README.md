
# SoccerReID

SoccerReid is useful for re-identifying the players in different frames of a broadcast video of a soccer game.
It is built using [TorchReid](https://github.com/KaiyangZhou/deep-person-reid.git)
to extract deep feature embeddings, whereas the pre-trained TorchReid model is provided by [SportsReID](https://github.com/shallowlearn/sportsreid)


---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/bprk007/soccereid.git
cd soccereid 
```
### 2. Create a python environment
```
conda create -n soccereid python=3.11
conda activate soccereid
```
### 3. Install dependencies
```
pip install -r requirements.txt

```
### 4. Install torchreid using instructions at [TorchReid](https://github.com/KaiyangZhou/deep-person-reid.git) but do NOT reinstall torch or torchvision as any other version than requirements.txt may result in errors.
### 5. Update model paths in config accordingly and run using:
```
python main.py
```
