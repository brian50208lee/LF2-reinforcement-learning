# LF2-reinforcement-learning
2017 ADLxMLDS final project - AliceChienChien

## Depedency
```
https://github.com/Project-F/F.LF.git
https://github.com/Project-F/LF2_19.git
```
Will be clone when run train/test.sh

## Environment
Python
```
tensorflow 1.3.0
numpy 1.13.3
Flask 0.12.2 (http request)
Flask-Cors 3.0.3 (http request)

pip install -r requirment.txt
```

Javascript
```jQuery```

OS
```MacOS, Linux```

Browser
```Firefox (recommend), Chrome```

## Train
```
bash train.sh
```

## Test Demo
```
bash test_demo.sh
or
bash test_demo.sh [pretrain_model_path]
```
1P: Bandit (DeepFighter)

2P: RandomChar (dumbass AI)

## Test PK
```
bash test_pk.sh
or
bash test_pk.sh [pretrain_model_path]
```
Change computer setting in "Character Selection" page:

Player:  DeepFighter

Fighter: Bandit

## Agent Actions
do_nothing, left, right, up, down, def, jump, att,
hold_left, hold_right, hold_up, hold_down

## Keyboard Control
up, down, left, right, z(def), a(jump), q(att)