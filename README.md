# LF2-reinforcement-learning
2017 ADLxMLDS final project - AliceChienChien

## Environment
```
tensorflow 1.3.0
numpy 1.13.3
Flask 0.12.2 (http request)
Flask-Cors 3.0.3 (http request)
pip install -r requirment.txt
```

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

## Test PK
```
bash test_pk.sh
or
bash test_pk.sh [pretrain_model_path]
```

## Agent Actions
do_nothing, left, right, up, down, def, jump, att
hold_left, hold_right, hold_up, hold_down

## Keyboard Control
up, down, left, right, z(def), a(jump), q(att)