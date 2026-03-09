## A research of the network dynamics mechanisms of seizure propagation and syncronization

### Function
#### methods
- model
  - HR
  - 6D-Epileptor
  - 2D-Epileptor
- analysis
  - fft
  - Butterworth filter
  - phase
  - EI index
  - sync and energy index

#### network
- Random
- Small world
- Scale-free
- Allen mouse brain

### Run
```base
python main.py --task generation --exp base
python main.py --task analysis --exp base
python main.py --task visualization  --exp base
```
  
```number
python main.py --task generation --exp number
python main.py --task analysis --exp number
python main.py --task visualization  --exp number
```

```scalefree_dist_number
python main.py --task generation --exp scalefree_dist_number 
python main.py --task analysis --exp scalefree_dist_number 
python main.py --task visualization  --exp scalefree_dist_number 
```

```random_index
python main.py --task generation --exp randomIdx
python main.py --task analysis --exp randomIdx
python main.py --task visualization  --exp randomIdx
```

```coupling_strength
python main.py --task generation --exp coupling_strength
python main.py --task analysis --exp coupling_strength
python main.py --task visualization  --exp coupling_strength
```

```random_num
python main.py --task generation --exp random_num
python main.py --task analysis --exp random_num
python main.py --task visualization  --exp random_num
```

```scalefree_dist
python main.py --task generation --exp scalefree_dist
python main.py --task analysis --exp scalefree_dist
python main.py --task visualization  --exp scalefree_dist
```

```mouse_chimera
python main.py --task generation --exp mouse_chimera
python main.py --task analysis --exp mouse_chimera
python main.py --task visualization  --exp mouse_chimera
```

```mouse_connect
python main.py --task generation --exp mouse_connect
python main.py --task analysis --exp mouse_connect
python main.py --task visualization  --exp mouse_connect
```

```mouse_control
python main.py --task generation --exp mouse_control
python main.py --task analysis --exp mouse_control
python main.py --task visualization  --exp mouse_control
```