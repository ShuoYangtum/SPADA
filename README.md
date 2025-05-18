Doubling Your Data in Minutes: Ultra-fast Tabular Data Generation via LLM-Induced Dependency Graphs
====

Data sets
----
All of the datasets we used are open-soursed.<br>
Adult Income dataset: [https://www.kaggle.com/datasets/wenruliu/adult-income-dataset](https://www.kaggle.com/datasets/wenruliu/adult-income-dataset)<br>
HELOC dataset: [https://www.kaggle.com/datasets/averkiyoliabev/home-equity-line-of-creditheloc](https://www.kaggle.com/datasets/averkiyoliabev/home-equity-line-of-creditheloc)<br>
Iris dataset: [https://archive.ics.uci.edu/dataset/53/iris](https://archive.ics.uci.edu/dataset/53/iris)<br>
California Housing dataset: [https://www.kaggle.com/datasets/camnugent/california-housing-prices](https://www.kaggle.com/datasets/camnugent/california-housing-prices)<br>
The CDC dataset: [https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)<br>
The Mushroom dataset: [https://archive.ics.uci.edu/dataset/848/secondary+mushroom+dataset](https://archive.ics.uci.edu/dataset/848/secondary+mushroom+dataset)<br>


Setup
----
To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Quickstart

```python
from SPADA import SPADA

# Using a LLM to label out the dependencies in your dataset.
response_mushroom='''
cap-diameter: []
stem-height: []
stem-width: []
class: [cap-shape->class, cap-surface->class, cap-color->class, does-bruise-or-bleed->class, gill-attachment->class, gill-spacing->class, gill-color->class, stem-root->class, stem-surface->class, stem-color->class, veil-type->class, veil-color->class, has-ring->class, ring-type->class, spore-print-color->class, habitat->class, season->class]
cap-shape: []
cap-surface: []
cap-color: []
does-bruise-or-bleed: []
gill-attachment: []
gill-spacing: []
gill-color: []
stem-root: []
stem-surface: []
stem-color: []
veil-type: []
veil-color: []
has-ring: []
ring-type: []
spore-print-color: []
habitat: []
season: []
'''

spada=SPADA()
spada.fit(data='mushroom/train.csv',  # your dataset
          description="A Dataset of simulated mushrooms for binary classification into edible and poisonous.", 
         response=response_mushroom)

spada.sample(100) # generate 100 new samples.

```

Demonstration
----
To run our program and check the output in each step, you can simply execute the `demonstration.ipynb` file located in the root directory.<br> 
