# Performative Prediction

This repository contains the code for reproducing the experiments in

* J. C. Perdomo, T. Zrnic, C. Mendler-Dünner, M.Hardt. [Performative Prediction](https://proceedings.icml.cc/static/paper_files/icml/2020/325-Paper.pdf). ICML 2020.
* C. Mendler-Dünner, J. C. Perdomo, T. Zrnic, M. Hardt. [Stochastic Optimization for Performative Prediction](https://arxiv.org/abs/2006.06887). NeurIPS 2020.

## Background

The core theme in performative prediction is that the choice of a predictive model influences the distribution of future data, typically through actions taken based on the model's predictions. 
Performativity arises naturally in consequential statistical decision-making problems in
domains ranging from financial markets to online advertising.
Traffic predictions influence traffic patterns, crime location prediction influences police allocations that may deter crime,
recommendations shape preferences and thus consumption, stock price prediction determines
trading activity and hence prices.

Learning in this non-stationary setting induces new considerations, challenges and leads to new solution concepts.
In the scope of this project we develop a risk minimization framework for performative prediction and formally study the dynamics of different retraining strategies within this framework.

## Organization of the Repo

The python notebooks for reproducing the simulation experiments of the two papers can be found in the respective folders. 
