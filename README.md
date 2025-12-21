# Project Overview

Today's understanding of human reasoning generally holds that individuals are equipped with various cognitive heuristics and mental concepts they can draw upon for different tasks. For example we can only store a limited number of unique objects in working memory[^1]. Because of this biological and physical limitation, there are multiple heuristics for calculating the size of larger groups of objects. A good way of modelling this process is by using probabilistic program induction[^2] where the model represents concepts as simple programs that best explain observed examples under a Bayesian criterion which are built up compositionally to create more complex programs capable of computing more complex tasks.
However this naturally raises the question of how we subconsciously determine when to use which heuristic. Furthermore it is possible for us to learn new concepts that we can often generalize successfully from just a few examples. Strategy selection as rational metareasoning is one way to model this behaviour[^3]. According to this model people learn to efficiently choose the heuristic with
the best cost–benefit tradeoff by learning a predictive model of each heuristic’s performance. This project aims to combine these two principals to create a python package that enables researchers to flexibly design and run models to solve a variety of reasoning tasks that make use of both or either paradigms.

## Citations

[^1]: Miller, G. A. (1956). The magical number seven, plus or minus two: Some limits on our capacity for processing information. Psychological review, 63(2), 81.

[^2]: Lake, B. M., Salakhutdinov, R., & Tenenbaum, J. B. (2015). Human-level concept learning through probabilistic program induction. Science, 350(6266), 1332-1338

[^3]: Lieder, F., & Griffiths, T. L. (2017). Strategy selection as rational metareasoning. Psychological review, 124(6), 762.