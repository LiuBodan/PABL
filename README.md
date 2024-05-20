# MNIST Addition Under Contradictory Information

This example shows a simple implementation of [MNIST Addition](https://arxiv.org/abs/1805.10872) task, where pairs of MNIST handwritten images and their sums are given, alongwith a domain knowledge base containing information on rules of addition is given as well as artificial inconsistent information. The task is to recognize the digits of handwritten images and accurately determine their sum under the existence of inconsistency in the knowledge base.


## About the Codes

1. `kb` contains the knowledge base code for ABL as well as the logic programming code for addition rules (`add.pl` for classical logic based knowledge base, `para_refine.pl` for the paraconsistent logic QMPT0 based knowledge base).
2. `training_logs` contrains training logs and the weights for both versions of ABL on the task.
   
## Run
Install [SWI-Prolog](https://www.swi-prolog.org/Download.html), please make sure the version of SWI-Prolog is below 9.2 as `pyswip` does not support version 9.2+, `swipl` should work in your command line shell.

```bash
pip install -r requirements.txt
python main.py
```

## Environment

Details on the specifications are listed in the table below.

<table class="tg" style="margin-left: auto; margin-right: auto;">
<thead>
<tr>
    <th>CPU</th>
    <th>OS</th>
</tr>
</thead>
<tbody>
<tr>
<td>Intel i5-8400 CPU @ 2.80GHz and 16 GB memory</td>
    <td>Windows 11 Pro</td>
</tr>
</tbody>
</table>

## References
For more information about ABL, please refer to: [ABLKit](https://ablkit.readthedocs.io/en/latest/index.html), [Zhou, 2019](http://scis.scichina.com/en/2019/076101.pdf) and [Zhou and Huang, 2022](https://www.lamda.nju.edu.cn/publication/chap_ABL.pdf).

For more information about paraconsistent logic programming QMPT0, please refer to [Goto, 2018](https://da.lib.kobe-u.ac.jp/da/kernel/D1007206/D1007206.pdf) and [Coniglio and Marcelo, 2016](http://logica.dmi.unisa.it/sysmics/sysmics16/slides/Oliveira_SYSMICS2016.pdf).

For the other compared methods, their codes can be found below:
[DeepProlog](https://github.com/ML-KULeuven/deepproblog/tree/master/src/deepproblog/examples/MNIST),
[LTN](https://github.com/logictensornetworks/logictensornetworks/tree/master/examples/mnist),
[DeepStochLog](https://github.com/ML-KULeuven/deepstochlog/tree/main/examples/addition_simple),
[NeurASP](https://github.com/azreasoners/NeurASP/tree/master/examples/mnistAdd).
