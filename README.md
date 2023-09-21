# martFL
The official code of ACM CCS 2023 paper "martFL: Enabling Utility-Driven Data Marketplace with a Robust and Verifiable Federated Learning Architecture". [Arxiv](https://arxiv.org/abs/2309.01098)

![martFL](figure/martFL.png)

## Overview

We implemented our martFL framework with Pytorch, Zokrates, and Solidity. We've outlined the organization of this repository:

- `src`: It contains the Python code for martFL.
- `datasets`: It contains the necessary dataset files.
- `circuit`: It contains the circuit of the verifiable aggregation, which is implemented using [ZoKrates](https://zokrates.github.io/). 
- `smart_contract`: It contains the trading smart contract, which is implemented using [Solidity](https://soliditylang.org/).
- `SEAL-Python`: The SEAL library for homomorphic encryption. Repository link: https://github.com/Huelse/SEAL-Python.git
- `quantized_data`: The quantized data that was generated during the model training process.
- `save_result & models`: It contains the training results and models, respectively.

## Dependencies

The script has been tested running under Python 3.7.13. The dependencies are summarised in `requirements.txt`.

## Reference
If you take advantage of martFL in your research, please cite the following in your manuscript:

```
@article{li2023martfl,
  title={martFL: Enabling Utility-Driven Data Marketplace with a Robust and Verifiable Federated Learning Architecture},
  author={Li, Qi and Liu, Zhuotao and Li, Qi and Xu, Ke},
  journal={arXiv preprint arXiv:2309.01098},
  year={2023}
}
```
