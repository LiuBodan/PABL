# Modification of ABL to inconsistent knowledge base from LAMDA available here:
# https://github.com/AbductiveLearning/ABLkit/tree/main
# 
# This version of the script will use the paraconsistent knowledge base `para_refine.pl`. 

"""
Copyright (c) 2024 LAMDA.  All rights reserved.
"""

import inspect
import logging
import os
from abc import ABC, abstractmethod
from itertools import combinations, product
from typing import Any, Callable, List, Optional
import numpy as np
from utils.cache import abl_cache
from utils.logger import print_log
from utils.utils import flatten, reform_list, to_hashable


class KBBase(ABC):
    """
    Base class for knowledge base.

    Parameters
    ----------
    pseudo_label_list : List[Any]
        List of possible pseudo-labels. It's recommended to arrange the pseudo-labels in this
        list so that each aligns with its corresponding index in the base model: the first with
        the 0th index, the second with the 1st, and so forth.
    max_err : float, optional
        The upper tolerance limit when comparing the similarity between the reasoning result of
        pseudo-labels and the ground truth. This is only applicable when the reasoning
        result is of a numerical type. This is particularly relevant for regression problems where
        exact matches might not be feasible. Defaults to 1e-10.
    use_cache : bool, optional
        Whether to use abl_cache for previously abduced candidates to speed up subsequent
        operations. Defaults to True.
    key_func : Callable, optional
        A function employed for hashing in abl_cache. This is only operational when use_cache
        is set to True. Defaults to ``to_hashable``.
    cache_size: int, optional
        The cache size in abl_cache. This is only operational when use_cache is set to
        True. Defaults to 4096.

    Notes
    -----
    Users should derive from this base class to build their own knowledge base. For the
    user-build KB (a derived subclass), it's only required for the user to provide the
    ``pseudo_label_list`` and override the ``logic_forward`` function (specifying how to
    perform logical reasoning). After that, other operations (e.g. how to perform abductive
    reasoning) will be automatically set up.
    """

    def __init__(
        self,
        pseudo_label_list: List[Any],
        max_err: float = 1e-10,
        use_cache: bool = True,
        key_func: Callable = to_hashable,
        cache_size: int = 4096,
    ):
        if not isinstance(pseudo_label_list, list):
            raise TypeError(f"pseudo_label_list should be list, got {type(pseudo_label_list)}")
        self.pseudo_label_list = pseudo_label_list
        self.max_err = max_err

        self.use_cache = use_cache
        self.key_func = key_func
        self.cache_size = cache_size

        argspec = inspect.getfullargspec(self.logic_forward)
        self._num_args = len(argspec.args) - 1
        if (
            self._num_args == 2 and self.use_cache
        ):  # If the logic_forward function has 2 arguments, then disable cache
            self.use_cache = False
            print_log(
                "The logic_forward function has 2 arguments, so the cache is disabled. ",
                logger="current",
                level=logging.WARNING,
            )

    @abstractmethod
    def logic_forward(self, pseudo_label: List[Any], x: Optional[List[Any]] = None) -> Any:
        """
        How to perform (deductive) logical reasoning, i.e. matching an example's
        pseudo-labels to its reasoning result. Users are required to provide this.

        Parameters
        ----------
        pseudo_label : List[Any]
            Pseudo-labels of an example.
        x : List[Any], optional
            The example. If deductive logical reasoning does not require any
            information from the example, the overridden function provided by the user can omit
            this parameter.

        Returns
        -------
        Any
            The reasoning result.
        """

    def abduce_candidates(
        self,
        pseudo_label: List[Any],
        y: Any,
        x: List[Any],
        max_revision_num: int,
        require_more_revision: int,
    ) -> List[List[Any]]:
        """
        Perform abductive reasoning to get a candidate compatible with the knowledge base.

        Parameters
        ----------
        pseudo_label : List[Any]
            Pseudo-labels of an example (to be revised by abductive reasoning).
        y : Any
            Ground truth of the reasoning result for the example.
        x : List[Any]
            The example. If the information from the example
            is not required in the reasoning process, then this parameter will not have
            any effect.
        max_revision_num : int
            The upper limit on the number of revised labels for each example.
        require_more_revision : int
            Specifies additional number of revisions permitted beyond the minimum required.

        Returns
        -------
        Tuple[List[List[Any]], List[Any]]
            A tuple of two elements. The first element is a list of candidate revisions,
            i.e. revised pseudo-labels of the example. that are compatible with the knowledge
            base. The second element is a list of reasoning results corresponding to each
            candidate, i.e., the outcome of the ``logic_forward`` function.
        """
        return self._abduce_by_search(pseudo_label, y, x, max_revision_num, require_more_revision)

    def _check_equal(self, reasoning_result: Any, y: Any) -> bool:
        """
        Check whether the reasoning result of a pseduo label example is equal to the ground truth
        (or, within the maximum error allowed for numerical results).

        Returns
        -------
        bool
            The result of the check.
        """
        if reasoning_result is None:
            return False

        if isinstance(reasoning_result, (int, float)) and isinstance(y, (int, float)):
            return abs(reasoning_result - y) <= self.max_err
        else:
            return reasoning_result == y

    def revise_at_idx(
        self,
        pseudo_label: List[Any],
        y: Any,
        x: List[Any],
        revision_idx: List[int],
    ) -> List[List[Any]]:
        """
        Revise the pseudo-labels at specified index positions.

        Parameters
        ----------
        pseudo_label : List[Any]
            Pseudo-labels of an example (to be revised).
        y : Any
            Ground truth of the reasoning result for the example.
        x : List[Any]
            The example. If the information from the example
            is not required in the reasoning process, then this parameter will not have
            any effect.
        revision_idx : List[int]
            A list specifying indices of where revisions should be made to the pseudo-labels.

        Returns
        -------
        Tuple[List[List[Any]], List[Any]]
            A tuple of two elements. The first element is a list of candidate revisions,
            i.e. revised pseudo-labels of the example. that are compatible with the knowledge
            base. The second element is a list of reasoning results corresponding to each
            candidate, i.e., the outcome of the ``logic_forward`` function.
        """
        candidates, reasoning_results = [], []
        abduce_c = product(self.pseudo_label_list, repeat=len(revision_idx))
        for c in abduce_c:
            candidate = pseudo_label.copy()
            for i, idx in enumerate(revision_idx):
                candidate[idx] = c[i]
            reasoning_result = self.logic_forward(candidate, *(x,) if self._num_args == 2 else ())
            if self._check_equal(reasoning_result, y):
                candidates.append(candidate)
                reasoning_results.append(reasoning_result)
        return candidates, reasoning_results

    def _revision(
        self,
        revision_num: int,
        pseudo_label: List[Any],
        y: Any,
        x: List[Any],
    ) -> List[List[Any]]:
        """
        For a specified number of labels in an example's pseudo-labels to revise, iterate through
        all possible indices to find any candidates that are compatible with the knowledge base.
        """
        new_candidates, new_reasoning_results = [], []
        revision_idx_list = combinations(range(len(pseudo_label)), revision_num)
        for revision_idx in revision_idx_list:
            #print(pseudo_label, y, x, revision_idx)
            candidates, reasoning_results = self.revise_at_idx(pseudo_label, y, x, revision_idx)
            new_candidates.extend(candidates)
            new_reasoning_results.extend(reasoning_results)
        return new_candidates, new_reasoning_results

    @abl_cache()
    def _abduce_by_search(
        self,
        pseudo_label: List[Any],
        y: Any,
        x: List[Any],
        max_revision_num: int,
        require_more_revision: int,
    ) -> List[List[Any]]:
        """
        Perform abductive reasoning by exhaustive search. Specifically, begin with 0 and
        continuously increase the number of labels to revise, until
        candidates that are compatible with the knowledge base are found.

        Parameters
        ----------
        pseudo_label : List[Any]
            Pseudo-labels of an example (to be revised).
        y : Any
            Ground truth of the reasoning result for the example.
        x : List[Any]
            The example. If the information from the example
            is not required in the reasoning process, then this parameter will not have
            any effect.
        max_revision_num : int
            The upper limit on the number of revisions.
        require_more_revision : int
            If larger than 0, then after having found any candidates compatible with the
            knowledge base, continue to increase the number of labels to
            revise to get more possible compatible candidates.

        Returns
        -------
        Tuple[List[List[Any]], List[Any]]
            A tuple of two elements. The first element is a list of candidate revisions,
            i.e. revised pseudo-labels of the example. that are compatible with the knowledge
            base. The second element is a list of reasoning results corresponding to each
            candidate, i.e., the outcome of the ``logic_forward`` function.
        """
        candidates, reasoning_results = [], []
        for revision_num in range(len(pseudo_label) + 1):
            new_candidates, new_reasoning_results = self._revision(revision_num, pseudo_label, y, x)
            candidates.extend(new_candidates)
            reasoning_results.extend(new_reasoning_results)
            if len(candidates) > 0:
                min_revision_num = revision_num
                break
            if revision_num >= max_revision_num:
                return [], []

        for revision_num in range(
            min_revision_num + 1, min_revision_num + require_more_revision + 1
        ):
            if revision_num > max_revision_num:
                return candidates, reasoning_results
            new_candidates, new_reasoning_results = self._revision(revision_num, pseudo_label, y, x)
            candidates.extend(new_candidates)
            reasoning_results.extend(new_reasoning_results)
        return candidates, reasoning_results

    def __repr__(self):
        return (
            f"{self.__class__.__name__} is a KB with "
            f"pseudo_label_list={self.pseudo_label_list!r}, "
            f"max_err={self.max_err!r}, "
            f"use_cache={self.use_cache!r}."
        )


class PrologKB(KBBase):
    """
    Knowledge base provided by a Prolog (.pl) file.

    Parameters
    ----------
    pseudo_label_list : List[Any]
        Refer to class ``KBBase``.
    pl_file : str
        Prolog file containing the KB.

    Notes
    -----
    Users can instantiate this class to build their own knowledge base. During the
    instantiation, users are only required to provide the ``pseudo_label_list`` and ``pl_file``.
    To use the default logic forward and abductive reasoning methods in this class, in the
    Prolog (.pl) file, there needs to be a rule which is strictly formatted as
    ``logic_forward(Pseudo_labels, Res).``, e.g., ``logic_forward([A,B], C) :- C is A+B``.
    For specifics, refer to the ``logic_forward`` and ``get_query_string`` functions in this
    class. Users are also welcome to override related functions for more flexible support.
    """

    def __init__(self, pseudo_label_list: List[Any], pl_file: str):
        super().__init__(pseudo_label_list)

        try:
            import pyswip  # pylint: disable=import-outside-toplevel
        except (IndexError, ImportError):
            print(
                "A Prolog-based knowledge base is in use. Please install SWI-Prolog using the"
                + "command 'sudo apt-get install swi-prolog' for Linux users, or download it "
                + "following the guide in https://github.com/yuce/pyswip/blob/master/INSTALL.md "
                + "for Windows and Mac users."
            )

        self.prolog = pyswip.Prolog()
        self.pl_file = pl_file
        if not os.path.exists(self.pl_file):
            raise FileNotFoundError(f"The Prolog file {self.pl_file} does not exist.")
        self.prolog.consult(self.pl_file)

    def logic_forward(self, pseudo_label: List[Any], x: Optional[List[Any]] = None) -> Any:
        """
        Consult prolog with the query ``logic_forward(pseudo_labels, Res).``, and set the
        returned ``Res`` as the reasoning results. To use this default function, there must be
        a ``logic_forward`` method in the pl file to perform reasoning.
        Otherwise, users would override this function.

        Parameters
        ----------
        pseudo_label : List[Any]
            Pseudo-labels of an example.
        x : List[Any]
            The corresponding input example. If the information from the input
            is not required in the reasoning process, then this parameter will not have
            any effect.
        """
        result = list(self.prolog.query(f"solver_setup, qmpt0_solve(logic_forward({pseudo_label}, Res))."))[0]["Res"]
        if result == "true":
            return True
        if result == "false":
            return False
        return result

    def _revision_pseudo_label(
        self,
        pseudo_label: List[Any],
        revision_idx: List[int],
    ) -> List[Any]:
        import re  # pylint: disable=import-outside-toplevel

        revision_pseudo_label = pseudo_label.copy()
        revision_pseudo_label = flatten(revision_pseudo_label)

        for idx in revision_idx:
            revision_pseudo_label[idx] = "P" + str(idx)
        revision_pseudo_label = reform_list(revision_pseudo_label, pseudo_label)

        regex = r"'P\d+'"
        return re.sub(regex, lambda x: x.group().replace("'", ""), str(revision_pseudo_label))

    def get_query_string(
        self,
        pseudo_label: List[Any],
        y: Any,
        x: List[Any],  # pylint: disable=unused-argument
        revision_idx: List[int],
    ) -> str:
        """
        Get the query to be used for consulting Prolog.
        This is a default function for demo, users would override this function to adapt to
        their own Prolog file. In this demo function, return query
        ``logic_forward([kept_labels, Revise_labels], Res).``.

        Parameters
        ----------
        pseudo_label : List[Any]
            Pseudo-labels of an example (to be revised by abductive reasoning).
        y : Any
            Ground truth of the reasoning result for the example.
        x : List[Any]
            The corresponding input example. If the information from the input
            is not required in the reasoning process, then this parameter will not have
            any effect.
        revision_idx : List[int]
            A list specifying indices of where revisions should be made to the pseudo-labels.

        Returns
        -------
        str
            A string of the query.
        """

        query_string = "qmpt0_solve(logic_forward("
        query_string += self._revision_pseudo_label(pseudo_label, revision_idx)
        key_is_none_flag = y is None or (isinstance(y, list) and y[0] is None)

        query_string += f",{y}))." if not key_is_none_flag else "))."
        #print("check")
        return query_string

    def revise_at_idx(
        self,
        pseudo_label: List[Any],
        y: Any,
        x: List[Any],
        revision_idx: List[int],
    ) -> List[List[Any]]:
        """
        Revise the pseudo-labels at specified index positions by querying Prolog.

        Parameters
        ----------
        pseudo_label : List[Any]
            Pseudo-labels of an example (to be revised).
        y : Any
            Ground truth of the reasoning result for the example.
        x : List[Any]
            The corresponding input example. If the information from the input
            is not required in the reasoning process, then this parameter will not have
            any effect.
        revision_idx : List[int]
            A list specifying indices of where revisions should be made to the pseudo-labels.

        Returns
        -------
        Tuple[List[List[Any]], List[Any]]
            A tuple of two elements. The first element is a list of candidate revisions,
            i.e. revised pseudo-labels of the example. that are compatible with the knowledge
            base. The second element is a list of reasoning results corresponding to each
            candidate, i.e., the outcome of the ``logic_forward`` function.
        """
        candidates, reasoning_results = [], []
        query_string = self.get_query_string(pseudo_label, y, x, revision_idx)
        save_pseudo_label = pseudo_label
        pseudo_label = flatten(pseudo_label)
        abduce_c = [list(z.values()) for z in self.prolog.query(query_string)]
        for c in abduce_c:
            candidate = pseudo_label.copy()
            for i, idx in enumerate(revision_idx):
                candidate[idx] = c[i]
            candidate = reform_list(candidate, save_pseudo_label)
            candidates.append(candidate)
            reasoning_results.append(y)
        return candidates, reasoning_results

    def __repr__(self):
        return (
            f"{self.__class__.__name__} is a KB with "
            f"pseudo_label_list={self.pseudo_label_list!r}, "
            f"defined by "
            f"Prolog file {self.pl_file!r}."
        )
