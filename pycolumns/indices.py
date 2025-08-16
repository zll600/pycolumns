"""
this is not the index on a column, but set of indices
"""

from __future__ import annotations

from typing import Optional, Tuple, Union, Any, List
import numpy as np


class Indices(np.ndarray):
    """
    Represent indices returned by querying a column index.  This object
    inherits from normal numpy arrays, but behaves differently under the "&"
    and "|" operators.  These return the intersection or union of values in two
    Indices objects.

    Parameters
    ----------
    init_data: array or Indices
        The initial data
    copy: bool, optional
        If set to True, ensure a copy of the input data is made
    is_sorted: bool, optional
        If set to True, the input data are sorted
    is_checked: bool, optional
        If set to True, the input rows have been checked to be in bounds,
        and negatives are converted to positives.  This can be set later
        with the .is_checked setter

    Methods:
        The "&" and "|" operators are defined.

        array(): Return an ordinary np.ndarray view of the Indices.

    Examples:
        >>> i1=Indices([3,4,5])
        >>> i2=Indices([4,5,6])
        >>> (i1 & i2)
        Indices([4, 5])
        >>> (i1 | i2)
        Indices([3, 4, 5, 6])

    """

    def __new__(
        self,
        init_data: Union[np.ndarray, Indices, Any],
        copy: bool = False,
        is_sorted: bool = False,
        is_checked: bool = False,
    ) -> Indices:
        # always force i8 and native byte order since we send this to C code
        # Use np.asarray for NumPy 2.x compatibility instead of copy=False
        if copy:
            arr = np.array(init_data, dtype="i8")
        else:
            arr = np.asarray(init_data, dtype="i8")
        shape = arr.shape

        ret = np.ndarray.__new__(self, shape, arr.dtype, buffer=arr)

        self._is_sorted = is_sorted
        if arr.ndim == 0:
            self._is_sorted = True

        self._is_checked = is_checked

        return ret

    def get_minmax(self) -> Tuple[int, int]:
        if self.ndim == 0:
            mm = int(self), int(self)
        else:
            if self.is_sorted:
                imin, imax = 0, self.size - 1
            else:
                s = self.sort_index
                if s is not None:
                    imin, imax = s[0], s[-1]
                else:
                    # Fallback: if sort_index is None, find min/max directly
                    imin, imax = int(self.argmin()), int(self.argmax())

            mm = self[imin], self[imax]

        return mm

    @property
    def sort_index(self) -> Optional[np.ndarray]:
        """
        get an array that sorts the index
        """
        if self.is_sorted:
            return None
        else:
            if not hasattr(self, "_sort_index"):
                self._sort_index = self.argsort()  # type: ignore[assignment]
            return self._sort_index

    @property
    def is_sorted(self) -> bool:
        """
        returns True if sort has been run
        """
        return self._is_sorted

    @property
    def is_checked(self) -> bool:
        """
        returns True if the rows have been checked for negatives and
        fixed
        """
        return self._is_checked

    @is_checked.setter
    def is_checked(self, val: bool) -> None:
        """
        returns True if the rows have been checked for negatives and
        fixed
        """
        self._is_checked = val

    def sort(self, axis: int = -1, kind: Optional[str] = None, order: Optional[Union[str, List[str]]] = None) -> None:  # type: ignore[override]
        """
        sort and set the is_sorted flag
        """
        if not self.is_sorted:
            if self.ndim > 0:
                super(Indices, self).sort(axis=axis, kind=kind, order=order)  # type: ignore[misc]
            self._sort_index = None  # type: ignore[assignment]
            self._is_sorted = True

    def array(self) -> np.ndarray:
        return self.view(np.ndarray)

    def __and__(self, ind: Indices) -> Indices:  # type: ignore[override]
        # take the intersection
        if isinstance(ind, Indices):
            w = np.intersect1d(self, ind)
        else:
            raise ValueError("comparison index must be an Indices object")

        return Indices(w, is_sorted=True)

    def __or__(self, ind: Indices) -> Indices:  # type: ignore[override]
        # take the unique union
        if isinstance(ind, Indices):
            w = np.union1d(self, ind)
        else:
            raise ValueError("comparison index must be an Indices object")

        return Indices(w, is_sorted=True)

    def __repr__(self) -> str:
        arep = np.ndarray.__repr__(self)
        arep = arep.replace("array", "Indices")

        rep = [
            "Indices:",
            f"    size: {self.size}",
            f"    sorted: {self.is_sorted}",
            arep,
        ]
        return "\n".join(rep)
