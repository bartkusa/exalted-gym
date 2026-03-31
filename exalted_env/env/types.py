import numpy as np
from numpy.typing import NDArray
from typing import NewType, TypeAlias


PZAgentId = NewType("PZAgentId", str)
PZObsType: TypeAlias = NDArray[np.int32]
PZActionType: TypeAlias = int | None
