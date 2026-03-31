import numpy as np
from numpy.typing import NDArray
from typing import NewType, TypeAlias


PZAgentId = NewType("PZAgentId", str)
PZObsType: TypeAlias = NDArray[np.float32]
PZActionType: TypeAlias = int | None


agent_red_1 = PZAgentId("agent_🔴_1")
agent_blue_1 = PZAgentId("agent_🟦_1")
