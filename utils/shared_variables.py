from dataclasses import dataclass, field
from typing import Optional

@dataclass
class SharedVariables:
    ear_left: Optional[float] = None
    ear_right: Optional[float] = None
    mar: Optional[float] = None
    ebr_left: Optional[float] = None
    ebr_right: Optional[float] = None
    lip_sync_value: Optional[float] = None
    yaw: Optional[float] = None
    pitch: Optional[float] = None
    roll: Optional[float] = None

    def reset(self) -> None:
        """
        Reset all metrics to None.
        """
        for field_name in self.__dataclass_fields__:
            setattr(self, field_name, None)
