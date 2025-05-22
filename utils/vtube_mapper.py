# utils/vtube_mapper.py

import logging
from typing import Dict, Any
from utils.shared_variables import SharedVariables

logger = logging.getLogger(__name__)

def map_metrics_to_vts_params(shared_vars: SharedVariables) -> Dict[str, Any]:
    """
    Map computed facial metrics from SharedVariables to the parameters expected by VTube Studio.

    Mappings:
      - Lip Sync Value -> "ParamMouthOpenY"
      - Eye Blink (derived from average EAR) -> "ParamEyesClosed"
      - Head Pose (yaw, pitch, roll) -> "ParamHeadYaw", "ParamHeadPitch", "ParamHeadRoll"

    Args:
        shared_vars (SharedVariables): Object containing computed facial metrics.

    Returns:
        Dict[str, Any]: Dictionary mapping VTube Studio parameter IDs to computed values.
    """
    params: Dict[str, Any] = {}

    # Map the lip sync value from detection (ensuring it is clamped to the range [0.0, 1.0]).
    if shared_vars.lip_sync_value is not None:
        params["ParamMouthOpenY"] = float(min(max(shared_vars.lip_sync_value, 0.0), 1.0))
        logger.debug(f"Mapped lip sync value: {params['ParamMouthOpenY']}")

    # Map eye blink intensity by averaging the two EAR values.
    # The blink parameter is computed as the inverse of the average EAR (since a lower EAR indicates a blink).
    if shared_vars.ear_left is not None and shared_vars.ear_right is not None:
        avg_ear = (shared_vars.ear_left + shared_vars.ear_right) / 2.0
        blink_value = 1.0 - float(min(max(avg_ear, 0.0), 1.0))
        params["ParamEyesClosed"] = blink_value
        logger.debug(f"Mapped blink value (inverted EAR): {params['ParamEyesClosed']}")

    # Map head pose values.
    if shared_vars.yaw is not None:
        params["ParamHeadYaw"] = float(shared_vars.yaw)
    if shared_vars.pitch is not None:
        params["ParamHeadPitch"] = float(shared_vars.pitch)
    if shared_vars.roll is not None:
        params["ParamHeadRoll"] = float(shared_vars.roll)

    logger.debug(f"VTube Studio parameter mapping: {params}")
    return params